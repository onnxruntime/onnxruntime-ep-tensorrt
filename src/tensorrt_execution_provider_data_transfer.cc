// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "tensorrt_execution_provider_data_transfer.h"

#include <cuda_runtime_api.h>
#include <cassert>
#include <gsl/span>

namespace trt_ep {

void CUDA_RETURN_IF_ERROR(cudaError_t res);

/*static*/
bool ORT_API_CALL TRTEpDataTransfer::CanCopyImpl(const OrtDataTransferImpl* this_ptr,
                                                 const OrtMemoryDevice* src_memory_device,
                                                 const OrtMemoryDevice* dst_memory_device) noexcept {
  auto& impl = *static_cast<const TRTEpDataTransfer*>(this_ptr);

  // logic copied from GPUDataTransfer::CanCopy
  OrtMemoryInfoDeviceType src_type = impl.ep_api.MemoryDevice_GetDeviceType(src_memory_device);
  OrtMemoryInfoDeviceType dst_type = impl.ep_api.MemoryDevice_GetDeviceType(dst_memory_device);
  auto src_vendor_id = impl.ep_api.MemoryDevice_GetVendorId(src_memory_device);
  auto dst_vendor_id = impl.ep_api.MemoryDevice_GetVendorId(dst_memory_device);

  // Reject if GPU device is not NVIDIA
  if ((src_type == OrtMemoryInfoDeviceType_GPU && src_vendor_id != kNvidiaVendorId) ||
      (dst_type == OrtMemoryInfoDeviceType_GPU && dst_vendor_id != kNvidiaVendorId)) {
    return false;
  }

  // DLA uses pinned host memory, identified by the NVIDIA NPU vendor and
  // HOST_ACCESSIBLE memory type. Never claim transfers for foreign NPUs.
  const bool src_is_dla = src_type == OrtMemoryInfoDeviceType_NPU;
  const bool dst_is_dla = dst_type == OrtMemoryInfoDeviceType_NPU;
  if ((src_is_dla && (src_vendor_id != kNvidiaNpuVendorId ||
                     impl.ep_api.MemoryDevice_GetMemoryType(src_memory_device) != OrtDeviceMemoryType_HOST_ACCESSIBLE)) ||
      (dst_is_dla && (dst_vendor_id != kNvidiaNpuVendorId ||
                     impl.ep_api.MemoryDevice_GetMemoryType(dst_memory_device) != OrtDeviceMemoryType_HOST_ACCESSIBLE))) {
    return false;
  }

  // A validated DLA endpoint does not make a foreign endpoint host-accessible.
  const bool src_supported = src_type == OrtMemoryInfoDeviceType_CPU ||
                             src_type == OrtMemoryInfoDeviceType_GPU || src_is_dla;
  const bool dst_supported = dst_type == OrtMemoryInfoDeviceType_CPU ||
                             dst_type == OrtMemoryInfoDeviceType_GPU || dst_is_dla;
  if (!src_supported || !dst_supported) return false;

  // CPU-to-CPU copies are handled by ORT's CPU transfer implementation.
  return src_type == OrtMemoryInfoDeviceType_GPU || dst_type == OrtMemoryInfoDeviceType_GPU ||
         src_is_dla || dst_is_dla;
}

// function to copy one or more tensors.
// implementation can optionally use async copy if a stream is available for the input.
/*static*/
OrtStatus* ORT_API_CALL TRTEpDataTransfer::CopyTensorsImpl(OrtDataTransferImpl* this_ptr,
                                                           const OrtValue** src_tensors_ptr,
                                                           OrtValue** dst_tensors_ptr,
                                                           OrtSyncStream** streams_ptr,
                                                           size_t num_tensors) noexcept {
  auto& impl = *static_cast<TRTEpDataTransfer*>(this_ptr);

  auto src_tensors = gsl::make_span<const OrtValue*>(src_tensors_ptr, num_tensors);
  auto dst_tensors = gsl::make_span<OrtValue*>(dst_tensors_ptr, num_tensors);
  auto streams = gsl::make_span<OrtSyncStream*>(streams_ptr, num_tensors);

  for (size_t i = 0; i < num_tensors; ++i) {
    // NOTE: Stream support will be a separate PR. ignore teh streams_ptr values for now

    const OrtMemoryDevice* src_device = nullptr;
    const OrtMemoryDevice* dst_device = nullptr;
    src_device = impl.ep_api.Value_GetMemoryDevice(src_tensors[i]);
    dst_device = impl.ep_api.Value_GetMemoryDevice(dst_tensors[i]);

    // Enforce the same contract even when CopyTensors is called directly.
    if (!CanCopyImpl(this_ptr, src_device, dst_device)) {
      return impl.ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                      "TensorRT EP does not support this memory-device transfer.");
    }

    OrtMemoryInfoDeviceType src_device_type = impl.ep_api.MemoryDevice_GetDeviceType(src_device);
    OrtMemoryInfoDeviceType dst_device_type = impl.ep_api.MemoryDevice_GetDeviceType(dst_device);
    OrtDeviceMemoryType src_mem_type = impl.ep_api.MemoryDevice_GetMemoryType(src_device);
    OrtDeviceMemoryType dst_mem_type = impl.ep_api.MemoryDevice_GetMemoryType(dst_device);
    bool copy_involves_pinned_memory = src_mem_type == OrtDeviceMemoryType_HOST_ACCESSIBLE ||
                                       dst_mem_type == OrtDeviceMemoryType_HOST_ACCESSIBLE;

    const void* src_data = nullptr;
    void* dst_data = nullptr;
    RETURN_IF_ERROR(impl.ort_api.GetTensorData(src_tensors[i], &src_data));
    RETURN_IF_ERROR(impl.ort_api.GetTensorMutableData(dst_tensors[i], &dst_data));

    size_t bytes = 0;
    RETURN_IF_ERROR(impl.ort_api.GetTensorSizeInBytes(src_tensors[i], &bytes));

    // for the sync version of memcpy, launch to cuda default stream
    if (dst_device_type == OrtMemoryInfoDeviceType_GPU) {
      if (src_device_type == OrtMemoryInfoDeviceType_GPU) {
        // GPU -> GPU
        // Copy only if the two addresses are different and bytes > 0.
        if (dst_data != src_data && bytes > 0) {
          CUDA_RETURN_IF_ERROR(cudaMemcpy(dst_data, src_data, bytes, cudaMemcpyDeviceToDevice));
          // For device memory to device memory copy, no host-side synchronization is performed by cudaMemcpy.
          // see https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html
          CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(nullptr));
        }
      } else {
        // CPU -> GPU, this is blocking
        CUDA_RETURN_IF_ERROR(cudaMemcpy(dst_data, src_data, bytes, cudaMemcpyHostToDevice));
        if (src_mem_type != OrtDeviceMemoryType_HOST_ACCESSIBLE) {
          // For cudaMemcpy from pageable host memory to device memory, DMA to final destination may not have completed.
          // see https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html
          CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(nullptr));
        }
      }
    } else if (src_device_type == OrtMemoryInfoDeviceType_GPU) {
      // GPU -> CPU, this is blocking
      CUDA_RETURN_IF_ERROR(cudaMemcpy(dst_data, src_data, bytes, cudaMemcpyDeviceToHost));
    } else {
      // CPU <-> DLA and DLA <-> DLA use pinned host memory and a host memcpy.
      // ORT_ENFORCE(dst_data != src_data);
      memcpy(dst_data, src_data, bytes);
    }
  }

  return nullptr;
}

/*static*/
void ORT_API_CALL TRTEpDataTransfer::ReleaseImpl(OrtDataTransferImpl* this_ptr) noexcept {
  delete static_cast<TRTEpDataTransfer*>(this_ptr);
}
}  // namespace trt_ep
