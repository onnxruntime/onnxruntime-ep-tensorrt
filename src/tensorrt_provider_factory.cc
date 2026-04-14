#include "tensorrt_provider_factory.h"
#include "tensorrt_execution_provider.h"
#include "tensorrt_execution_provider_kernel_registration.h"
#include "cuda_allocator.h"

#include <gsl/gsl>
#include <cassert>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// TensorRT builder placeholder for test scenarios.
//
// TensorRT loads/unloads heavy internal libraries every time all IBuilder
// instances are destroyed. During unit testing (e.g., onnxruntime_provider_test)
// EPs are rapidly created and torn down, causing repeated overhead.
//
// ORT's test_main.cc has the same optimization behind `#ifdef USE_TENSORRT`,
// but that define is never set for plugin EPs. Instead we guard creation with
// an environment variable that the test harness can set:
//
//   set ORT_TRT_EP_ENABLE_BUILDER_PLACEHOLDER=1
//
// The placeholder is created once in CreateEpFactories() and destroyed in
// ReleaseEpFactory(), matching the factory's lifetime.
// ---------------------------------------------------------------------------
namespace {

class PlaceholderTrtLogger : public nvinfer1::ILogger {
 public:
  void log(Severity /*severity*/, const char* /*msg*/) noexcept override {}
};

PlaceholderTrtLogger g_placeholder_trt_logger;
std::unique_ptr<nvinfer1::IBuilder> g_trt_builder_placeholder;

void MaybeCreateBuilderPlaceholder() {
  if (g_trt_builder_placeholder) return;  // already created

  const char* env = std::getenv("ORT_TRT_EP_ENABLE_BUILDER_PLACEHOLDER");
  if (env != nullptr && std::string(env) == "1") {
    g_trt_builder_placeholder.reset(nvinfer1::createInferBuilder(g_placeholder_trt_logger));
  }
}

void DestroyBuilderPlaceholder() {
  g_trt_builder_placeholder.reset();
}

}  // namespace

namespace trt_ep {

TensorrtExecutionProviderFactory::TensorrtExecutionProviderFactory(const char* ep_name, const OrtLogger& default_logger, ApiPtrs apis)
    : OrtEpFactory {},
      ApiPtrs(apis),
      default_logger_{default_logger},
      ep_name_{ep_name},
      ort_api_{apis.ort_api},
      ep_api_{apis.ep_api} {
  ort_version_supported = ORT_API_VERSION;  // set to the ORT version we were compiled with.
  GetName = GetNameImpl;
  GetVendor = GetVendorImpl;
  GetVersion = GetVersionImpl;
  GetSupportedDevices = GetSupportedDevicesImpl;
  CreateEp = CreateEpImpl;
  ReleaseEp = ReleaseEpImpl;
  CreateAllocator = CreateAllocatorImpl;
  ReleaseAllocator = ReleaseAllocatorImpl;
  CreateDataTransfer = CreateDataTransferImpl;
  IsStreamAware = IsStreamAwareImpl; 
}

TensorrtExecutionProviderFactory::~TensorrtExecutionProviderFactory() {
  if (kernel_registry_ != nullptr) {
    ep_api_.ReleaseKernelRegistry(kernel_registry_);
  }
}

const char* ORT_API_CALL TensorrtExecutionProviderFactory::GetNameImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const TensorrtExecutionProviderFactory*>(this_ptr);
  return factory->ep_name_.c_str();
}

const char* ORT_API_CALL TensorrtExecutionProviderFactory::GetVendorImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const TensorrtExecutionProviderFactory*>(this_ptr);
  return factory->vendor_.c_str();
}

const char* ORT_API_CALL TensorrtExecutionProviderFactory::GetVersionImpl(const OrtEpFactory* this_ptr) noexcept {
  const auto* factory = static_cast<const TensorrtExecutionProviderFactory*>(this_ptr);
  return factory->ep_version_.c_str();
}

const OrtMemoryInfo* TensorrtExecutionProviderFactory::GetMemoryInfoByOrdinal(int cuda_ordinal, bool is_pinned) {
    // Get default OrtMemoryInfo from factory's device cache
    const OrtMemoryInfo* mem_info = nullptr;
    auto* cache_entry = FindDeviceCacheEntryByOrdinal(cuda_ordinal);
    if (cache_entry != nullptr) {
        mem_info = is_pinned ? cache_entry->pinned_memory_info :
                               cache_entry->device_memory_info; // Ort::MemoryInfo implicitly converts to OrtMemoryInfo*
    }
    return mem_info;
}

TensorrtExecutionProviderFactory::HardwareDeviceKey TensorrtExecutionProviderFactory::MakeDeviceKey(const OrtApi& ort_api,
    const OrtHardwareDevice& device,
    int cuda_ordinal) {
    return {
        ort_api.HardwareDevice_Type(&device),
        ort_api.HardwareDevice_VendorId(&device),
        ort_api.HardwareDevice_DeviceId(&device),
        cuda_ordinal,
    };
}

OrtStatus* ORT_API_CALL TensorrtExecutionProviderFactory::GetSupportedDevicesImpl(
    OrtEpFactory* this_ptr,
    const OrtHardwareDevice* const* devices,
    size_t num_devices,
    OrtEpDevice** ep_devices,
    size_t max_ep_devices,
    size_t* p_num_ep_devices) noexcept {
  size_t& num_ep_devices = *p_num_ep_devices;
  auto* factory = static_cast<TensorrtExecutionProviderFactory*>(this_ptr);

  // Clear stale ordinal mappings from any prior enumeration.
  {
    std::lock_guard<std::mutex> lock(factory->device_cache_mutex_);
    factory->ordinal_to_device_key_.clear();
  }

  auto release_ep_devices = [&](OrtStatus* status) -> OrtStatus* {
    for (size_t j = 0; j < num_ep_devices; ++j) {
      factory->ep_api.ReleaseEpDevice(ep_devices[j]);
      ep_devices[j] = nullptr;
    }
    num_ep_devices = 0;
    return status;
  };

  // Query CUDA device count once upfront so we can validate assigned ordinals.
  int cuda_device_count = 0;
  cudaError_t cuda_err = cudaGetDeviceCount(&cuda_device_count);
  if (cuda_err != cudaSuccess) {
    cuda_device_count = 0;  // no CUDA devices available
  }

  if (cuda_device_count == 0) {
    RETURN_IF_ERROR(factory->ort_api.Logger_LogMessage(&factory->default_logger_,
                    OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                    "No CUDA devices found on the system. No OrtEpDevice will be created and returned.",
                    ORT_FILE, __LINE__, __FUNCTION__));
  }

  int cuda_device_index_fallback = 0;  // fallback counter when metadata lacks PCI bus ID
  for (size_t i = 0; i < num_devices && num_ep_devices < max_ep_devices; ++i) {
    const OrtHardwareDevice& device = *devices[i];

    if (factory->ort_api.HardwareDevice_Type(&device) != OrtHardwareDeviceType::OrtHardwareDeviceType_GPU ||
        factory->ort_api.HardwareDevice_VendorId(&device) != kNvidiaVendorId) {
      continue;
    }

    // Try to resolve the CUDA ordinal from pci_bus_id metadata if available.
    // This is more reliable than counter-based ordinal assignment because it is
    // not affected by enumeration order, CUDA_VISIBLE_DEVICES remapping, or
    // mixed-vendor GPU configurations.
    int current_device_id = -1;
    const OrtKeyValuePairs* metadata = factory->ort_api_.HardwareDevice_Metadata(&device);
    if (metadata != nullptr) {
      const char* pci_bus_id = factory->ort_api_.GetKeyValue(metadata, "pci_bus_id");
      if (pci_bus_id != nullptr && pci_bus_id[0] != '\0') {
        int resolved_ordinal = -1;
        cudaError_t err = cudaDeviceGetByPCIBusId(&resolved_ordinal, pci_bus_id);
        if (err == cudaSuccess && resolved_ordinal >= 0 && resolved_ordinal < cuda_device_count) {
          current_device_id = resolved_ordinal;
        }
      }
    }

    // Fallback: if pci_bus_id was not available, use counter-based ordinal assignment.
    if (current_device_id < 0) {
      current_device_id = cuda_device_index_fallback++;
    }

    // Validate the assigned ordinal is within the range of CUDA-visible devices.
    // If hardware enumeration reports GPUs not visible to CUDA (e.g. due to
    // CUDA_VISIBLE_DEVICES), skip them to avoid failures in allocator/stream creation.
    if (current_device_id >= cuda_device_count) {
      continue;
    }

    const auto device_key = MakeDeviceKey(factory->ort_api, device, current_device_id);
    DeviceCacheEntry* cache_entry = nullptr;
    {
      std::lock_guard<std::mutex> lock(factory->device_cache_mutex_);
      auto [it, inserted] = factory->device_cache_.try_emplace(device_key);
      if (inserted) {
        it->second.cuda_device_id = current_device_id;
        it->second.device_memory_info = Ort::MemoryInfo{"Cuda",
                                                        OrtMemoryInfoDeviceType_GPU,
                                                        kNvidiaVendorId,
                                                        static_cast<uint32_t>(current_device_id),
                                                        OrtDeviceMemoryType_DEFAULT,
                                                        /*alignment is default*/ 0,
                                                        OrtAllocatorType::OrtDeviceAllocator};
        it->second.pinned_memory_info = Ort::MemoryInfo{"CudaPinned",
                                                        OrtMemoryInfoDeviceType_GPU,
                                                        kNvidiaVendorId,
                                                        static_cast<uint32_t>(current_device_id),
                                                        OrtDeviceMemoryType_HOST_ACCESSIBLE,
                                                        /*alignment is default*/ 0,
                                                        OrtAllocatorType::OrtDeviceAllocator};
      }

      cache_entry = &it->second;
      current_device_id = cache_entry->cuda_device_id;
      // Build ordinal -> key mapping for CreateAllocatorImpl lookups.
      factory->ordinal_to_device_key_[current_device_id] = device_key;
    }

    // These can be returned as nullptr if EP has nothing to add.
    OrtKeyValuePairs* ep_metadata = nullptr;
    OrtKeyValuePairs* ep_options = nullptr;
    factory->ort_api.CreateKeyValuePairs(&ep_metadata);
    factory->ort_api.CreateKeyValuePairs(&ep_options);
    factory->ort_api.AddKeyValuePair(ep_metadata, "cuda_device_id", std::to_string(current_device_id).c_str());
    factory->ort_api.AddKeyValuePair(ep_options, "device_id", std::to_string(current_device_id).c_str());

    // Get CUDA device properties for metadata
    {
      cudaDeviceProp prop;
      if (cudaGetDeviceProperties(&prop, current_device_id) == cudaSuccess) {
        factory->ort_api.AddKeyValuePair(ep_metadata, "cuda_device_name", prop.name);
        factory->ort_api.AddKeyValuePair(ep_metadata, "cuda_compute_capability",
                                         (std::to_string(prop.major) + "." + std::to_string(prop.minor)).c_str());
      }
    }

    // OrtEpDevice copies ep_metadata and ep_options.
    OrtEpDevice* ep_device = nullptr;
    auto* status = factory->ort_api.GetEpApi()->CreateEpDevice(factory, &device, ep_metadata, ep_options, &ep_device);

    factory->ort_api.ReleaseKeyValuePairs(ep_metadata);
    factory->ort_api.ReleaseKeyValuePairs(ep_options);

    if (status != nullptr) {
      return release_ep_devices(status);
    }

    auto release_current_ep_device = [factory](OrtEpDevice* device) {
      factory->ep_api.ReleaseEpDevice(device);
    };

    // ep_device_guard owns the current device. On error, release_ep_devices cleans up
    // previously committed devices [0, num_ep_devices), while the guard cleans up this one.
    std::unique_ptr<OrtEpDevice, decltype(release_current_ep_device)> ep_device_guard(ep_device, release_current_ep_device);

    // Register allocator info for GPU device memory
    status = factory->ep_api.EpDevice_AddAllocatorInfo(ep_device, cache_entry->device_memory_info);
    if (status != nullptr) {
      return release_ep_devices(status);
    }

    // Register allocator info for pinned host memory associated with the
    // same CUDA ordinal as the device allocator above.
    status = factory->ep_api.EpDevice_AddAllocatorInfo(ep_device, cache_entry->pinned_memory_info);
    if (status != nullptr) {
      return release_ep_devices(status);
    }

    ep_devices[num_ep_devices++] = ep_device_guard.release();
  }

  return nullptr;
}

OrtStatus* ORT_API_CALL TensorrtExecutionProviderFactory::CreateEpImpl(
    OrtEpFactory* this_ptr,
    _In_reads_(num_devices) const OrtHardwareDevice* const* /*devices*/,
    _In_reads_(num_devices) const OrtKeyValuePairs* const* /*ep_metadata*/,
    _In_ size_t num_devices,
    _In_ const OrtSessionOptions* session_options,
    _In_ const OrtLogger* logger, _Out_ OrtEp** ep) noexcept {
  auto* factory = static_cast<TensorrtExecutionProviderFactory*>(this_ptr);
  *ep = nullptr;

  if (num_devices != 1) {
    // we only registered for GPU and only expected to be selected for one GPU
    // if you register for multiple devices (e.g. CPU, GPU and maybe NPU) you will get an entry for each device
    // the EP has been selected for.
    return factory->ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                         "TensorRT EP only supports selection for one device.");
  }

  // Create the execution provider
  RETURN_IF_ERROR(factory->ort_api.Logger_LogMessage(logger,
                                                     OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
                                                     "Creating TensorRT EP", ORT_FILE, __LINE__, __FUNCTION__));

  // use properties from the device and ep_metadata if needed
  // const OrtHardwareDevice* device = devices[0];
  // const OrtKeyValuePairs* ep_metadata = ep_metadata[0];

  auto trt_ep = std::make_unique<TensorrtExecutionProvider>(*factory, factory->ep_name_, *session_options, *logger);

  *ep = trt_ep.release();
  return nullptr;
}

void ORT_API_CALL TensorrtExecutionProviderFactory::ReleaseEpImpl(OrtEpFactory* /*this_ptr*/, OrtEp* ep) noexcept {
  TensorrtExecutionProvider* trt_ep = static_cast<TensorrtExecutionProvider*>(ep);
  delete trt_ep;
}

OrtStatus* ORT_API_CALL TensorrtExecutionProviderFactory::CreateAllocatorImpl(OrtEpFactory* this_ptr,
                                                                              const OrtMemoryInfo* memory_info,
                                                                              const OrtKeyValuePairs* /*allocator_options*/,
                                                                              OrtAllocator** allocator) noexcept {
  auto& factory = *static_cast<TensorrtExecutionProviderFactory*>(this_ptr);

  // NOTE: The factory implementation is free to return a shared OrtAllocator* instance instead of creating a new
  //       allocator on each call. To do this have an allocator instance as an OrtEpFactory class member and make
  //       ReleaseAllocatorImpl a no-op.

  // NOTE: EP should implement its own arena logic. ep_arena.cc/h is provided as a reference and we use it here for
  //       device memory. `allocator_options` can be used for arena configuration and there is a helper in ep_arena.h
  //       to convert from OrtKeyValuePairs to the same arena config settings that ORT uses.
  //       You are of course free to have completely different settings.

  const OrtMemoryDevice* mem_device = factory.ep_api.MemoryInfo_GetMemoryDevice(memory_info);
  uint32_t device_id = factory.ep_api.MemoryDevice_GetDeviceId(mem_device);

  if (factory.ep_api.MemoryDevice_GetMemoryType(mem_device) == OrtDeviceMemoryType_DEFAULT) {
    // use the one that previously created
    if (factory.cuda_gpu_allocators.find(device_id) != factory.cuda_gpu_allocators.end()) {
      *allocator = factory.cuda_gpu_allocators[device_id].get();
      return nullptr;
    }

    // create a CUDA allocator
    auto cuda_allocator = std::make_unique<CUDAAllocator>(memory_info, static_cast<DeviceId>(device_id));

    *allocator = cuda_allocator.get();
    factory.cuda_gpu_allocators[device_id] = std::move(cuda_allocator);

  } else if (factory.ep_api.MemoryDevice_GetMemoryType(mem_device) == OrtDeviceMemoryType_HOST_ACCESSIBLE) {
    // use the one that previously created
    if (factory.cuda_pinned_allocators.find(device_id) != factory.cuda_pinned_allocators.end()) {
      *allocator = factory.cuda_pinned_allocators[device_id].get();
      return nullptr;
    }

    // create a CUDA PINNED allocator
    auto cuda_pinned_allocator = std::make_unique<CUDAPinnedAllocator>(memory_info);

    *allocator = cuda_pinned_allocator.get();
    factory.cuda_pinned_allocators[device_id] = std::move(cuda_pinned_allocator);

  } else {
    return factory.ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                        "INTERNAL ERROR! Unknown memory info provided to CreateAllocator. "
                                        "Value did not come directly from an OrtEpDevice returned by this factory.");
  }

  return nullptr;
}

void ORT_API_CALL TensorrtExecutionProviderFactory::ReleaseAllocatorImpl(OrtEpFactory* /*this*/,
                                                                         OrtAllocator* allocator) noexcept {
  // no-op. The allocators will be shared across sessions.
  // delete static_cast<CUDAAllocator*>(allocator);
}

OrtStatus* ORT_API_CALL TensorrtExecutionProviderFactory::CreateDataTransferImpl(
    OrtEpFactory* this_ptr,
    OrtDataTransferImpl** data_transfer) noexcept {
  auto& factory = *static_cast<TensorrtExecutionProviderFactory*>(this_ptr);

  auto data_transfer_impl = std::make_unique<TRTEpDataTransfer>(static_cast<const ApiPtrs&>(factory));
  *data_transfer = data_transfer_impl.release();

  return nullptr;
}

bool ORT_API_CALL TensorrtExecutionProviderFactory::IsStreamAwareImpl(const OrtEpFactory* /*this_ptr*/) noexcept {
  return true;
}

OrtStatus* TensorrtExecutionProviderFactory::GetKernelRegistryForEp(TensorrtExecutionProvider& ep,
                                                                    const OrtKernelRegistry** out_kernel_registry) {
  *out_kernel_registry = nullptr;

  if (GetNumKernels() == 0) {
    return nullptr;
  }

  if (kernel_registry_ == nullptr) {
    const char* ep_name = ep.GetName(static_cast<const OrtEp*>(&ep));

    // This statement creates the kernel registry and caches it in the OrtEpFactory instance.
    // We assume that all EPs created by this factory can use the same kernel registry. This may not be the
    // case in a more complex OrtEpFactory that can create EP instances that are each configured for different
    // hardware devices. In such a scenario, a different kernel registry may be created for each EP configuration.
    RETURN_IF_ERROR(CreateKernelRegistry(ep_name, nullptr, &kernel_registry_));
  }

  *out_kernel_registry = kernel_registry_;
  return nullptr;
}

TensorrtExecutionProviderFactory::DeviceCacheEntry* TensorrtExecutionProviderFactory::FindDeviceCacheEntryByOrdinalLocked(int cuda_ordinal) {
    auto key_it = ordinal_to_device_key_.find(cuda_ordinal);
    if (key_it == ordinal_to_device_key_.end()) {
        return nullptr;
    }
    auto cache_it = device_cache_.find(key_it->second);
    if (cache_it == device_cache_.end()) {
        return nullptr;
    }
    return &cache_it->second;
}

// IMPORTANT: Entries are never erased from device_cache_ after insertion.
// This guarantees pointer stability for DeviceCacheEntry* returned by
// FindDeviceCacheEntryByOrdinal() after the lock is released.
TensorrtExecutionProviderFactory::DeviceCacheEntry* TensorrtExecutionProviderFactory::FindDeviceCacheEntryByOrdinal(int cuda_ordinal) {
    std::lock_guard<std::mutex> lock(device_cache_mutex_);
    return FindDeviceCacheEntryByOrdinalLocked(cuda_ordinal);
}

}  // namespace trt_ep

#define EXPORT_SYMBOL

extern "C" {
//
// Public symbols
//
EXPORT_SYMBOL OrtStatus* CreateEpFactories(const char* registration_name, const OrtApiBase* ort_api_base,
                                           const OrtLogger* default_logger,
                                           OrtEpFactory** factories, size_t max_factories, size_t* num_factories) {
  const OrtApi* ort_api = ort_api_base->GetApi(ORT_API_VERSION);
  const OrtEpApi* ort_ep_api = ort_api->GetEpApi();
  const OrtModelEditorApi* model_editor_api = ort_api->GetModelEditorApi();

  // Manual init for the C++ API
  Ort::InitApi(ort_api);

  int cuda_device_count = 0;
  const cudaError_t cuda_err = cudaGetDeviceCount(&cuda_device_count);
  if (cuda_err != cudaSuccess) {
    cuda_device_count = 0;  // no CUDA devices available
  }

  if (cuda_device_count == 0) {
    RETURN_IF_ERROR(ort_api->Logger_LogMessage(default_logger,
                    OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                    "No CUDA devices found on the system."
                    "TensorRT execution provider will still be created but will not be able to run any models.",
                    ORT_FILE, __LINE__, __FUNCTION__));
  }

  // Create TRT builder placeholder if running under a test harness.
  // This prevents TensorRT from repeatedly loading/unloading internal
  // libraries as EP instances are created and destroyed across tests.
  MaybeCreateBuilderPlaceholder();

  // Factory could use registration_name or define its own EP name.
  std::unique_ptr<OrtEpFactory> factory = std::make_unique<trt_ep::TensorrtExecutionProviderFactory>(registration_name, *default_logger, ApiPtrs{*ort_api, *ort_ep_api, *model_editor_api});

  if (max_factories < 1) {
    return ort_api->CreateStatus(ORT_INVALID_ARGUMENT,
                                 "Not enough space to return EP factory. Need at least one.");
  }

  factories[0] = factory.release();
  *num_factories = 1;

  return nullptr;
}

EXPORT_SYMBOL OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) {
  delete static_cast<trt_ep::TensorrtExecutionProviderFactory*>(factory);

  // Release the placeholder builder when the last factory is torn down.
  DestroyBuilderPlaceholder();

  return nullptr;
}

}  // extern "C"
