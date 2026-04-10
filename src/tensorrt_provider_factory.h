#pragma once

#include "utils/ep_utils.h"
#include "tensorrt_execution_provider_data_transfer.h"
#include "cuda_allocator.h"

#include <mutex>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

using MemoryInfoUniquePtr = std::unique_ptr<OrtMemoryInfo, std::function<void(OrtMemoryInfo*)>>;

namespace trt_ep {

struct TensorrtExecutionProvider;

///
/// Plugin TensorRT EP factory that can create an OrtEp and return information about the supported hardware devices.
///
struct TensorrtExecutionProviderFactory : public OrtEpFactory, public ApiPtrs {
 public:
  TensorrtExecutionProviderFactory(const char* ep_name, const OrtLogger& default_logger, ApiPtrs apis);
  ~TensorrtExecutionProviderFactory();

  // Called by child OrtEp instances to retrieve the cached kernel registry for that EP.
  OrtStatus* GetKernelRegistryForEp(TensorrtExecutionProvider& ep, /*out*/ const OrtKernelRegistry** kernel_registry);

  const OrtMemoryInfo* GetMemoryInfoByOrdinal(int cuda_ordinal, bool is_pinned);

  // Keeps allocators per ep device in factory so they can be shared across sessions.
  std::unordered_map<uint32_t, std::unique_ptr<CUDAAllocator>> cuda_gpu_allocators;  // device id -> allocator
  std::unordered_map<uint32_t, std::unique_ptr<CUDAPinnedAllocator>> cuda_pinned_allocators;

 private:
  static const char* ORT_API_CALL GetNameImpl(const OrtEpFactory* this_ptr) noexcept;

  static const char* ORT_API_CALL GetVendorImpl(const OrtEpFactory* this_ptr) noexcept;

  static const char* ORT_API_CALL GetVersionImpl(const OrtEpFactory* this_ptr) noexcept;

  static OrtStatus* ORT_API_CALL GetSupportedDevicesImpl(OrtEpFactory* this_ptr,
                                                         const OrtHardwareDevice* const* devices, size_t num_devices,
                                                         OrtEpDevice** ep_devices, size_t max_ep_devices,
                                                         size_t* p_num_ep_devices) noexcept;

  static OrtStatus* ORT_API_CALL CreateEpImpl(OrtEpFactory* this_ptr, const OrtHardwareDevice* const* /*devices*/,
                                              const OrtKeyValuePairs* const* /*ep_metadata*/, size_t num_devices,
                                              const OrtSessionOptions* session_options, const OrtLogger* logger,
                                              OrtEp** ep) noexcept;

  static void ORT_API_CALL ReleaseEpImpl(OrtEpFactory* /*this_ptr*/, OrtEp* ep) noexcept;

  static OrtStatus* ORT_API_CALL CreateAllocatorImpl(OrtEpFactory* this_ptr, const OrtMemoryInfo* memory_info,
                                                     const OrtKeyValuePairs* /*allocator_options*/,
                                                     OrtAllocator** allocator) noexcept;

  static void ORT_API_CALL ReleaseAllocatorImpl(OrtEpFactory* /*this*/, OrtAllocator* allocator) noexcept;

  static OrtStatus* ORT_API_CALL CreateDataTransferImpl(OrtEpFactory* this_ptr,
                                                        OrtDataTransferImpl** data_transfer) noexcept;

  static bool ORT_API_CALL IsStreamAwareImpl(const OrtEpFactory* /*this_ptr*/) noexcept;

  const std::string ep_name_;              // EP name
  const std::string vendor_{"Nvidia"};     // EP vendor name
  const std::string ep_version_{"0.1.0"};  // EP version

  const OrtApi& ort_api_;
  const OrtEpApi& ep_api_;
  const OrtLogger& default_logger_;

  // Cached kernel registry used by all OrtEp instances created by this factory. Refer to OrtEp::GetKernelRegistry.
  //
  // Note: If this factory instead created EP instances that each supported different hardware configurations, then
  // the factory could cache a different kernel registry per EP configuration.
  OrtKernelRegistry* kernel_registry_ = nullptr;

  struct HardwareDeviceKey {
    OrtHardwareDeviceType type{ OrtHardwareDeviceType::OrtHardwareDeviceType_CPU };
    uint32_t vendor_id{ 0 };
    uint32_t device_id{ 0 };  // PCI device ID — identifies the hardware model, NOT a unique device
    int cuda_ordinal{ -1 };   // CUDA ordinal — unique per physical GPU on this host

    bool operator==(const HardwareDeviceKey& other) const noexcept {
      return type == other.type &&
             vendor_id == other.vendor_id &&
             device_id == other.device_id &&
             cuda_ordinal == other.cuda_ordinal;
    }
  };

  struct HardwareDeviceKeyHasher {
    size_t operator()(const HardwareDeviceKey& key) const noexcept {
      size_t hash = static_cast<size_t>(key.type);
      hash = (hash * 1315423911u) ^ static_cast<size_t>(key.vendor_id);
      hash = (hash * 1315423911u) ^ static_cast<size_t>(key.device_id);
      hash = (hash * 1315423911u) ^ static_cast<size_t>(key.cuda_ordinal);
      return hash;
    }
  };

  static HardwareDeviceKey MakeDeviceKey(const OrtApi& ort_api,
                                         const OrtHardwareDevice& device,
                                         int cuda_ordinal);

  struct DeviceCacheEntry {
    int cuda_device_id{ -1 };
    Ort::MemoryInfo device_memory_info{ nullptr };
    Ort::MemoryInfo pinned_memory_info{ nullptr };
  };

  // Per-physical-device cache. The key includes the CUDA ordinal to distinguish
  // identical GPUs (same PCI vendor/device ID) on multi-GPU hosts.
  std::mutex device_cache_mutex_;
  std::unordered_map<HardwareDeviceKey, DeviceCacheEntry, HardwareDeviceKeyHasher> device_cache_;

  // Ordinal-to-HardwareDeviceKey mapping built during GetSupportedDevicesImpl.
  std::unordered_map<int, HardwareDeviceKey> ordinal_to_device_key_;

  /// Find the DeviceCacheEntry for a given CUDA ordinal.
  /// Returns nullptr if the ordinal has not been registered.
  DeviceCacheEntry* FindDeviceCacheEntryByOrdinal(int cuda_ordinal);

  /// Same as FindDeviceCacheEntryByOrdinal but assumes device_cache_mutex_ is already held.
  DeviceCacheEntry* FindDeviceCacheEntryByOrdinalLocked(int cuda_ordinal);
};
}  // namespace trt_ep