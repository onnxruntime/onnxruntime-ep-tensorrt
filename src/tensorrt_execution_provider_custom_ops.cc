// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "tensorrt_execution_provider_custom_ops.h"
#include "nv_includes.h"

#include <cstring>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace trt_ep {

/*
 * Create custom op domain list for TRT plugins.
 *
 * Collects all registered TRT plugins from the TRT registry and creates custom ops
 * with "trt.plugins" domain. Additionally, if users specify extra plugin libraries,
 * TRT EP will load them at runtime which will register those plugins to the TRT
 * plugin registry.
 *
 * Note: Current TRT plugin doesn't have APIs to get number of inputs/outputs of the plugin.
 * So, TensorRTCustomOp uses variadic inputs/outputs to pass ONNX graph validation.
 */
OrtStatus* CreateTensorRTCustomOpDomainList(const char* ep_name,
                                            const std::string& extra_plugin_lib_paths,
                                            std::vector<OrtCustomOpDomain*>& domain_list) {
  // Static storage for the custom op domain and custom ops.
  // These must persist for the process lifetime since ORT holds raw pointers to them.
  static std::unique_ptr<Ort::CustomOpDomain> custom_op_domain;
  static std::vector<std::unique_ptr<TensorRTCustomOp>> created_custom_op_list;
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);

  // If already initialized, just return the cached domain.
  if (custom_op_domain != nullptr) {
    domain_list.push_back(*custom_op_domain);
    return nullptr;
  }

  // Load any extra TRT plugin libraries if specified.
  // When the TRT plugin library is loaded, the global static object is created and the
  // plugin is registered to TRT registry. This is done through macro, for example,
  // REGISTER_TENSORRT_PLUGIN(VisionTransformerPluginCreator).
  // extra_plugin_lib_paths has the format of "path_1;path_2....;path_n"
  if (!extra_plugin_lib_paths.empty()) {
    std::stringstream extra_plugin_libs(extra_plugin_lib_paths);
    std::string lib;
    while (std::getline(extra_plugin_libs, lib, ';')) {
#ifdef _WIN32
      HMODULE handle = LoadLibraryA(lib.c_str());
      if (handle == nullptr) {
        // Log but don't fail - some plugins may be optional
      }
#else
      void* handle = dlopen(lib.c_str(), RTLD_NOW | RTLD_GLOBAL);
      if (handle == nullptr) {
        // Log but don't fail
      }
#endif
    }
  }

  try {
    // Initialize default TRT plugins
    initLibNvInferPlugins(nullptr, "");

    // Get all registered TRT plugins from registry
    int num_plugin_creator = 0;
    auto plugin_creators = getPluginRegistry()->getAllCreators(&num_plugin_creator);
    std::unordered_set<std::string> registered_plugin_names;

    custom_op_domain = std::make_unique<Ort::CustomOpDomain>("trt.plugins");

    for (int i = 0; i < num_plugin_creator; i++) {
      auto plugin_creator = plugin_creators[i];
      nvinfer1::AsciiChar const* plugin_name = nullptr;
      if (std::strcmp(plugin_creators[i]->getInterfaceInfo().kind, "PLUGIN CREATOR_V1") == 0) {
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)  // Ignore warning C4996: deprecated API
#endif
        auto plugin_creator_v1 = static_cast<nvinfer1::IPluginCreator const*>(plugin_creator);
        plugin_name = plugin_creator_v1->getPluginName();
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
      } else if (std::strcmp(plugin_creators[i]->getInterfaceInfo().kind, "PLUGIN CREATOR_V3ONE") == 0) {
        auto plugin_creator_v3 = static_cast<nvinfer1::IPluginCreatorV3One const*>(plugin_creator);
        plugin_name = plugin_creator_v3->getPluginName();
      } else {
        continue;  // Unknown plugin creator type, skip
      }

      // Each plugin may have different versions; we only register once per name
      if (registered_plugin_names.find(plugin_name) != registered_plugin_names.end()) {
        continue;
      }

      auto custom_op = std::make_unique<TensorRTCustomOp>(ep_name, nullptr);
      custom_op->SetName(plugin_name);
      custom_op_domain->Add(custom_op.get());
      created_custom_op_list.push_back(std::move(custom_op));
      registered_plugin_names.insert(plugin_name);
    }

    domain_list.push_back(*custom_op_domain);
  } catch (const std::exception&) {
    // Failed to get TRT plugins. The domain won't be added but this is not fatal.
    custom_op_domain.reset();
  }

  return nullptr;
}

}  // namespace trt_ep
