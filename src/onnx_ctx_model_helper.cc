// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include <fstream>
#include <filesystem>
#include <cstring>

#include "utils/ep_utils.h"
#include "utils/path_string.h"
#include "onnx_ctx_model_helper.h"
#include "onnx/onnx_pb.h"

namespace trt_ep {
extern TensorrtLogger& GetTensorrtLogger(bool verbose_log, const OrtLogger& ort_default_logger,
                                         const OrtApi* ort_api);

bool IsAbsolutePath(const std::string& path_string) {
  if (path_string.empty()) {
    return false;
  }

  const auto path = std::filesystem::u8path(path_string);
  return path.is_absolute();
}

bool IsRelativePathToParentPath(const std::string& path_string) {
  if (path_string.empty())
    return false;

  auto path = std::filesystem::u8path(path_string);

  // Normalize things like "a/../b" or "foo//bar/.."
  path = path.lexically_normal();

  // Check each path component
  for (const auto& part : path) {
    if (part == "..") {
      return true;
    }
  }
  return false;
}

/*
 * Return the directory where the ep context model locates
 */
std::filesystem::path GetPathOrParentPathOfCtxModel(const std::string& ep_context_file_path) {
  if (ep_context_file_path.empty()) {
    return std::filesystem::path();
  }
  const auto ctx_path = std::filesystem::u8path(ep_context_file_path);
  if (std::filesystem::is_directory(ctx_path)) {
    return ctx_path;
  } else {
    return ctx_path.parent_path();
  }
}

bool IsWeightStrippedEngineCache(std::filesystem::path& engine_cache_path) {
  // The weight-stripped engine cache has the naming of xxx.stripped.engine
  return engine_cache_path.stem().extension().u8string() == ".stripped";
}

/*
 * Create an EPContext OrtNode from a fused_node
 */
OrtStatus* EPContextNodeHelper::CreateEPContextNode(const std::string& engine_cache_path,
                                                    char* engine_data,
                                                    size_t size,
                                                    const int64_t embed_mode,
                                                    const std::string& compute_capability,
                                                    const std::string& onnx_model_path,
                                                    int trt_version,
                                                    OrtNode** ep_context_node) {
  // Helper to collect input or output names from an array of OrtValueInfo instances.
  auto collect_input_output_names = [&](gsl::span<const OrtValueInfo* const> value_infos,
                                        std::vector<const char*>& result) -> OrtStatus* {
    size_t num_values = value_infos.size();
    std::vector<const char*> value_names(num_values);

    for (size_t i = 0; i < num_values; ++i) {
      const OrtValueInfo* value_info = value_infos[i];
      RETURN_IF_ERROR(ort_api.GetValueInfoName(value_info, &value_names[i]));
    }

    result = std::move(value_names);
    return nullptr;
  };

  const char* fused_node_name = nullptr;

  RETURN_IF_ERROR(ort_api.Node_GetName(fused_node_, &fused_node_name));

  size_t num_fused_node_inputs = 0;
  size_t num_fused_node_outputs = 0;
  RETURN_IF_ERROR(ort_api.Node_GetNumInputs(fused_node_, &num_fused_node_inputs));
  RETURN_IF_ERROR(ort_api.Node_GetNumOutputs(fused_node_, &num_fused_node_outputs));

  std::vector<const OrtValueInfo*> fused_node_inputs(num_fused_node_inputs);
  std::vector<const OrtValueInfo*> fused_node_outputs(num_fused_node_outputs);
  RETURN_IF_ERROR(ort_api.Node_GetInputs(fused_node_, fused_node_inputs.data(), fused_node_inputs.size()));
  RETURN_IF_ERROR(ort_api.Node_GetOutputs(fused_node_, fused_node_outputs.data(), fused_node_outputs.size()));

  std::vector<const char*> input_names;
  std::vector<const char*> output_names;

  RETURN_IF_ERROR(collect_input_output_names(fused_node_inputs, /*out*/ input_names));
  RETURN_IF_ERROR(collect_input_output_names(fused_node_outputs, /*out*/ output_names));

  // Create node attributes. The CreateNode() function copies the attributes, so we have to release them.
  std::array<OrtOpAttr*, 8> attributes = {};
  DeferOrtRelease<OrtOpAttr> defer_release_attrs(attributes.data(), attributes.size(), ort_api.ReleaseOpAttr);

  RETURN_IF_ERROR(ort_api.CreateOpAttr("embed_mode", &embed_mode, sizeof(int64_t), ORT_OP_ATTR_INT, &attributes[0]));

  RETURN_IF_NOT(embed_mode == 0 || embed_mode == 1, "EPContext embed_mode must be 0 or 1.");
  RETURN_IF_NOT(!embed_mode || (engine_data != nullptr && size > 0), "EPContext engine data is empty.");
  std::string engine_data_str = "";
  if (embed_mode) {
    if (size > 0) {
      engine_data_str.assign(engine_data, size);
    }
    RETURN_IF_ERROR(
        ort_api.CreateOpAttr("ep_cache_context", engine_data_str.c_str(), engine_data_str.size(), ORT_OP_ATTR_STRING, &attributes[1]));
  } else {
    RETURN_IF_ERROR(ort_api.CreateOpAttr("ep_cache_context", engine_cache_path.c_str(), engine_cache_path.size(), ORT_OP_ATTR_STRING, &attributes[1]));
  }

  RETURN_IF_ERROR(ort_api.CreateOpAttr("hardware_architecture", compute_capability.c_str(), compute_capability.size(),
                                      ORT_OP_ATTR_STRING, &attributes[2]));
  const std::string onnx_model_filename = std::filesystem::u8path(onnx_model_path).filename().u8string();
  RETURN_IF_ERROR(ort_api.CreateOpAttr("onnx_model_filename", onnx_model_filename.c_str(), onnx_model_filename.size(),
                                      ORT_OP_ATTR_STRING, &attributes[3]));

  const int64_t main_context = 1;  // Each partition contains an independent engine.
  RETURN_IF_ERROR(ort_api.CreateOpAttr("main_context", &main_context, sizeof(main_context), ORT_OP_ATTR_INT, &attributes[4]));
  RETURN_IF_ERROR(ort_api.CreateOpAttr("partition_name", fused_node_name, std::strlen(fused_node_name),
                                      ORT_OP_ATTR_STRING, &attributes[5]));
  const std::string sdk_version = std::to_string(trt_version);
  RETURN_IF_ERROR(ort_api.CreateOpAttr("ep_sdk_version", sdk_version.c_str(), sdk_version.size(), ORT_OP_ATTR_STRING, &attributes[6]));
  const std::string source = "TensorrtExecutionProvider";
  RETURN_IF_ERROR(ort_api.CreateOpAttr("source", source.c_str(), source.size(), ORT_OP_ATTR_STRING, &attributes[7]));

  RETURN_IF_ERROR(model_editor_api.CreateNode("EPContext", "com.microsoft", fused_node_name, input_names.data(),
                                              input_names.size(), output_names.data(), output_names.size(),
                                              attributes.data(), attributes.size(), ep_context_node));

  return nullptr;
}

// Identifies EPContext nodes that this EP can claim and deserialize. GetCapability
// checks each node before bypassing the TensorRT ONNX parser; compilation and
// context loading reuse the same check to avoid consuming another EP's context.
// A matching node has domain "com.microsoft" and source "TensorrtExecutionProvider".
// Missing or empty source attributes are accepted for legacy TensorRT models.
// Returns nullptr when the check succeeds, with the result in is_context_node;
// returns an error status for ORT API failures or malformed source metadata.
OrtStatus* EPContextNodeReader::IsTensorRTContextNode(const OrtNode* node, const OrtApi& ort_api,
                                                       bool& is_context_node) {
  is_context_node = false;
  const char* op_type = nullptr;
  const char* domain = nullptr;
  RETURN_IF_ERROR(ort_api.Node_GetOperatorType(node, &op_type));
  RETURN_IF_ERROR(ort_api.Node_GetDomain(node, &domain));
  if (std::strcmp(op_type, "EPContext") != 0 || std::strcmp(domain, "com.microsoft") != 0) return nullptr;

  const OrtOpAttr* source_attr = nullptr;
  OrtStatus* status = ort_api.Node_GetAttributeByName(node, "source", &source_attr);
  if (status != nullptr) {
    // Only a missing attribute is a legacy case; propagate other lookup failures.
    if (ort_api.GetErrorCode(status) != ORT_NOT_FOUND) return status;
    ort_api.ReleaseStatus(status);
  }
  // Older TensorRT context models did not write a source attribute.
  if (source_attr == nullptr) {
    is_context_node = true;
    return nullptr;
  }

  OrtOpAttrType type;
  RETURN_IF_ERROR(ort_api.OpAttr_GetType(source_attr, &type));
  RETURN_IF_NOT(type == ORT_OP_ATTR_STRING, "EPContext source must be a string.");
  std::string source;
  RETURN_IF_ERROR(Ort::ConstOpAttr(source_attr).GetValue(source));
  is_context_node = source.empty() || source == "TensorrtExecutionProvider";
  return nullptr;
}

// Finds whether a fused graph contains a TensorRT context, so CompileImpl can
// choose engine deserialization instead of building an engine from ONNX nodes.
OrtStatus* EPContextNodeReader::GraphHasCtxNode(const OrtGraph* graph, const OrtApi& ort_api,
                                               bool& has_context_node) {
  has_context_node = false;
  size_t num_nodes = 0;
  RETURN_IF_ERROR(ort_api.Graph_GetNumNodes(graph, &num_nodes));
  std::vector<const OrtNode*> nodes(num_nodes);
  RETURN_IF_ERROR(ort_api.Graph_GetNodes(graph, nodes.data(), nodes.size()));
  for (const auto* node : nodes) {
    if (node == nullptr) continue;
    bool is_context_node = false;
    RETURN_IF_ERROR(IsTensorRTContextNode(node, ort_api, is_context_node));
    if (is_context_node) {
      has_context_node = true;
      break;
    }
  }
  return nullptr;
}

/*
 * The sanity check for EP context contrib op.
 */
OrtStatus* EPContextNodeReader::ValidateEPCtxNode(const OrtGraph* graph) const {
  size_t num_nodes = 0;
  RETURN_IF_ERROR(ort_api.Graph_GetNumNodes(graph, &num_nodes));
  RETURN_IF_NOT(num_nodes == 1, "Graph contains more than one node.");

  std::vector<const OrtNode*> nodes(num_nodes);
  RETURN_IF_ERROR(ort_api.Graph_GetNodes(graph, nodes.data(), nodes.size()));

  bool is_context_node = false;
  RETURN_IF_ERROR(IsTensorRTContextNode(nodes[0], ort_api, is_context_node));
  RETURN_IF_NOT(is_context_node, "Node is not a TensorRT EPContext node.");

  // TODO: Check compute capability and others

  return nullptr;
}

OrtStatus* EPContextNodeReader::GetEpContextFromGraph(const OrtGraph& graph) {
  RETURN_IF_ERROR(ValidateEPCtxNode(&graph));

  size_t num_nodes = 0;
  RETURN_IF_ERROR(ort_api.Graph_GetNumNodes(&graph, &num_nodes));

  auto ort_graph = Ort::ConstGraph(&graph);
  std::vector<Ort::ConstNode> nodes(num_nodes);
  nodes = ort_graph.GetNodes();

  // ValidateEPCtxNode() already checked ENFORCE(num_nodes == 1)
  auto& node = nodes[0];
  Ort::ConstOpAttr node_attr;

  // Get "embed_mode" attribute
  RETURN_IF_ERROR(node.GetAttributeByName("embed_mode", node_attr));
  RETURN_IF_NOT(node_attr.GetType() == OrtOpAttrType::ORT_OP_ATTR_INT, "\'embed_mode\' attribute should be integer type.");

  int64_t embed_mode = 0;
  RETURN_IF_ERROR(node_attr.GetValue(embed_mode));
  RETURN_IF_NOT(embed_mode == 0 || embed_mode == 1, "EPContext embed_mode must be 0 or 1.");

  // Only make path checks if model not provided as byte buffer
  bool make_secure_path_checks = !ort_graph.GetModelPath().empty();

  if (embed_mode) {
    // Get engine from byte stream.
    RETURN_IF_ERROR(node.GetAttributeByName("ep_cache_context", node_attr));
    RETURN_IF_NOT(node_attr.GetType() == OrtOpAttrType::ORT_OP_ATTR_STRING, "\'ep_cache_context\' attribute should be string type.");

    std::string context_binary;
    RETURN_IF_ERROR(node_attr.GetValue<std::string>(context_binary));
    RETURN_IF_NOT(!context_binary.empty(), "EPContext engine data is empty.");

    *(trt_engine_) = std::unique_ptr<nvinfer1::ICudaEngine>(trt_runtime_->deserializeCudaEngine(const_cast<char*>(context_binary.c_str()),
                                                                                                static_cast<size_t>(context_binary.length())));

    std::string message = "[TensorRT EP] Read engine as binary data from \"ep_cache_context\" attribute of ep context node and deserialized it";
    Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger_,
                                                OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
    if (!(*trt_engine_)) {
      return ort_api.CreateStatus(ORT_EP_FAIL, "TensorRT EP could not deserialize engine from binary data");
    }

    if (weight_stripped_engine_refit_) {
      RETURN_IF_ERROR(node.GetAttributeByName("onnx_model_filename", node_attr));
      RETURN_IF_NOT(node_attr.GetType() == OrtOpAttrType::ORT_OP_ATTR_STRING, "\'onnx_model_filename\' attribute should be string type.");
      std::string onnx_model_filename;
      RETURN_IF_ERROR(node_attr.GetValue<std::string>(onnx_model_filename));
      std::string placeholder;
      RETURN_IF_ERROR(ep_.RefitEngine(onnx_model_filename,
                                      onnx_model_folder_path_,
                                      placeholder,
                                      make_secure_path_checks,
                                      onnx_model_bytestream_,
                                      onnx_model_bytestream_size_,
                                      onnx_external_data_bytestream_,
                                      onnx_external_data_bytestream_size_,
                                      (*trt_engine_).get(),
                                      false,  // serialize refitted engine to disk
                                      detailed_build_log_));
    }
  } else {
    // Get engine from cache file.
    RETURN_IF_ERROR(node.GetAttributeByName("ep_cache_context", node_attr));
    RETURN_IF_NOT(node_attr.GetType() == OrtOpAttrType::ORT_OP_ATTR_STRING, "\'ep_cache_context\' attribute should be string type.");
    std::string cache_path;
    RETURN_IF_ERROR(node_attr.GetValue<std::string>(cache_path));

    // For security purpose, in the case of running context model, TRT EP won't allow
    // engine cache path to be the relative path like "../file_path" or the absolute path.
    // It only allows the engine cache to be in the same directory or sub directory of the context model.
    if (IsAbsolutePath(cache_path)) {
      std::string message = "For security purpose, the ep_cache_context attribute should be set with a relative path, but it is an absolute path:  " + cache_path;
      return ort_api.CreateStatus(ORT_EP_FAIL, message.c_str());
    }
    if (IsRelativePathToParentPath(cache_path)) {
      std::string message = "The file path in ep_cache_context attribute has '..'. For security purpose, it's not allowed to point outside the directory.";
      return ort_api.CreateStatus(ORT_EP_FAIL, message.c_str());
    }

    // The engine cache and context model (current model) should be in the same directory
    // Prefer the loaded model's actual location so the model and engine can move together.
    // A memory-loaded model can instead provide ep.context_file_path as its base path.
    const auto model_path = ort_graph.GetModelPath();
    RETURN_IF_NOT(!model_path.empty() || !ep_context_model_path_.empty(),
                  "External EPContext loaded from memory requires ep.context_file_path.");
    const auto ctx_model_dir = model_path.empty()
                                  ? GetPathOrParentPathOfCtxModel(ep_context_model_path_)
                                  : std::filesystem::path(model_path).parent_path();
    auto engine_cache_path = ctx_model_dir / std::filesystem::u8path(cache_path);

    std::string message = "[TensorRT EP] GetEpContextFromGraph engine_cache_path: " + engine_cache_path.u8string();
    Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger_,
                                                OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));

    // If it's a weight-stripped engine cache, it needs to be refitted even though the refit flag is not enabled
    if (!weight_stripped_engine_refit_) {
      weight_stripped_engine_refit_ = IsWeightStrippedEngineCache(engine_cache_path);
    }

    // If the serialized refitted engine is present, use it directly without refitting the engine again
    if (weight_stripped_engine_refit_) {
      const auto refitted_engine_cache_path = std::filesystem::u8path(GetWeightRefittedEnginePath(engine_cache_path.u8string()));
      if (std::filesystem::exists(refitted_engine_cache_path)) {
        std::string message = "[TensorRT EP] " + refitted_engine_cache_path.u8string() + " exists.";
        Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger_,
                                                    OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                    message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));
        engine_cache_path = refitted_engine_cache_path;
        weight_stripped_engine_refit_ = false;
      }
    }

    if (!std::filesystem::exists(engine_cache_path)) {
      std::string error_msg =
          "TensorRT EP can't find engine cache: " + engine_cache_path.u8string() +
          ". Please make sure engine cache is in the same directory or sub-directory of context model.";
      return ort_api.CreateStatus(ORT_EP_FAIL, error_msg.c_str());
    }

    std::ifstream engine_file(engine_cache_path, std::ios::binary | std::ios::in);
    RETURN_IF_NOT(engine_file.is_open(), "Cannot open EPContext engine cache: ", engine_cache_path.u8string());
    engine_file.seekg(0, std::ios::end);
    const auto file_size = engine_file.tellg();
    RETURN_IF_NOT(file_size > 0, "EPContext engine cache is empty or unreadable: ", engine_cache_path.u8string());
    const size_t engine_size = static_cast<size_t>(file_size);
    engine_file.seekg(0, std::ios::beg);
    std::unique_ptr<char[]> engine_buf{new char[engine_size]};
    RETURN_IF_NOT(engine_file.read(engine_buf.get(), static_cast<std::streamsize>(engine_size)),
                  "Cannot read EPContext engine cache: ", engine_cache_path.u8string());
    *(trt_engine_) = std::unique_ptr<nvinfer1::ICudaEngine>(trt_runtime_->deserializeCudaEngine(engine_buf.get(), engine_size));
    if (!(*trt_engine_)) {
      std::string error_msg = "TensorRT EP could not deserialize engine from cache: " + engine_cache_path.u8string();
      return ort_api.CreateStatus(ORT_EP_FAIL, error_msg.c_str());
    }

    message = "[TensorRT EP] DeSerialized " + engine_cache_path.u8string();
    Ort::ThrowOnError(ort_api.Logger_LogMessage(&logger_,
                                                OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
                                                message.c_str(), ORT_FILE, __LINE__, __FUNCTION__));

    if (weight_stripped_engine_refit_) {
      RETURN_IF_ERROR(node.GetAttributeByName("onnx_model_filename", node_attr));
      RETURN_IF_NOT(node_attr.GetType() == OrtOpAttrType::ORT_OP_ATTR_STRING, "\'onnx_model_filename\' attribute should be string type.");
      std::string onnx_model_filename;
      RETURN_IF_ERROR(node_attr.GetValue<std::string>(onnx_model_filename));
      std::string weight_stripped_engine_cache = engine_cache_path.u8string();
      auto status = ep_.RefitEngine(onnx_model_filename,
                                    onnx_model_folder_path_,
                                    weight_stripped_engine_cache,
                                    make_secure_path_checks,
                                    onnx_model_bytestream_,
                                    onnx_model_bytestream_size_,
                                    onnx_external_data_bytestream_,
                                    onnx_external_data_bytestream_size_,
                                    (*trt_engine_).get(),
                                    true,  // serialize refitted engine to disk
                                    detailed_build_log_);
      if (status != nullptr) {
        return ort_api.CreateStatus(ORT_EP_FAIL, "RefitEngine failed.");
      }
    }
  }
  return nullptr;
}

/*
 * Get the weight-refitted engine cache path from a weight-stripped engine cache path
 *
 * Weight-stipped engine:
 * An engine with weights stripped and its size is smaller than a regualr engine.
 * The cache name of weight-stripped engine is TensorrtExecutionProvider_TRTKernel_XXXXX.stripped.engine
 *
 * Weight-refitted engine:
 * An engine that its weights have been refitted and it's simply a regular engine.
 * The cache name of weight-refitted engine is TensorrtExecutionProvider_TRTKernel_XXXXX.engine
 */
std::string GetWeightRefittedEnginePath(std::string stripped_engine_cache) {
  const auto stripped_engine_cache_path = std::filesystem::u8path(stripped_engine_cache);
  std::string refitted_engine_cache_path = stripped_engine_cache_path.stem().stem().u8string() + ".engine";
  return refitted_engine_cache_path;
}
}  // namespace trt_ep
