// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <string>
#include <vector>

namespace trt_ep {

struct TensorRTCustomKernel {
  TensorRTCustomKernel(const OrtKernelInfo* /*info*/, void* compute_stream)
      : compute_stream_(compute_stream) {
  }

  void Compute(OrtKernelContext* /*context*/) {
    // The implementation is in TensorRT plugin. No need to implement it here.
  };

 private:
  void* compute_stream_;
};

struct TensorRTCustomOp : Ort::CustomOpBase<TensorRTCustomOp, TensorRTCustomKernel> {
  explicit TensorRTCustomOp(const char* provider, void* compute_stream) : provider_(provider),
                                                                          compute_stream_(compute_stream) {
  }

  void* CreateKernel(const OrtApi& /* api */, const OrtKernelInfo* info) const {
    return new TensorRTCustomKernel(info, compute_stream_);
  };

  const char* GetName() const { return name_.c_str(); };

  void SetName(const char* name) { name_ = name; };

  const char* GetExecutionProviderType() const { return provider_.c_str(); };

  size_t GetInputTypeCount() const { return num_inputs_; };

  void SetInputTypeCount(size_t num) { num_inputs_ = num; };

  ONNXTensorElementDataType GetInputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED; };

  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t) const {
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC;
  };

  size_t GetOutputTypeCount() const { return num_outputs_; };

  void SetOutputTypeCount(size_t num) { num_outputs_ = num; };

  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED; };

  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t) const {
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC;
  };

  bool GetVariadicInputHomogeneity() const {
    return false;  // heterogenous
  }

  bool GetVariadicOutputHomogeneity() const {
    return false;  // heterogeneous
  }

 private:
  std::string provider_;
  void* compute_stream_;
  std::string name_;
  size_t num_inputs_ = 1;   // set to 1 to match with default min_arity for variadic input
  size_t num_outputs_ = 1;  // set to 1 to match with default min_arity for variadic output
};

/// Creates custom op domains for TRT plugins. Returns the number of domains created.
/// The domain_list is populated with OrtCustomOpDomain pointers that remain valid
/// for the lifetime of the process (static storage).
OrtStatus* CreateTensorRTCustomOpDomainList(const char* ep_name,
                                            const std::string& extra_plugin_lib_paths,
                                            std::vector<OrtCustomOpDomain*>& domain_list);

}  // namespace trt_ep
