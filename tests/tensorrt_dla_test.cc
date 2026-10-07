// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include <onnx/onnx_pb.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#if ORT_API_VERSION >= 27
namespace {

constexpr uint32_t kDlaVendorId = 0x4144564E;

// FP16 Conv with unit weights: [1,4,8,8] -> [1,8,6,6]. Each output is 36*x.
std::string MakeDlaConvModel() {
  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(8);
  model.add_opset_import()->set_version(13);
  auto* graph = model.mutable_graph();
  graph->set_name("dla_conv");
  auto add_type = [](ONNX_NAMESPACE::ValueInfoProto* value, const char* name,
                     std::initializer_list<int64_t> shape) {
    value->set_name(name);
    auto* tensor_type = value->mutable_type()->mutable_tensor_type();
    tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
    for (auto dimension : shape) tensor_type->mutable_shape()->add_dim()->set_dim_value(dimension);
  };
  add_type(graph->add_input(), "x", {1, 4, 8, 8});
  add_type(graph->add_output(), "y", {1, 8, 6, 6});

  auto* weights = graph->add_initializer();
  weights->set_name("weights");
  weights->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  for (auto dimension : {8, 4, 3, 3}) weights->add_dims(dimension);
  std::vector<Ort::Float16_t> data(8 * 4 * 3 * 3, Ort::Float16_t(1.0f));
  weights->set_raw_data(data.data(), data.size() * sizeof(Ort::Float16_t));

  auto* conv = graph->add_node();
  conv->set_name("conv");
  conv->set_op_type("Conv");
  conv->add_input("x");
  conv->add_input("weights");
  conv->add_output("y");
  auto* kernel_shape = conv->add_attribute();
  kernel_shape->set_name("kernel_shape");
  kernel_shape->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INTS);
  kernel_shape->add_ints(3);
  kernel_shape->add_ints(3);
  return model.SerializeAsString();
}

class TensorrtDlaTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const char* library_path = std::getenv("TRT_EP_LIBRARY_PATH");
    if (!library_path || !*library_path) GTEST_SKIP() << "TRT_EP_LIBRARY_PATH is not set";
    Ort::InitApi();
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "TensorrtDlaTest");
#ifdef _WIN32
    const std::string path(library_path);
    env_->RegisterExecutionProviderLibrary("TRTPluginEP", std::wstring(path.begin(), path.end()));
#else
    env_->RegisterExecutionProviderLibrary("TRTPluginEP", std::string(library_path));
#endif
    registered_ = true;
    for (const auto& device : env_->GetEpDevices()) {
      if (std::strcmp(device.EpName(), "TRTPluginEP") == 0 &&
          device.Device().Type() == OrtHardwareDeviceType_NPU &&
          device.Device().VendorId() == kDlaVendorId) {
        devices_.push_back(device);
        break;
      }
    }
    if (devices_.empty()) GTEST_SKIP() << "No TensorRT DLA EpDevice discovered";
  }

  void TearDown() override {
    devices_.clear();
    if (registered_) env_->UnregisterExecutionProviderLibrary("TRTPluginEP");
    env_.reset();
  }

  Ort::Session MakeSession(const std::string& model,
                           const std::unordered_map<std::string, std::string>& options = {}) {
    Ort::SessionOptions session_options;
    session_options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
    auto provider_options = options;
    provider_options.emplace("trt_fp16_enable", "1");
    // Exercise the defaults attached to the DLA EpDevice: no explicit dla_enable.
    session_options.AppendExecutionProvider_V2(*env_, devices_, provider_options);
    return Ort::Session(*env_, model.data(), model.size(), session_options);
  }

  std::unique_ptr<Ort::Env> env_;
  std::vector<Ort::ConstEpDevice> devices_;
  bool registered_ = false;
};

TEST_F(TensorrtDlaTest, DeviceUsesHostAccessibleNpuMemory) {
  const auto& device = devices_.front();
  ASSERT_STREQ(device.EpMetadata().GetValue("device_type"), "DLA");
  ASSERT_STREQ(device.EpOptions().GetValue("trt_dla_enable"), "1");
  auto memory_info = device.GetMemoryInfo(OrtDeviceMemoryType_HOST_ACCESSIBLE);
  ASSERT_NE(static_cast<const OrtMemoryInfo*>(memory_info), nullptr);
  EXPECT_EQ(memory_info.GetAllocatorName(), "CudaDLA");
  EXPECT_EQ(memory_info.GetDeviceType(), OrtMemoryInfoDeviceType_NPU);
  EXPECT_EQ(memory_info.GetDeviceMemoryType(), OrtDeviceMemoryType_HOST_ACCESSIBLE);
  EXPECT_EQ(memory_info.GetVendorId(), kDlaVendorId);
  EXPECT_EQ(memory_info.GetAllocatorType(), OrtDeviceAllocator);
}

TEST_F(TensorrtDlaTest, RepeatedRunsCopyCpuInputsAndOutputs) {
  const auto model = MakeDlaConvModel();
  auto session = MakeSession(model);
  const std::array<int64_t, 4> shape{1, 4, 8, 8};
  auto cpu_memory = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  const char* input_names[] = {"x"};
  const char* output_names[] = {"y"};
  // Distinct addresses and changing data exercise re-registration between runs.
  std::array<std::vector<Ort::Float16_t>, 2> inputs;
  inputs[0].assign(256, Ort::Float16_t(1.0f));
  inputs[1].assign(256, Ort::Float16_t(2.0f));
  for (size_t run = 0; run < 8; ++run) {
    auto& data = inputs[run % 2];
    auto input = Ort::Value::CreateTensor<Ort::Float16_t>(cpu_memory, data.data(), data.size(), shape.data(), shape.size());
    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, &input, 1, output_names, 1);
    ASSERT_EQ(outputs.size(), 1u);
    ASSERT_EQ(outputs[0].GetTensorTypeAndShapeInfo().GetShape(), (std::vector<int64_t>{1, 8, 6, 6}));
    EXPECT_EQ(outputs[0].GetTensorMemoryInfo().GetDeviceType(), OrtMemoryInfoDeviceType_CPU);
    const auto* result = outputs[0].GetTensorData<Ort::Float16_t>();
    for (size_t i = 0; i < 288; ++i) EXPECT_FLOAT_EQ(result[i].ToFloat(), run % 2 ? 72.0f : 36.0f);
  }
}

TEST_F(TensorrtDlaTest, BoundOutputIsCudaPinnedMemory) {
  const auto model = MakeDlaConvModel();
  auto session = MakeSession(model);
  const std::array<int64_t, 4> shape{1, 4, 8, 8};
  std::vector<Ort::Float16_t> data(256, Ort::Float16_t(1.0f));
  auto cpu_memory = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  auto input = Ort::Value::CreateTensor<Ort::Float16_t>(cpu_memory, data.data(), data.size(), shape.data(), shape.size());
  Ort::IoBinding binding(session);
  binding.BindInput("x", input);
  binding.BindOutput("y", devices_.front().GetMemoryInfo(OrtDeviceMemoryType_HOST_ACCESSIBLE));
  session.Run(Ort::RunOptions{nullptr}, binding);
  auto outputs = binding.GetOutputValues();
  ASSERT_EQ(outputs.size(), 1u);
  EXPECT_EQ(outputs[0].GetTensorMemoryInfo().GetDeviceType(), OrtMemoryInfoDeviceType_NPU);
  cudaPointerAttributes attributes{};
  ASSERT_EQ(cudaPointerGetAttributes(&attributes, outputs[0].GetTensorData<Ort::Float16_t>()), cudaSuccess);
  EXPECT_EQ(attributes.type, cudaMemoryTypeHost);
  const auto* result = outputs[0].GetTensorData<Ort::Float16_t>();
  for (size_t i = 0; i < 288; ++i) EXPECT_FLOAT_EQ(result[i].ToFloat(), 36.0f);
}

TEST_F(TensorrtDlaTest, RejectsDisablingDlaOnDlaDevice) {
  const auto model = MakeDlaConvModel();
  EXPECT_THROW(MakeSession(model, {{"trt_dla_enable", "0"}}), Ort::Exception);
}

TEST_F(TensorrtDlaTest, RejectsDlaCudaGraphCapture) {
  const auto model = MakeDlaConvModel();
  EXPECT_THROW(MakeSession(model, {{"trt_cuda_graph_enable", "1"}}), Ort::Exception);
}

TEST_F(TensorrtDlaTest, RejectsInvalidDlaCore) {
  const auto model = MakeDlaConvModel();
  EXPECT_THROW(MakeSession(model, {{"trt_dla_core", "2147483647"}}), Ort::Exception);
}

}  // namespace
#else
TEST(TensorrtDlaTest, RequiresOrtApi27) {
  GTEST_SKIP() << "DLA requires ORT API 27 or later";
}
#endif
