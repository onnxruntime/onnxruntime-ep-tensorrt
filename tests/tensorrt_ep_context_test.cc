// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include <onnx/onnx_pb.h>
#include <cuda_runtime_api.h>

#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

namespace {

// The same FP16 Conv is supported by GPU and DLA without graph transforms.
ONNX_NAMESPACE::ModelProto MakeContextTestModel() {
  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(8);
  model.add_opset_import()->set_version(13);
  auto* graph = model.mutable_graph();
  graph->set_name("context_conv");
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
  return model;
}

void WriteModel(const ONNX_NAMESPACE::ModelProto& model, const std::filesystem::path& path) {
  std::ofstream stream(path, std::ios::binary);
  ASSERT_TRUE(stream.is_open());
  ASSERT_TRUE(model.SerializeToOstream(&stream));
}

ONNX_NAMESPACE::ModelProto ReadModel(const std::filesystem::path& path) {
  ONNX_NAMESPACE::ModelProto model;
  std::ifstream stream(path, std::ios::binary);
  EXPECT_TRUE(model.ParseFromIstream(&stream)) << path;
  return model;
}

const ONNX_NAMESPACE::AttributeProto* FindAttribute(const ONNX_NAMESPACE::NodeProto& node, const char* name) {
  for (const auto& attribute : node.attribute()) {
    if (attribute.name() == name) return &attribute;
  }
  return nullptr;
}

// Parameters: use DLA, embed_mode. DLA cases skip on older ORT SDKs or absent hardware.
class TensorrtContextTest : public ::testing::TestWithParam<std::tuple<bool, int>> {
 protected:
  bool IsDla() const { return std::get<0>(GetParam()); }
  int EmbedMode() const { return std::get<1>(GetParam()); }

  void SetUp() override {
#if ORT_API_VERSION < 27
    if (IsDla()) GTEST_SKIP() << "DLA requires ORT API 27 or later";
#endif
    const char* library_path = std::getenv("TRT_EP_LIBRARY_PATH");
    if (!library_path || !*library_path) GTEST_SKIP() << "TRT_EP_LIBRARY_PATH is not set";
    Ort::InitApi();
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "TensorrtContextTest");
    env_->RegisterExecutionProviderLibrary("TRTPluginEP", std::filesystem::path(library_path).native());
    registered_ = true;
    for (const auto& device : env_->GetEpDevices()) {
      const auto type = IsDla() ? OrtHardwareDeviceType_NPU : OrtHardwareDeviceType_GPU;
      const uint32_t vendor = IsDla() ? 0x4144564E : 0x10DE;
      if (std::strcmp(device.EpName(), "TRTPluginEP") == 0 &&
          device.Device().Type() == type && device.Device().VendorId() == vendor) {
        devices_.push_back(device);
        break;
      }
    }
    if (devices_.empty()) GTEST_SKIP() << "No TensorRT " << (IsDla() ? "DLA" : "GPU") << " device discovered";
    // Create a unique directory atomically; parallel processes cannot share caches.
    const auto seed = std::chrono::steady_clock::now().time_since_epoch().count();
    for (size_t attempt = 0; ; ++attempt) {
      work_dir_ = std::filesystem::temp_directory_path() /
                  ("trt_context_" + std::to_string(seed) + "_" + std::to_string(attempt));
      if (std::filesystem::create_directory(work_dir_)) break;
    }
    bundle_dir_ = work_dir_ / "bundle";
    std::filesystem::create_directory(bundle_dir_);
    source_path_ = work_dir_ / "original_model.onnx";
    context_path_ = bundle_dir_ / "context.onnx";
    ASSERT_NO_FATAL_FAILURE(WriteModel(MakeContextTestModel(), source_path_));
  }

  void TearDown() override {
    devices_.clear();
    if (registered_) env_->UnregisterExecutionProviderLibrary("TRTPluginEP");
    env_.reset();
    if (!work_dir_.empty()) std::filesystem::remove_all(work_dir_);
  }

  Ort::SessionOptions MakeOptions(bool generate,
      std::unordered_map<std::string, std::string> provider_options = {}) {
    Ort::SessionOptions options;
    options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
    options.AddConfigEntry("ep.context_enable", generate ? "1" : "0");
    if (generate) {
      const auto embed_mode = std::to_string(EmbedMode());
      options.AddConfigEntry("ep.context_embed_mode", embed_mode.c_str());
      options.AddConfigEntry("ep.context_file_path", context_path_.string().c_str());
      provider_options.emplace("trt_engine_cache_enable", "1");
      provider_options.emplace("trt_engine_cache_path", "engines");
    }
    provider_options.emplace("trt_fp16_enable", "1");
    options.AppendExecutionProvider_V2(*env_, devices_, provider_options);
    return options;
  }

  void CheckRun(Ort::Session& session, float input_value, bool check_dla_memory = false) {
    const std::array<int64_t, 4> shape{1, 4, 8, 8};
    std::vector<Ort::Float16_t> data(256, Ort::Float16_t(input_value));
    auto cpu_memory = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    auto input = Ort::Value::CreateTensor<Ort::Float16_t>(cpu_memory, data.data(), data.size(), shape.data(), shape.size());
    const char* input_names[] = {"x"};
    const char* output_names[] = {"y"};
    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, &input, 1, output_names, 1);
    ASSERT_EQ(outputs.size(), 1u);
    ASSERT_EQ(outputs[0].GetTensorTypeAndShapeInfo().GetShape(), (std::vector<int64_t>{1, 8, 6, 6}));
    const auto* result = outputs[0].GetTensorData<Ort::Float16_t>();
    for (size_t i = 0; i < 288; ++i) EXPECT_FLOAT_EQ(result[i].ToFloat(), 36.0f * input_value);
#if ORT_API_VERSION >= 27
    if (check_dla_memory && IsDla()) {
      Ort::IoBinding binding(session);
      binding.BindInput("x", input);
      binding.BindOutput("y", devices_.front().GetMemoryInfo(OrtDeviceMemoryType_HOST_ACCESSIBLE));
      session.Run(Ort::RunOptions{nullptr}, binding);
      auto bound_outputs = binding.GetOutputValues();
      ASSERT_EQ(bound_outputs.size(), 1u);
      EXPECT_EQ(bound_outputs[0].GetTensorMemoryInfo().GetDeviceType(), OrtMemoryInfoDeviceType_NPU);
      EXPECT_EQ(bound_outputs[0].GetTensorMemoryInfo().GetDeviceMemoryType(), OrtDeviceMemoryType_HOST_ACCESSIBLE);
      const auto* bound_result = bound_outputs[0].GetTensorData<Ort::Float16_t>();
      cudaPointerAttributes attributes{};
      ASSERT_EQ(cudaPointerGetAttributes(&attributes, bound_result), cudaSuccess);
      EXPECT_EQ(attributes.type, cudaMemoryTypeHost);
      for (size_t i = 0; i < 288; ++i) EXPECT_FLOAT_EQ(bound_result[i].ToFloat(), 36.0f * input_value);
    }
#else
    (void)check_dla_memory;
#endif
  }

  void GenerateContext() {
    // Generic settings must override conflicting legacy provider options.
    auto options = MakeOptions(true, {{"trt_dump_ep_context_model", "0"},
                                     {"trt_ep_context_embed_mode", std::to_string(1 - EmbedMode())},
                                     {"trt_ep_context_file_path", (work_dir_ / "wrong_context.onnx").string()}});
    Ort::Session session(*env_, source_path_.c_str(), options);
    CheckRun(session, 1.0f);
    ASSERT_TRUE(std::filesystem::exists(context_path_));
  }

  std::unique_ptr<Ort::Env> env_;
  std::vector<Ort::ConstEpDevice> devices_;
  bool registered_ = false;
  std::filesystem::path work_dir_, bundle_dir_, source_path_, context_path_;
};

TEST_P(TensorrtContextTest, SaveMoveAndReload) {
  ASSERT_NO_FATAL_FAILURE(GenerateContext());
  const auto model = ReadModel(context_path_);
  ASSERT_EQ(model.graph().node_size(), 1);
  const auto& node = model.graph().node(0);
  ASSERT_EQ(node.op_type(), "EPContext");
  ASSERT_EQ(node.domain(), "com.microsoft");
  for (const char* name : {"embed_mode", "ep_cache_context", "source", "main_context", "partition_name",
                          "ep_sdk_version", "hardware_architecture", "onnx_model_filename"}) {
    ASSERT_NE(FindAttribute(node, name), nullptr) << name;
  }
  EXPECT_EQ(FindAttribute(node, "embed_mode")->i(), EmbedMode());
  EXPECT_EQ(FindAttribute(node, "source")->s(), "TensorrtExecutionProvider");
  EXPECT_EQ(FindAttribute(node, "main_context")->i(), 1);
  EXPECT_EQ(FindAttribute(node, "partition_name")->s(), node.name());
  EXPECT_FALSE(FindAttribute(node, "ep_sdk_version")->s().empty());
  EXPECT_FALSE(FindAttribute(node, "hardware_architecture")->s().empty());
  EXPECT_EQ(FindAttribute(node, "onnx_model_filename")->s(), "original_model.onnx");
  const auto& cache = FindAttribute(node, "ep_cache_context")->s();
  ASSERT_FALSE(cache.empty());
  if (EmbedMode() == 0) {
    const std::filesystem::path reference(cache);
    EXPECT_FALSE(reference.is_absolute());
    EXPECT_EQ(reference.parent_path(), "engines");
    ASSERT_TRUE(std::filesystem::exists(bundle_dir_ / reference));
  } else {
    // A cached engine may have been written while generating; reload must not need it.
    std::filesystem::remove_all(bundle_dir_ / "engines");
  }
  std::filesystem::remove(source_path_);
  const auto relocated = work_dir_ / "relocated";
  std::filesystem::rename(bundle_dir_, relocated);
  const auto relocated_model = relocated / "context.onnx";
  auto options = MakeOptions(false);
  {
    Ort::Session session(*env_, relocated_model.c_str(), options);
    CheckRun(session, 2.0f, true);
    CheckRun(session, 1.5f, true);
  }
  // Memory-loaded external contexts use the supplied context path as their base.
  const auto bytes = ReadModel(relocated_model).SerializeAsString();
  options.AddConfigEntry("ep.context_file_path", relocated_model.string().c_str());
  Ort::Session memory_session(*env_, bytes.data(), bytes.size(), options);
  CheckRun(memory_session, 2.0f, true);
}

TEST_P(TensorrtContextTest, EmbeddedContextCanUseCachedEngine) {
  if (EmbedMode() != 1) GTEST_SKIP() << "Applies to embedded contexts";
  // Warm the same engine cache without context generation.
  auto warm_options = MakeOptions(false, {{"trt_engine_cache_enable", "1"},
                                         {"trt_engine_cache_path", (bundle_dir_ / "engines").string()}});
  {
    Ort::Session session(*env_, source_path_.c_str(), warm_options);
    CheckRun(session, 1.0f);
  }
  std::unordered_map<std::string, std::filesystem::file_time_type> cache_files;
  for (const auto& file : std::filesystem::directory_iterator(bundle_dir_ / "engines")) {
    if (file.path().extension() == ".engine") cache_files[file.path().filename().string()] = file.last_write_time();
  }
  ASSERT_EQ(cache_files.size(), 1u);
  ASSERT_NO_FATAL_FAILURE(GenerateContext());
  // The engine was loaded from cache, rather than rebuilt and rewritten.
  size_t engine_count = 0;
  for (const auto& file : std::filesystem::directory_iterator(bundle_dir_ / "engines")) {
    if (file.path().extension() == ".engine") ++engine_count;
  }
  ASSERT_EQ(engine_count, cache_files.size());
  for (const auto& [name, timestamp] : cache_files) {
    EXPECT_EQ(std::filesystem::last_write_time(bundle_dir_ / "engines" / name), timestamp);
  }
  std::filesystem::remove_all(bundle_dir_ / "engines");
  auto options = MakeOptions(false);
  Ort::Session session(*env_, context_path_.c_str(), options);
  CheckRun(session, 2.0f, true);
}

TEST_P(TensorrtContextTest, DoesNotClaimForeignContextInMixedGraph) {
  ASSERT_NO_FATAL_FAILURE(GenerateContext());
  auto model = ReadModel(context_path_);
  ASSERT_EQ(model.graph().node_size(), 1);
  auto* graph = model.mutable_graph();
  auto* foreign_node = graph->add_node();
  *foreign_node = graph->node(0);
  foreign_node->set_name("foreign_context");
  foreign_node->set_output(0, "foreign_y");
  for (auto& attribute : *foreign_node->mutable_attribute()) {
    if (attribute.name() == "source") attribute.set_s("ForeignExecutionProvider");
    if (attribute.name() == "partition_name") attribute.set_s("foreign_context");
    if (attribute.name() == "ep_cache_context") attribute.set_s("invalid_foreign_engine");
  }
  auto* foreign_output = graph->add_output();
  *foreign_output = graph->output(0);
  foreign_output->set_name("foreign_y");
  WriteModel(model, context_path_);
  auto options = MakeOptions(false);
  try {
    Ort::Session session(*env_, context_path_.c_str(), options);
    FAIL() << "A context owned by an unregistered EP must remain unassigned";
  } catch (const Ort::Exception& error) {
    // Claiming it would invoke TensorRT deserialization or external-cache lookup.
    const std::string message = error.what();
    EXPECT_EQ(message.find("Failed to parse the ONNX model"), std::string::npos) << message;
    EXPECT_EQ(message.find("could not deserialize engine"), std::string::npos) << message;
    EXPECT_EQ(message.find("can't find engine cache"), std::string::npos) << message;
    EXPECT_EQ(message.find("not a TensorRT EPContext node"), std::string::npos) << message;
    EXPECT_TRUE(message.find("EPContext") != std::string::npos ||
                message.find("fallback") != std::string::npos) << message;
  }
}

INSTANTIATE_TEST_SUITE_P(GpuAndDla, TensorrtContextTest,
    ::testing::Values(std::make_tuple(false, 0), std::make_tuple(false, 1),
                      std::make_tuple(true, 0), std::make_tuple(true, 1)),
    [](const ::testing::TestParamInfo<TensorrtContextTest::ParamType>& info) {
      return std::string(std::get<0>(info.param) ? "Dla" : "Gpu") +
             (std::get<1>(info.param) ? "Embedded" : "External");
    });

}  // namespace
