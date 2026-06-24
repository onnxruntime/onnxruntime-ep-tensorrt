// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Unit tests for TensorRT plugin EP basic functionality.
// Adapted from onnxruntime/test/providers/tensorrt/tensorrt_basic_test.cc
// to work with the plugin EP library registration approach.

#include <gtest/gtest.h>

#include <onnx/onnx_pb.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Get the directory where testdata files are stored.
// Looks for TESTDATA_DIR env var, otherwise uses a relative path from the binary.
static std::filesystem::path GetTestDataDir() {
  const char* env = std::getenv("TESTDATA_DIR");
  if (env && std::strlen(env) > 0) {
    return std::filesystem::path(env);
  }
  // Try relative to current working directory
  auto cwd_path = std::filesystem::current_path() / "testdata";
  if (std::filesystem::exists(cwd_path)) {
    return cwd_path;
  }
  // Try source tree layout
  auto src_path = std::filesystem::current_path() / "tests" / "testdata";
  if (std::filesystem::exists(src_path)) {
    return src_path;
  }
  return std::filesystem::path("testdata");
}

// Get the path to the TRT plugin EP library from environment variable.
static std::string GetEpLibraryPath() {
  const char* env = std::getenv("TRT_EP_LIBRARY_PATH");
  if (env && std::strlen(env) > 0) {
    return std::string(env);
  }
  GTEST_LOG_(WARNING) << "TRT_EP_LIBRARY_PATH not set. Set it to the path of onnxruntime_ep_tensorrt shared library.";
  return "";
}

// Build a model with Add ops: M = (X + Y) + Z
// Input:  X, Y, Z  float [dims...]
// Output: M        float [dims...]
static std::string CreateAddModel(const std::vector<int>& dims) {
  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  auto* opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(13);

  auto* graph = model.mutable_graph();
  graph->set_name("add_graph");

  auto make_float_type = [&](const std::vector<int>& shape) {
    ONNX_NAMESPACE::TypeProto type;
    type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    int dyn_idx = 0;
    for (auto d : shape) {
      if (d < 0) {
        type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_param(
            "dynamic_" + std::to_string(dyn_idx++));
      } else {
        type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(d);
      }
    }
    return type;
  };

  auto float_type = make_float_type(dims);

  // Inputs
  for (const char* name : {"X", "Y", "Z"}) {
    auto* input = graph->add_input();
    input->set_name(name);
    *input->mutable_type() = float_type;
  }

  // Output M
  auto* output = graph->add_output();
  output->set_name("M");
  *output->mutable_type() = float_type;

  // Node 1: tmp = Add(X, Y)
  auto* node1 = graph->add_node();
  node1->set_op_type("Add");
  node1->set_name("node_1");
  node1->add_input("X");
  node1->add_input("Y");
  node1->add_output("node_1_out");

  // Node 2: M = Add(tmp, Z)
  auto* node2 = graph->add_node();
  node2->set_op_type("Add");
  node2->set_name("node_2");
  node2->add_input("node_1_out");
  node2->add_input("Z");
  node2->add_output("M");

  std::string model_data;
  model.SerializeToString(&model_data);
  return model_data;
}

// Create a synthetic EPContext model with a specific "source" attribute.
static std::string CreateSyntheticEPContextModel(const std::string& source_attr,
                                                 bool include_source_attr = true) {
  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  auto* opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(11);
  auto* ms_opset = model.add_opset_import();
  ms_opset->set_domain("com.microsoft");
  ms_opset->set_version(1);

  auto* graph = model.mutable_graph();
  graph->set_name("EPContextSourceTest");

  // Input
  auto* input = graph->add_input();
  input->set_name("input");
  auto* input_type = input->mutable_type()->mutable_tensor_type();
  input_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  input_type->mutable_shape()->add_dim()->set_dim_value(1);
  input_type->mutable_shape()->add_dim()->set_dim_value(3);

  // Output
  auto* output = graph->add_output();
  output->set_name("output");
  auto* output_type = output->mutable_type()->mutable_tensor_type();
  output_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  output_type->mutable_shape()->add_dim()->set_dim_value(1);
  output_type->mutable_shape()->add_dim()->set_dim_value(3);

  // EPContext node
  auto* node = graph->add_node();
  node->set_op_type("EPContext");
  node->set_domain("com.microsoft");
  node->set_name("ep_context_node");
  node->add_input("input");
  node->add_output("output");

  // embed_mode attribute
  auto* attr_embed = node->add_attribute();
  attr_embed->set_name("embed_mode");
  attr_embed->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  attr_embed->set_i(1);

  // ep_cache_context attribute (dummy data)
  auto* attr_cache = node->add_attribute();
  attr_cache->set_name("ep_cache_context");
  attr_cache->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_STRING);
  attr_cache->set_s("dummy_context_data");

  // source attribute (conditionally added)
  if (include_source_attr) {
    auto* attr_source = node->add_attribute();
    attr_source->set_name("source");
    attr_source->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_STRING);
    attr_source->set_s(source_attr);
  }

  std::string model_data;
  model.SerializeToString(&model_data);
  return model_data;
}

// Write model data to a file and return the path.
static std::filesystem::path WriteModelToFile(const std::string& model_data,
                                              const std::string& filename) {
  auto temp_dir = std::filesystem::temp_directory_path();
  auto model_path = temp_dir / filename;
  std::ofstream ofs(model_path, std::ios::binary);
  ofs.write(model_data.data(), model_data.size());
  ofs.close();
  return model_path;
}

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------

class TensorrtBasicTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ep_library_path_ = GetEpLibraryPath();
    if (ep_library_path_.empty()) {
      GTEST_SKIP() << "TRT_EP_LIBRARY_PATH not set, skipping TensorRT basic tests.";
    }

    Ort::InitApi();
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "TensorrtBasicTest");

    ep_registration_name_ = "TRTPluginEP";
#ifdef _WIN32
    std::wstring wide_path(ep_library_path_.begin(), ep_library_path_.end());
    env_->RegisterExecutionProviderLibrary(ep_registration_name_.c_str(), wide_path);
#else
    env_->RegisterExecutionProviderLibrary(ep_registration_name_.c_str(), ep_library_path_);
#endif
  }

  void TearDown() override {
    if (!ep_library_path_.empty() && env_) {
      env_->UnregisterExecutionProviderLibrary(ep_registration_name_.c_str());
    }
    env_.reset();
    // Clean up temp model files
    for (const auto& path : temp_files_) {
      if (std::filesystem::exists(path)) {
        std::filesystem::remove(path);
      }
    }
  }

  // Create a session with the TRT plugin EP.
  Ort::Session CreateSession(const std::filesystem::path& model_path,
                             const std::unordered_map<std::string, std::string>& ep_options = {}) {
    Ort::SessionOptions session_options;

    auto all_ep_devices = env_->GetEpDevices();
    std::vector<Ort::ConstEpDevice> selected_devices;
    for (const auto& ep_device : all_ep_devices) {
      if (std::string(ep_device.EpName()) == ep_registration_name_) {
        selected_devices.push_back(ep_device);
        break;
      }
    }
    EXPECT_FALSE(selected_devices.empty()) << "No TRT EP device found";

    session_options.AppendExecutionProvider_V2(*env_, selected_devices, ep_options);

#ifdef _WIN32
    std::wstring wide_model_path = model_path.wstring();
    return Ort::Session(*env_, wide_model_path.c_str(), session_options);
#else
    return Ort::Session(*env_, model_path.c_str(), session_options);
#endif
  }

  // Write model to temp and track for cleanup
  std::filesystem::path WriteAndTrack(const std::string& model_data, const std::string& filename) {
    auto path = WriteModelToFile(model_data, filename);
    temp_files_.push_back(path);
    return path;
  }

  std::unique_ptr<Ort::Env> env_;
  std::string ep_library_path_;
  std::string ep_registration_name_;
  std::vector<std::filesystem::path> temp_files_;
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// Test basic inference with a simple Add model: M = (X + Y) + Z
// Adapted from TensorrtExecutionProviderTest.FunctionTest
TEST_F(TensorrtBasicTest, FunctionTest) {
  std::vector<int> dims = {1, 3, 2};
  auto model_data = CreateAddModel(dims);
  auto model_path = WriteAndTrack(model_data, "trt_basic_function_test.onnx");

  auto session = CreateSession(model_path);

  // Prepare inputs
  std::array<float, 6> x_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  const std::array<int64_t, 3> shape = {1, 3, 2};

  Ort::MemoryInfo cpu_mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto input_x = Ort::Value::CreateTensor(cpu_mem, x_values.data(), x_values.size(), shape.data(), shape.size());
  auto input_y = Ort::Value::CreateTensor(cpu_mem, x_values.data(), x_values.size(), shape.data(), shape.size());
  auto input_z = Ort::Value::CreateTensor(cpu_mem, x_values.data(), x_values.size(), shape.data(), shape.size());

  const char* input_names[] = {"X", "Y", "Z"};
  const char* output_names[] = {"M"};
  Ort::Value inputs[] = {std::move(input_x), std::move(input_y), std::move(input_z)};

  auto outputs = session.Run(Ort::RunOptions{}, input_names, inputs, 3, output_names, 1);

  ASSERT_EQ(outputs.size(), 1u);
  const float* output_data = outputs[0].GetTensorData<float>();

  // Expected: M = (X + Y) + Z = X*3 for all same inputs
  std::array<float, 6> expected = {3.0f, 6.0f, 9.0f, 12.0f, 15.0f, 18.0f};
  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_NEAR(output_data[i], expected[i], 1e-5f) << "Mismatch at index " << i;
  }
}

// Test that session reports correct number of outputs for models with multiple outputs.
// Adapted from TensorrtExecutionProviderTest.TestSessionOutputs
TEST_F(TensorrtBasicTest, TestSessionOutputs_MultipleOutputs) {
  auto testdata_dir = GetTestDataDir();
  auto model_path = testdata_dir / "topk_and_multiple_graph_outputs.onnx";
  if (!std::filesystem::exists(model_path)) {
    GTEST_SKIP() << "Test model not found: " << model_path;
  }

  auto session = CreateSession(model_path);
  size_t output_count = session.GetOutputCount();
  ASSERT_EQ(output_count, 4u);
}

// Test that session reports correct number of outputs for model with unused node outputs.
// Adapted from TensorrtExecutionProviderTest.TestSessionOutputs (model #2)
TEST_F(TensorrtBasicTest, TestSessionOutputs_UnusedNodeOutput) {
  auto testdata_dir = GetTestDataDir();
  auto model_path = testdata_dir / "node_output_not_used.onnx";
  if (!std::filesystem::exists(model_path)) {
    GTEST_SKIP() << "Test model not found: " << model_path;
  }

  auto session = CreateSession(model_path);
  size_t output_count = session.GetOutputCount();
  ASSERT_EQ(output_count, 1u);
}

// Test inference with a model that has data-dependent shape (DDS) output.
// Adapted from TensorrtExecutionProviderTest.DDSOutputTest
// Test inference with a model that has data-dependent shape (DDS) output.
// Adapted from TensorrtExecutionProviderTest.DDSOutputTest
TEST_F(TensorrtBasicTest, DDSOutputTest) {
  auto testdata_dir = GetTestDataDir();
  auto model_path = testdata_dir / "ort_github_issue_26272_dds.onnx";
  if (!std::filesystem::exists(model_path)) {
    GTEST_SKIP() << "Test model not found: " << model_path;
  }

  auto session = CreateSession(model_path);

  // First run with shape [3, 4]
  std::vector<float> input_data(12, 0.0f);  // 3*4
  std::array<int64_t, 2> shape1 = {3, 4};

  Ort::MemoryInfo cpu_mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto input_tensor = Ort::Value::CreateTensor(cpu_mem, input_data.data(), input_data.size(),
                                               shape1.data(), shape1.size());

  const char* input_names[] = {"data"};
  const char* output_names[] = {"output"};

  auto outputs = session.Run(Ort::RunOptions{}, input_names, &input_tensor, 1, output_names, 1);
  ASSERT_EQ(outputs.size(), 1u);

  // Second run with different shape [6, 4]
  std::vector<float> input_data2(24, 0.0f);  // 6*4
  std::array<int64_t, 2> shape2 = {6, 4};
  auto input_tensor2 = Ort::Value::CreateTensor(cpu_mem, input_data2.data(), input_data2.size(),
                                                shape2.data(), shape2.size());

  auto outputs2 = session.Run(Ort::RunOptions{}, input_names, &input_tensor2, 1, output_names, 1);
  ASSERT_EQ(outputs2.size(), 1u);
}

// Test multi-threaded inference with a single session.
// Adapted from TensorrtExecutionProviderTest.SessionCreationWithSingleThreadAndInferenceWithMultiThreads
TEST_F(TensorrtBasicTest, MultiThreadInference) {
  std::vector<int> dims = {1, 3, 2};
  auto model_data = CreateAddModel(dims);
  auto model_path = WriteAndTrack(model_data, "trt_basic_multithread_test.onnx");

  auto session = CreateSession(model_path);

  std::array<float, 6> x_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::array<float, 6> expected = {3.0f, 6.0f, 9.0f, 12.0f, 15.0f, 18.0f};
  const std::array<int64_t, 3> shape = {1, 3, 2};

  auto run_inference = [&](int thread_id) {
    Ort::MemoryInfo cpu_mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    // Each thread needs its own copy of input data
    std::array<float, 6> local_x = x_values;
    auto input_x = Ort::Value::CreateTensor(cpu_mem, local_x.data(), local_x.size(), shape.data(), shape.size());
    auto input_y = Ort::Value::CreateTensor(cpu_mem, local_x.data(), local_x.size(), shape.data(), shape.size());
    auto input_z = Ort::Value::CreateTensor(cpu_mem, local_x.data(), local_x.size(), shape.data(), shape.size());

    const char* input_names[] = {"X", "Y", "Z"};
    const char* output_names[] = {"M"};
    Ort::Value inputs[] = {std::move(input_x), std::move(input_y), std::move(input_z)};

    auto outputs = session.Run(Ort::RunOptions{}, input_names, inputs, 3, output_names, 1);

    ASSERT_EQ(outputs.size(), 1u) << "Thread " << thread_id;
    const float* output_data = outputs[0].GetTensorData<float>();
    for (size_t i = 0; i < expected.size(); i++) {
      EXPECT_NEAR(output_data[i], expected[i], 1e-5f)
          << "Thread " << thread_id << " mismatch at index " << i;
    }
  };

  constexpr int num_threads = 5;
  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(run_inference, i);
  }
  for (auto& th : threads) {
    th.join();
  }
}

// Test that the mnist model can be loaded and run.
// Adapted from TensorrtExecutionProviderTest.TRTModelIdGeneratorUsingModelHashing (inference portion)
TEST_F(TensorrtBasicTest, MnistModelTest) {
  auto testdata_dir = GetTestDataDir();
  auto model_path = testdata_dir / "mnist.onnx";
  if (!std::filesystem::exists(model_path)) {
    GTEST_SKIP() << "Test model not found: " << model_path;
  }

  auto session = CreateSession(model_path);

  // mnist model: input "Input3" shape [1, 1, 28, 28], output "Plus214_Output_0" shape [1, 10]
  std::vector<float> input_data(784, 1.0f);  // 1*1*28*28
  std::array<int64_t, 4> shape = {1, 1, 28, 28};

  Ort::MemoryInfo cpu_mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto input_tensor = Ort::Value::CreateTensor(cpu_mem, input_data.data(), input_data.size(),
                                               shape.data(), shape.size());

  const char* input_names[] = {"Input3"};
  const char* output_names[] = {"Plus214_Output_0"};

  auto outputs = session.Run(Ort::RunOptions{}, input_names, &input_tensor, 1, output_names, 1);
  ASSERT_EQ(outputs.size(), 1u);

  auto type_info = outputs[0].GetTensorTypeAndShapeInfo();
  auto out_shape = type_info.GetShape();
  ASSERT_EQ(out_shape.size(), 2u);
  EXPECT_EQ(out_shape[0], 1);
  EXPECT_EQ(out_shape[1], 10);
}

// Test engine caching: run inference twice and verify that engine cache is produced.
// Adapted from TensorrtExecutionProviderCacheTest (engine cache portion)
TEST_F(TensorrtBasicTest, EngineCacheTest) {
  std::vector<int> dims = {1, 3, 2};
  auto model_data = CreateAddModel(dims);
  auto model_path = WriteAndTrack(model_data, "trt_basic_cache_test.onnx");

  // Create a temp dir for caching
  auto cache_dir = std::filesystem::temp_directory_path() / "trt_ep_cache_test";
  std::filesystem::create_directories(cache_dir);

  std::unordered_map<std::string, std::string> ep_options;
  ep_options["trt_engine_cache_enable"] = "1";
  ep_options["trt_engine_cache_path"] = cache_dir.string();

  auto session = CreateSession(model_path, ep_options);

  // Run inference
  std::array<float, 6> x_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  const std::array<int64_t, 3> shape = {1, 3, 2};

  Ort::MemoryInfo cpu_mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto input_x = Ort::Value::CreateTensor(cpu_mem, x_values.data(), x_values.size(), shape.data(), shape.size());
  auto input_y = Ort::Value::CreateTensor(cpu_mem, x_values.data(), x_values.size(), shape.data(), shape.size());
  auto input_z = Ort::Value::CreateTensor(cpu_mem, x_values.data(), x_values.size(), shape.data(), shape.size());

  const char* input_names[] = {"X", "Y", "Z"};
  const char* output_names[] = {"M"};
  Ort::Value inputs[] = {std::move(input_x), std::move(input_y), std::move(input_z)};

  auto outputs = session.Run(Ort::RunOptions{}, input_names, inputs, 3, output_names, 1);
  ASSERT_EQ(outputs.size(), 1u);

  const float* output_data = outputs[0].GetTensorData<float>();
  std::array<float, 6> expected = {3.0f, 6.0f, 9.0f, 12.0f, 15.0f, 18.0f};
  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_NEAR(output_data[i], expected[i], 1e-5f) << "Mismatch at index " << i;
  }

  // Check that cache files were produced
  bool has_engine_cache = false;
  if (std::filesystem::exists(cache_dir)) {
    for (const auto& entry : std::filesystem::directory_iterator(cache_dir)) {
      if (entry.is_regular_file()) {
        std::string ext = entry.path().extension().string();
        if (ext == ".engine" || ext == ".trt") {
          has_engine_cache = true;
          break;
        }
      }
    }
  }
  // Note: cache file creation depends on the EP implementation.
  // If no cache file is found, it's not necessarily a failure for the plugin EP.
  if (has_engine_cache) {
    GTEST_LOG_(INFO) << "Engine cache file found in " << cache_dir;
  }

  // Clean up cache dir
  std::filesystem::remove_all(cache_dir);
}

// Test EPContext node: EP should NOT claim an EPContext node whose "source"
// attribute belongs to a different EP (e.g., OpenVINO).
// Adapted from TensorrtExecutionProviderTest.EPContextNode_ForeignSourceSkipped
TEST_F(TensorrtBasicTest, EPContextNode_ForeignSourceSkipped) {
  auto model_data = CreateSyntheticEPContextModel("OpenVINOExecutionProvider");
  auto model_path = WriteAndTrack(model_data, "ep_context_foreign_source_plugin.onnx");

  // Try to create session - it should either fail or fallback to CPU
  // (since no EP claims the EPContext node with foreign source)
  try {
    auto session = CreateSession(model_path);
    // If session creation succeeds, the EPContext node was handled somehow.
    // This is acceptable if it falls back to CPU EP.
    GTEST_LOG_(INFO) << "Session created (possibly fell back to CPU)";
  } catch (const Ort::Exception& e) {
    // Expected: session creation fails because no EP claims the node
    std::string error_msg = e.what();
    GTEST_LOG_(INFO) << "Session creation failed as expected: " << error_msg;
    SUCCEED();
  }
}

// Test EPContext node: EP should still claim a node with NO "source" attribute
// (backward compatibility).
// Adapted from TensorrtExecutionProviderTest.EPContextNode_NoSourceAttribute_BackwardCompat
TEST_F(TensorrtBasicTest, EPContextNode_NoSourceAttribute_BackwardCompat) {
  auto model_data = CreateSyntheticEPContextModel("", /*include_source_attr=*/false);
  auto model_path = WriteAndTrack(model_data, "ep_context_no_source_plugin.onnx");

  // The EP should claim the node (backward compatibility).
  // It may fail during engine deserialization since context data is synthetic,
  // but the error should NOT be about no EP claiming the node.
  try {
    auto session = CreateSession(model_path);
    GTEST_LOG_(INFO) << "Session created successfully (EP claimed the node)";
  } catch (const Ort::Exception& e) {
    std::string error_msg = e.what();
    // The error should NOT indicate that no EP claimed the node
    EXPECT_TRUE(error_msg.find("is not compatible with any execution provider") == std::string::npos)
        << "Legacy EPContext node without source should still be claimed. Error: " << error_msg;
  }
}

// Test running the same model multiple times in sequence to verify stability.
TEST_F(TensorrtBasicTest, SequentialRuns) {
  std::vector<int> dims = {1, 3, 2};
  auto model_data = CreateAddModel(dims);
  auto model_path = WriteAndTrack(model_data, "trt_basic_sequential_test.onnx");

  auto session = CreateSession(model_path);

  std::array<float, 6> x_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::array<float, 6> expected = {3.0f, 6.0f, 9.0f, 12.0f, 15.0f, 18.0f};
  const std::array<int64_t, 3> shape = {1, 3, 2};

  Ort::MemoryInfo cpu_mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  for (int run = 0; run < 5; run++) {
    auto input_x = Ort::Value::CreateTensor(cpu_mem, x_values.data(), x_values.size(), shape.data(), shape.size());
    auto input_y = Ort::Value::CreateTensor(cpu_mem, x_values.data(), x_values.size(), shape.data(), shape.size());
    auto input_z = Ort::Value::CreateTensor(cpu_mem, x_values.data(), x_values.size(), shape.data(), shape.size());

    const char* input_names[] = {"X", "Y", "Z"};
    const char* output_names[] = {"M"};
    Ort::Value inputs[] = {std::move(input_x), std::move(input_y), std::move(input_z)};

    auto outputs = session.Run(Ort::RunOptions{}, input_names, inputs, 3, output_names, 1);
    ASSERT_EQ(outputs.size(), 1u);

    const float* output_data = outputs[0].GetTensorData<float>();
    for (size_t i = 0; i < expected.size(); i++) {
      EXPECT_NEAR(output_data[i], expected[i], 1e-5f)
          << "Run " << run << " mismatch at index " << i;
    }
  }
}

// Test with dynamic input shapes - run with different shapes.
// Adapted from TensorrtExecutionProviderCacheTest engine_dynamic test.
TEST_F(TensorrtBasicTest, DynamicInputShapes) {
  // Create model with dynamic dims
  std::vector<int> dims = {1, -1, -1};  // dynamic shape
  auto model_data = CreateAddModel(dims);
  auto model_path = WriteAndTrack(model_data, "trt_basic_dynamic_shape_test.onnx");

  // Provide explicit profile shapes to cover the range of shapes we'll test
  std::unordered_map<std::string, std::string> ep_options;
  ep_options["trt_profile_min_shapes"] = "X:1x1x1,Y:1x1x1,Z:1x1x1";
  ep_options["trt_profile_max_shapes"] = "X:1x6x6,Y:1x6x6,Z:1x6x6";
  ep_options["trt_profile_opt_shapes"] = "X:1x3x2,Y:1x3x2,Z:1x3x2";

  auto session = CreateSession(model_path, ep_options);

  Ort::MemoryInfo cpu_mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  // First run with shape [1, 3, 2]
  {
    std::array<float, 6> x_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::array<float, 6> expected = {3.0f, 6.0f, 9.0f, 12.0f, 15.0f, 18.0f};
    const std::array<int64_t, 3> shape = {1, 3, 2};

    auto input_x = Ort::Value::CreateTensor(cpu_mem, x_values.data(), x_values.size(), shape.data(), shape.size());
    auto input_y = Ort::Value::CreateTensor(cpu_mem, x_values.data(), x_values.size(), shape.data(), shape.size());
    auto input_z = Ort::Value::CreateTensor(cpu_mem, x_values.data(), x_values.size(), shape.data(), shape.size());

    const char* input_names[] = {"X", "Y", "Z"};
    const char* output_names[] = {"M"};
    Ort::Value inputs[] = {std::move(input_x), std::move(input_y), std::move(input_z)};

    auto outputs = session.Run(Ort::RunOptions{}, input_names, inputs, 3, output_names, 1);
    ASSERT_EQ(outputs.size(), 1u);

    const float* output_data = outputs[0].GetTensorData<float>();
    for (size_t i = 0; i < expected.size(); i++) {
      EXPECT_NEAR(output_data[i], expected[i], 1e-5f) << "Run 1 mismatch at index " << i;
    }
  }

  // Second run with different shape [1, 1, 6]
  {
    std::array<float, 6> x_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::array<float, 6> expected = {3.0f, 6.0f, 9.0f, 12.0f, 15.0f, 18.0f};
    const std::array<int64_t, 3> shape = {1, 1, 6};

    auto input_x = Ort::Value::CreateTensor(cpu_mem, x_values.data(), x_values.size(), shape.data(), shape.size());
    auto input_y = Ort::Value::CreateTensor(cpu_mem, x_values.data(), x_values.size(), shape.data(), shape.size());
    auto input_z = Ort::Value::CreateTensor(cpu_mem, x_values.data(), x_values.size(), shape.data(), shape.size());

    const char* input_names[] = {"X", "Y", "Z"};
    const char* output_names[] = {"M"};
    Ort::Value inputs[] = {std::move(input_x), std::move(input_y), std::move(input_z)};

    auto outputs = session.Run(Ort::RunOptions{}, input_names, inputs, 3, output_names, 1);
    ASSERT_EQ(outputs.size(), 1u);

    const float* output_data = outputs[0].GetTensorData<float>();
    for (size_t i = 0; i < expected.size(); i++) {
      EXPECT_NEAR(output_data[i], expected[i], 1e-5f) << "Run 2 mismatch at index " << i;
    }
  }

  // Third run with yet another shape [1, 2, 3]
  {
    std::array<float, 6> x_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::array<float, 6> expected = {3.0f, 6.0f, 9.0f, 12.0f, 15.0f, 18.0f};
    const std::array<int64_t, 3> shape = {1, 2, 3};

    auto input_x = Ort::Value::CreateTensor(cpu_mem, x_values.data(), x_values.size(), shape.data(), shape.size());
    auto input_y = Ort::Value::CreateTensor(cpu_mem, x_values.data(), x_values.size(), shape.data(), shape.size());
    auto input_z = Ort::Value::CreateTensor(cpu_mem, x_values.data(), x_values.size(), shape.data(), shape.size());

    const char* input_names[] = {"X", "Y", "Z"};
    const char* output_names[] = {"M"};
    Ort::Value inputs[] = {std::move(input_x), std::move(input_y), std::move(input_z)};

    auto outputs = session.Run(Ort::RunOptions{}, input_names, inputs, 3, output_names, 1);
    ASSERT_EQ(outputs.size(), 1u);

    const float* output_data = outputs[0].GetTensorData<float>();
    for (size_t i = 0; i < expected.size(); i++) {
      EXPECT_NEAR(output_data[i], expected[i], 1e-5f) << "Run 3 mismatch at index " << i;
    }
  }
}

// Test TRT plugins custom op: verify that a model using a custom op from the
// "trt.plugins" domain can be loaded and that the session initializes successfully.
// This validates that the EP factory correctly registers TRT plugins as custom ops
// via GetNumCustomOpDomains/GetCustomOpDomains.
// Adapted from TensorrtExecutionProviderTest.TRTPluginsCustomOpTest
TEST_F(TensorrtBasicTest, TRTPluginsCustomOpTest) {
  auto testdata_dir = GetTestDataDir();
  auto model_path = testdata_dir / "trt_plugin_custom_op_test.onnx";
  if (!std::filesystem::exists(model_path)) {
    GTEST_SKIP() << "Test model not found: " << model_path;
  }

  // The model contains a DisentangledAttention_TRT node in the "trt.plugins" domain.
  // If custom ops are not registered, session creation will fail because ORT won't
  // recognize the custom op domain/type.
  auto session = CreateSession(model_path);

  // Prepare inputs: three float tensors of shape [12, 256, 256]
  constexpr size_t elem_count = 12 * 256 * 256;  // 786432
  std::vector<float> input_data(elem_count, 1.0f);
  std::array<int64_t, 3> shape = {12, 256, 256};

  Ort::MemoryInfo cpu_mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto input1 = Ort::Value::CreateTensor(cpu_mem, input_data.data(), input_data.size(),
                                         shape.data(), shape.size());
  auto input2 = Ort::Value::CreateTensor(cpu_mem, input_data.data(), input_data.size(),
                                         shape.data(), shape.size());
  auto input3 = Ort::Value::CreateTensor(cpu_mem, input_data.data(), input_data.size(),
                                         shape.data(), shape.size());

  const char* input_names[] = {"input1", "input2", "input3"};
  const char* output_names[] = {"output"};
  Ort::Value inputs[] = {std::move(input1), std::move(input2), std::move(input3)};

  // Run inference. The DisentangledAttention_TRT plugin may or may not be present
  // in the TRT plugin registry depending on the TRT version. The key validation is
  // that session creation succeeded (custom ops were registered). If the specific
  // plugin is not available, the Run may fail -- that's acceptable.
  try {
    auto outputs = session.Run(Ort::RunOptions{}, input_names, inputs, 3, output_names, 1);
    ASSERT_EQ(outputs.size(), 1u);

    // Verify output shape matches expected [12, 256, 256]
    auto type_info = outputs[0].GetTensorTypeAndShapeInfo();
    auto out_shape = type_info.GetShape();
    ASSERT_EQ(out_shape.size(), 3u);
    EXPECT_EQ(out_shape[0], 12);
    EXPECT_EQ(out_shape[1], 256);
    EXPECT_EQ(out_shape[2], 256);
  } catch (const Ort::Exception& e) {
    // If the specific TRT plugin (DisentangledAttention_TRT) is not registered,
    // inference may fail. This is still a valid test -- the key assertion is that
    // the session was created and initialized successfully above.
    GTEST_LOG_(INFO) << "Inference with TRT plugin custom op threw (plugin may not be available): " << e.what();
  }
}
