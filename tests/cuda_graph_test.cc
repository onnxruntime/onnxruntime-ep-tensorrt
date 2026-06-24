// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Unit test for CUDA Graph support in the TensorRT plugin EP.
// Aligned with the basic_cuda_graph test from onnxruntime's test_inference.cc.

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>
#include <onnx/onnx_pb.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Build a simple Mul model: Y = X * X  (element-wise)
// Input:  X  float [3, 2]
// Output: Y  float [3, 2]
static std::string CreateMulModel() {
  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  auto* opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(13);

  auto* graph = model.mutable_graph();
  graph->set_name("mul_graph");

  // Input X
  auto* input = graph->add_input();
  input->set_name("X");
  auto* input_type = input->mutable_type()->mutable_tensor_type();
  input_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto* input_shape = input_type->mutable_shape();
  input_shape->add_dim()->set_dim_value(3);
  input_shape->add_dim()->set_dim_value(2);

  // Output Y
  auto* output = graph->add_output();
  output->set_name("Y");
  auto* output_type = output->mutable_type()->mutable_tensor_type();
  output_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto* output_shape = output_type->mutable_shape();
  output_shape->add_dim()->set_dim_value(3);
  output_shape->add_dim()->set_dim_value(2);

  // Node: Y = Mul(X, X)
  auto* node = graph->add_node();
  node->set_op_type("Mul");
  node->set_name("mul_0");
  node->add_input("X");
  node->add_input("X");
  node->add_output("Y");

  // Serialize to string
  std::string model_data;
  model.SerializeToString(&model_data);
  return model_data;
}

// Write model data to a temporary file and return the path.
static std::filesystem::path WriteModelToTempFile(const std::string& model_data) {
  auto temp_dir = std::filesystem::temp_directory_path();
  auto model_path = temp_dir / "trt_cuda_graph_test_mul.onnx";
  std::ofstream ofs(model_path, std::ios::binary);
  ofs.write(model_data.data(), model_data.size());
  ofs.close();
  return model_path;
}

// Get the path to the TRT plugin EP library from environment variable.
static std::string GetEpLibraryPath() {
  const char* env = std::getenv("TRT_EP_LIBRARY_PATH");
  if (env && std::strlen(env) > 0) {
    return std::string(env);
  }
  // Fallback: try to find it relative to the test binary
  GTEST_LOG_(WARNING) << "TRT_EP_LIBRARY_PATH not set. Set it to the path of onnxruntime_ep_tensorrt shared library.";
  return "";
}

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------

class CudaGraphTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ep_library_path_ = GetEpLibraryPath();
    if (ep_library_path_.empty()) {
      GTEST_SKIP() << "TRT_EP_LIBRARY_PATH not set, skipping CUDA graph tests.";
    }

    // Initialize ORT API (must be done before creating Env with ORT_API_MANUAL_INIT)
    Ort::InitApi();

    // Create the ORT environment
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "CudaGraphTest");

    // Build and write model
    auto model_data = CreateMulModel();
    model_path_ = WriteModelToTempFile(model_data);

    // Register the TRT plugin EP library
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
    // Clean up temp model file
    if (!model_path_.empty() && std::filesystem::exists(model_path_)) {
      std::filesystem::remove(model_path_);
    }
  }

  // Create a session with the TRT plugin EP, optionally enabling CUDA graph.
  Ort::Session CreateSession(bool enable_cuda_graph) {
    Ort::SessionOptions session_options;

    // Get available EP devices and find the TRT one
    auto all_ep_devices = env_->GetEpDevices();
    std::vector<Ort::ConstEpDevice> selected_devices;
    for (const auto& ep_device : all_ep_devices) {
      if (std::string(ep_device.EpName()) == ep_registration_name_) {
        selected_devices.push_back(ep_device);
        break;
      }
    }
    EXPECT_FALSE(selected_devices.empty()) << "No TRT EP device found";

    // EP options
    std::unordered_map<std::string, std::string> ep_options;
    if (enable_cuda_graph) {
      ep_options["trt_cuda_graph_enable"] = "1";
    }

    session_options.AppendExecutionProvider_V2(*env_, selected_devices, ep_options);

#ifdef _WIN32
    std::wstring wide_model_path = model_path_.wstring();
    return Ort::Session(*env_, wide_model_path.c_str(), session_options);
#else
    return Ort::Session(*env_, model_path_.c_str(), session_options);
#endif
  }

  std::unique_ptr<Ort::Env> env_;
  std::string ep_library_path_;
  std::string ep_registration_name_;
  std::filesystem::path model_path_;
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// Test basic CUDA graph capture and replay.
// Pattern: Run 1 captures the graph, Run 2 replays it, Run 3 updates input
// in-place and replays again.
// Aligned with CApiTest.basic_cuda_graph from onnxruntime test_inference.cc.
TEST_F(CudaGraphTest, BasicCudaGraph) {
  auto session = CreateSession(/*enable_cuda_graph=*/true);

  // Allocate input/output on CUDA device
  Ort::MemoryInfo mem_info("Cuda", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault);
  Ort::Allocator allocator(session, mem_info);

  constexpr int64_t num_elements = 3 * 2;
  const std::array<int64_t, 2> shape = {3, 2};

  // Pre-allocate device buffers
  auto input_alloc = allocator.GetAllocation(num_elements * sizeof(float));
  auto output_alloc = allocator.GetAllocation(num_elements * sizeof(float));

  // Initial input values
  std::array<float, num_elements> x_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  // Expected: Y = X * X
  std::array<float, num_elements> expected_y = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};

  // Copy input to device
  cudaMemcpy(input_alloc.get(), x_values.data(),
             num_elements * sizeof(float), cudaMemcpyHostToDevice);

  // Create bound tensors from pre-allocated device memory
  auto bound_x = Ort::Value::CreateTensor(
      mem_info, static_cast<float*>(input_alloc.get()),
      num_elements, shape.data(), shape.size());
  auto bound_y = Ort::Value::CreateTensor(
      mem_info, static_cast<float*>(output_alloc.get()),
      num_elements, shape.data(), shape.size());

  // Bind inputs/outputs
  Ort::IoBinding binding(session);
  binding.BindInput("X", bound_x);
  binding.BindOutput("Y", bound_y);

  // --- Run 1: This run triggers CUDA graph capture ---
  session.Run(Ort::RunOptions{}, binding);

  std::array<float, num_elements> y_values;
  cudaMemcpy(y_values.data(), output_alloc.get(),
             num_elements * sizeof(float), cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < num_elements; i++) {
    EXPECT_NEAR(y_values[i], expected_y[i], 1e-5f)
        << "Run 1 mismatch at index " << i;
  }

  // --- Run 2: This run replays the captured CUDA graph ---
  session.Run(Ort::RunOptions{}, binding);

  cudaMemcpy(y_values.data(), output_alloc.get(),
             num_elements * sizeof(float), cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < num_elements; i++) {
    EXPECT_NEAR(y_values[i], expected_y[i], 1e-5f)
        << "Run 2 (replay) mismatch at index " << i;
  }

  // --- Run 3: Update input in-place and replay the graph ---
  // CUDA graph replays use the same device pointers, so updating the input
  // buffer in-place will produce different outputs on the next replay.
  x_values = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
  expected_y = {100.0f, 400.0f, 900.0f, 1600.0f, 2500.0f, 3600.0f};

  cudaMemcpy(input_alloc.get(), x_values.data(),
             num_elements * sizeof(float), cudaMemcpyHostToDevice);

  binding.SynchronizeInputs();
  session.Run(Ort::RunOptions{}, binding);

  cudaMemcpy(y_values.data(), output_alloc.get(),
             num_elements * sizeof(float), cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < num_elements; i++) {
    EXPECT_NEAR(y_values[i], expected_y[i], 1e-5f)
        << "Run 3 (updated input replay) mismatch at index " << i;
  }

  binding.ClearBoundInputs();
  binding.ClearBoundOutputs();
}

// Test that inference works correctly without CUDA graph (baseline).
TEST_F(CudaGraphTest, WithoutCudaGraph) {
  auto session = CreateSession(/*enable_cuda_graph=*/false);

  std::array<float, 6> x_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::array<float, 6> expected_y = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};
  const std::array<int64_t, 2> shape = {3, 2};

  Ort::MemoryInfo cpu_mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto input_tensor = Ort::Value::CreateTensor(
      cpu_mem, x_values.data(), x_values.size(), shape.data(), shape.size());

  const char* input_names[] = {"X"};
  const char* output_names[] = {"Y"};
  auto outputs = session.Run(Ort::RunOptions{}, input_names, &input_tensor, 1, output_names, 1);

  ASSERT_EQ(outputs.size(), 1u);
  auto& output_tensor = outputs[0];
  const float* output_data = output_tensor.GetTensorData<float>();

  for (size_t i = 0; i < expected_y.size(); i++) {
    EXPECT_NEAR(output_data[i], expected_y[i], 1e-5f)
        << "Baseline mismatch at index " << i;
  }
}

// Test multiple sequential runs with CUDA graph to verify stability.
TEST_F(CudaGraphTest, MultipleReplays) {
  auto session = CreateSession(/*enable_cuda_graph=*/true);

  Ort::MemoryInfo mem_info("Cuda", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault);
  Ort::Allocator allocator(session, mem_info);

  constexpr int64_t num_elements = 3 * 2;
  const std::array<int64_t, 2> shape = {3, 2};

  auto input_alloc = allocator.GetAllocation(num_elements * sizeof(float));
  auto output_alloc = allocator.GetAllocation(num_elements * sizeof(float));

  std::array<float, num_elements> x_values = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  std::array<float, num_elements> expected_y = {4.0f, 9.0f, 16.0f, 25.0f, 36.0f, 49.0f};

  cudaMemcpy(input_alloc.get(), x_values.data(),
             num_elements * sizeof(float), cudaMemcpyHostToDevice);

  auto bound_x = Ort::Value::CreateTensor(
      mem_info, static_cast<float*>(input_alloc.get()),
      num_elements, shape.data(), shape.size());
  auto bound_y = Ort::Value::CreateTensor(
      mem_info, static_cast<float*>(output_alloc.get()),
      num_elements, shape.data(), shape.size());

  Ort::IoBinding binding(session);
  binding.BindInput("X", bound_x);
  binding.BindOutput("Y", bound_y);

  // Run multiple times — first run captures, rest replay
  constexpr int num_runs = 10;
  for (int run = 0; run < num_runs; run++) {
    session.Run(Ort::RunOptions{}, binding);

    std::array<float, num_elements> y_values;
    cudaMemcpy(y_values.data(), output_alloc.get(),
               num_elements * sizeof(float), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < num_elements; i++) {
      EXPECT_NEAR(y_values[i], expected_y[i], 1e-5f)
          << "Run " << (run + 1) << " mismatch at index " << i;
    }
  }

  binding.ClearBoundInputs();
  binding.ClearBoundOutputs();
}
