// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// C++ mirror of python/tests/test_dla_transforms.py::test_a16w8_matmul_with_transforms_succeeds.
//
// Builds a synthetic A16W8 MatMul subgraph (UINT16 activations, UINT8 weights),
// applies DLA transforms via the EP, and verifies that inference produces a
// result with the expected output shape on DLA.
//
// Required command line flags (parsed by test_main.cc):
//   --target=dla   — must be passed; test is skipped otherwise
//
// EP library and dla_transforms.dll are resolved automatically from the binary directory
// by test_main.cc (post-build copy steps place both files there). The DLA transforms
// test FAILS (not skips) when --target=dla is set but dla_transforms.dll is absent,
// to surface misconfiguration explicitly.
//
// Compiled only when USE_DLA_TRANSFORMS is defined (i.e. the EP was built with
// -Donnxruntime_ep_tensorrt_DLA_TRANSFORMS=ON).

#ifdef USE_DLA_TRANSFORMS

#include <gtest/gtest.h>
#include <onnx/onnx_pb.h>
#include <onnx/shape_inference/implementation.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "onnxruntime_cxx_api.h"
#include "test_config.h"

// Build a synthetic A16W8 MatMul model matching _make_a16w8_matmul_model() in
// python/tests/test_dla_transforms.py.
//
//   data (float32) [1,1,16,32]
//     -> Q (uint16, scale=5.39e-4, zp=31408)      [activation A16 quantize]
//     -> DQ(uint16)
//     -> MatMul  <- DQ(weight uint8[32,8],          [weight W8 dequantize]
//                       scale=2.50e-3, zp=127)
//     -> Q (uint16, scale=1.51e-4, zp=32795)       [output A16 quantize]
//     -> DQ(uint16)
//     -> output (float32) [1,1,16,8]
//
// Opset 21 is required for UINT16 zero-points in QuantizeLinear/DequantizeLinear.
// Shape inference is run before serialization so the DLA transforms library can
// identify UINT16 Q/DQ pairs by tensor dtype in value_info.
static std::string CreateA16W8MatMulModel() {
  using namespace ONNX_NAMESPACE;

  ModelProto model;
  model.set_ir_version(8);  // IR 8; transforms library supports up to IR 12

  auto* opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(21);  // UINT16 zero-points require opset 21+

  GraphProto* graph = model.mutable_graph();
  graph->set_name("a16w8_matmul");

  // ── graph input / output ─────────────────────────────────────────────────
  auto set_float_shape = [](ValueInfoProto* vi, const std::string& name,
                            std::initializer_list<int64_t> dims) {
    vi->set_name(name);
    auto* t = vi->mutable_type()->mutable_tensor_type();
    t->set_elem_type(TensorProto_DataType_FLOAT);
    for (int64_t d : dims)
      t->mutable_shape()->add_dim()->set_dim_value(d);
  };

  set_float_shape(graph->add_input(),  "data",   {1, 1, 16, 32});
  set_float_shape(graph->add_output(), "output", {1, 1, 16,  8});

  // ── initialiser helpers ──────────────────────────────────────────────────
  auto float_scalar = [&](const std::string& name, float v) {
    auto* init = graph->add_initializer();
    init->set_name(name);
    init->set_data_type(TensorProto_DataType_FLOAT);
    init->add_float_data(v);
  };
  auto uint8_scalar = [&](const std::string& name, uint8_t v) {
    auto* init = graph->add_initializer();
    init->set_name(name);
    init->set_data_type(TensorProto_DataType_UINT8);
    init->add_int32_data(static_cast<int32_t>(v));
  };
  auto uint16_scalar = [&](const std::string& name, uint16_t v) {
    auto* init = graph->add_initializer();
    init->set_name(name);
    init->set_data_type(TensorProto_DataType_UINT16);
    init->add_int32_data(static_cast<int32_t>(v));
  };

  // ── initialisers ─────────────────────────────────────────────────────────
  {
    // weight_quantized: uint8 [32, 8] — all elements = 127
    auto* init = graph->add_initializer();
    init->set_name("weight_quantized");
    init->set_data_type(TensorProto_DataType_UINT8);
    init->add_dims(32);
    init->add_dims(8);
    std::string raw(32 * 8, static_cast<char>(127));
    init->set_raw_data(raw);
  }

  float_scalar ("weight_scale",       0.0024978937581181526f);
  uint8_scalar ("weight_zero_point",  127);
  float_scalar ("act_scale",          0.0005390419391915202f);
  uint16_scalar("act_zero_point",     31408);
  float_scalar ("out_scale",          0.00015090894885361195f);
  uint16_scalar("out_zero_point",     32795);

  // ── nodes ────────────────────────────────────────────────────────────────
  auto add_node = [&](const char* op,
                      std::initializer_list<const char*> ins,
                      std::initializer_list<const char*> outs,
                      const char* name) {
    auto* n = graph->add_node();
    n->set_op_type(op);
    n->set_name(name);
    for (const char* i : ins)  n->add_input(i);
    for (const char* o : outs) n->add_output(o);
  };

  add_node("DequantizeLinear",
           {"weight_quantized", "weight_scale", "weight_zero_point"},
           {"weight_dequantized"}, "weight_DequantizeLinear");

  add_node("QuantizeLinear",
           {"data", "act_scale", "act_zero_point"},
           {"act_quantized"}, "act_QuantizeLinear");

  add_node("DequantizeLinear",
           {"act_quantized", "act_scale", "act_zero_point"},
           {"act_dequantized"}, "act_DequantizeLinear");

  add_node("MatMul",
           {"act_dequantized", "weight_dequantized"},
           {"matmul_output"}, "MatMul");

  add_node("QuantizeLinear",
           {"matmul_output", "out_scale", "out_zero_point"},
           {"out_quantized"}, "out_QuantizeLinear");

  add_node("DequantizeLinear",
           {"out_quantized", "out_scale", "out_zero_point"},
           {"output"}, "out_DequantizeLinear");

  // Shape inference must run before serialisation so that the DLA transforms
  // library can identify UINT16 Q/DQ pairs from value_info tensor dtypes.
  onnx::shape_inference::InferShapes(model);

  std::string buf;
  model.SerializeToString(&buf);
  return buf;
}

static std::filesystem::path WriteTempModel(const std::string& data,
                                            const std::string& filename) {
  auto path = std::filesystem::temp_directory_path() / filename;
  std::ofstream ofs(path, std::ios::binary);
  ofs.write(data.data(), static_cast<std::streamsize>(data.size()));
  return path;
}

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------

class DlaTransformsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Gate 1 — EP library
    if (trt_test::g_ep_lib_path.empty()) {
      GTEST_SKIP() << "EP library not found, skipping DLA transforms tests.";
    }

    // Gate 2 — DLA hardware
    if (trt_test::g_target != "dla") {
      GTEST_SKIP() << "--target is not 'dla' — pass --target=dla to run DLA tests.";
    }

    // Gate 3 — DLA transforms DLL must be present adjacent to the binary.
    // This is a hard failure: if --target=dla was passed but the DLL is absent,
    // the test environment is misconfigured rather than simply lacking hardware.
    if (trt_test::g_dla_transforms_dll_path.empty()) {
      GTEST_FAIL() << "dla_transforms.dll not found adjacent to the test binary. "
                      "Ensure the post-build copy step ran or copy the DLL manually.";
      return;
    }
  }

  void TearDown() override {
    for (const auto& p : temp_files_)
      if (std::filesystem::exists(p))
        std::filesystem::remove(p);
  }

  Ort::Session CreateSession(
      const std::filesystem::path& model_path,
      const std::unordered_map<std::string, std::string>& ep_opts) {
    Ort::SessionOptions so;

    auto devices = trt_test::g_ort_env->GetEpDevices();
    std::vector<Ort::ConstEpDevice> selected;
    for (const auto& d : devices) {
      if (std::string(d.EpName()) == trt_test::kEpName) {
        selected.push_back(d);
        break;
      }
    }
    EXPECT_FALSE(selected.empty()) << "No TRT EP device found";
    so.AppendExecutionProvider_V2(*trt_test::g_ort_env, selected, ep_opts);
    so.AddConfigEntry("session.disable_cpu_ep_fallback", "1");

#ifdef _WIN32
    std::wstring wide = model_path.wstring();
    return Ort::Session(*trt_test::g_ort_env, wide.c_str(), so);
#else
    return Ort::Session(*trt_test::g_ort_env, model_path.c_str(), so);
#endif
  }

  std::filesystem::path TrackTemp(const std::string& data, const std::string& name) {
    auto p = WriteTempModel(data, name);
    temp_files_.push_back(p);
    return p;
  }

  std::vector<std::filesystem::path> temp_files_;
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// Mirror of test_a16w8_matmul_with_transforms_succeeds (test_dla_transforms.py).
//
// Verifies that the DLA transforms pipeline (RemoveQDQ + MatMulToTransposeConvTranspose)
// allows a UINT16-quantised A16W8 MatMul subgraph to compile and run on DLA.
// Without trt_dla_transform_enable=1 the TRT parser would reject the UINT16
// zero-point initialisers and throw during session creation.
TEST_F(DlaTransformsTest, A16W8MatMulWithTransformsSucceeds) {
  auto model_data = CreateA16W8MatMulModel();
  auto model_path = TrackTemp(model_data, "dla_transforms_a16w8_matmul.onnx");

  const std::unordered_map<std::string, std::string> ep_opts = {
      {"trt_fp16_enable",                              "1"},
      {"trt_dla_enable",                               "1"},
      {"trt_dla_core",                                 "0"},
      {"trt_dla_gpu_fallback_enable",                  "0"},
      {"trt_dla_adjust_for_dla",                       "1"},
      {"trt_dla_enable_uint8_asymmetric_quantization", "1"},
      {"trt_dla_transform_enable",                     "1"},
  };

  Ort::Session session = CreateSession(model_path, ep_opts);

  // Input: data float32 [1, 1, 16, 32] — all ones
  constexpr size_t kElemCount = 1 * 1 * 16 * 32;
  std::vector<float> input_data(kElemCount, 1.0f);
  const std::array<int64_t, 4> input_shape = {1, 1, 16, 32};

  Ort::MemoryInfo cpu_mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto input_tensor = Ort::Value::CreateTensor(
      cpu_mem, input_data.data(), input_data.size(),
      input_shape.data(), input_shape.size());

  const char* input_names[]  = {"data"};
  const char* output_names[] = {"output"};

  auto outputs = session.Run(Ort::RunOptions{},
                             input_names, &input_tensor, 1,
                             output_names, 1);
  ASSERT_EQ(outputs.size(), 1u);

  auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
  ASSERT_EQ(shape.size(), 4u);
  EXPECT_EQ(shape[0], 1)  << "batch dim mismatch";
  EXPECT_EQ(shape[1], 1)  << "channel dim mismatch";
  EXPECT_EQ(shape[2], 16) << "spatial dim mismatch";
  EXPECT_EQ(shape[3], 8)  << "output feature dim mismatch";
}

#endif  // USE_DLA_TRANSFORMS
