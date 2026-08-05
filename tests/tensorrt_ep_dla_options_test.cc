// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// C++ mirror of python/tests/test_dla_options.py.
//
// Category 1 — DLA sub-options require trt_dla_enable=1 (no hardware needed):
//   trt_dla_gpu_fallback_enable, trt_dla_enable_uint8_asymmetric_quantization,
//   trt_dla_adjust_for_dla, trt_dla_transform_enable (USE_DLA_TRANSFORMS only)
//
// Category 3 — DLA memory pool limit (hardware required):
//   valid 1 GiB limit succeeds; 1 KiB limit on a large model must fail
//
// Category 4 — Static I/O buffers (hardware required):
//   single run and multiple consecutive runs with distinct inputs produce correct outputs
//
// Category 5 — UINT8 asymmetric quantization (hardware required, TRT 10.11+):
//   enabled succeeds with correct numerics; disabled must fail
//
// Category 6 — Adjust for DLA (hardware required, TRT 10.16+):
//   enabled succeeds; disabled must fail even with uint8_asymmetric_quantization=1
//
// All tests run only with --target=dla.

#include <gtest/gtest.h>
#include <onnx/onnx_pb.h>
#include <onnx/shape_inference/implementation.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "onnxruntime_cxx_api.h"
#include "test_config.h"

// ---------------------------------------------------------------------------
// FP16 helpers
// ---------------------------------------------------------------------------

// IEEE-754 half-precision bit patterns
static constexpr uint16_t kFp16Zero = 0x0000u;  // 0.0
static constexpr uint16_t kFp16One  = 0x3C00u;  // 1.0
static constexpr uint16_t kFp16Two  = 0x4000u;  // 2.0

// Convert an IEEE-754 half-precision value to float.
static float Fp16ToFloat(uint16_t h) {
    uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0)       bits = sign | (mant << 13);
    else if (exp == 31) bits = sign | 0x7F800000u | (mant << 13);
    else                bits = sign | ((exp + 112u) << 23) | (mant << 13);
    float f;
    std::memcpy(&f, &bits, sizeof f);
    return f;
}

// Create a CPU FLOAT16 tensor filled with a constant FP16 bit pattern.
static Ort::Value MakeFp16Tensor(const std::vector<int64_t>& shape, uint16_t fill) {
    size_t count = 1;
    for (auto d : shape) count *= static_cast<size_t>(d);
    Ort::AllocatorWithDefaultOptions alloc;
    auto tensor = Ort::Value::CreateTensor(alloc, shape.data(), shape.size(),
                                            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
    auto* raw = static_cast<uint16_t*>(tensor.GetTensorMutableRawData());
    std::fill(raw, raw + count, fill);
    return tensor;
}

// ---------------------------------------------------------------------------
// Model builders
// ---------------------------------------------------------------------------

// Minimal FP16 Conv: x[1,4,8,8] * weight[8,4,3,3]=ones -> output[1,8,6,6]
// Matches _minimal_model() in test_dla_options.py.
static std::string CreateMinimalFp16ConvModel() {
    using namespace ONNX_NAMESPACE;
    ModelProto model;
    model.set_ir_version(8);
    model.add_opset_import()->set_version(13);

    auto* g = model.mutable_graph();
    g->set_name("minimal_fp16_conv");

    auto fp16_vi = [](ValueInfoProto* vi, const std::string& name,
                       std::initializer_list<int64_t> dims) {
        vi->set_name(name);
        auto* t = vi->mutable_type()->mutable_tensor_type();
        t->set_elem_type(TensorProto_DataType_FLOAT16);
        for (int64_t d : dims) t->mutable_shape()->add_dim()->set_dim_value(d);
    };
    fp16_vi(g->add_input(),  "x",      {1, 4, 8, 8});
    fp16_vi(g->add_output(), "output", {1, 8, 6, 6});

    // Weight: FP16 [8,4,3,3] all ones  (0x3C00 per element)
    {
        auto* init = g->add_initializer();
        init->set_name("weight");
        init->set_data_type(TensorProto_DataType_FLOAT16);
        for (int64_t d : {8, 4, 3, 3}) init->add_dims(d);
        constexpr size_t N = 8 * 4 * 3 * 3;
        std::string raw(N * 2, '\0');
        for (size_t i = 0; i < N; ++i) {
            raw[i * 2]     = static_cast<char>(kFp16One & 0xFF);
            raw[i * 2 + 1] = static_cast<char>(kFp16One >> 8);
        }
        init->set_raw_data(raw);
    }

    auto* node = g->add_node();
    node->set_op_type("Conv");
    node->set_name("conv");
    node->add_input("x");
    node->add_input("weight");
    node->add_output("output");

    std::string buf;
    model.SerializeToString(&buf);
    return buf;
}

// QDQ Conv with UINT8 asymmetric activations (zero_point=128).
//
// Pattern:
//   x(float) -> Q(uint8, scale=1/127, zp=128) -> DQ
//            -> Conv <- DQ(int8 weight=ones, scale=1/127, zp=0)
//            -> Q(uint8, scale=1/127, zp=128) -> DQ -> output(float)
//
// Expected output with all-ones float input: each element ≈ 36/127 ≈ 0.283
// (36 kernel taps * 1/127 per weight element).
//
// Shape inference is run before serialization so TRT can inspect value_info dtypes.
// Matches _uint8_asymmetric_qdq_model() in test_dla_options.py.
static std::string CreateUint8AsymmetricQdqConvModel() {
    using namespace ONNX_NAMESPACE;
    ModelProto model;
    model.set_ir_version(8);
    model.add_opset_import()->set_version(13);

    auto* g = model.mutable_graph();
    g->set_name("uint8_asymmetric_qdq");

    auto float_vi = [](ValueInfoProto* vi, const std::string& name,
                        std::initializer_list<int64_t> dims) {
        vi->set_name(name);
        auto* t = vi->mutable_type()->mutable_tensor_type();
        t->set_elem_type(TensorProto_DataType_FLOAT);
        for (int64_t d : dims) t->mutable_shape()->add_dim()->set_dim_value(d);
    };
    float_vi(g->add_input(),  "x",      {1, 4, 8, 8});
    float_vi(g->add_output(), "output", {1, 8, 6, 6});

    constexpr float kScale = 1.0f / 127.0f;

    auto float_scalar = [&](const std::string& name, float v) {
        auto* t = g->add_initializer();
        t->set_name(name);
        t->set_data_type(TensorProto_DataType_FLOAT);
        t->add_float_data(v);
    };
    auto uint8_scalar = [&](const std::string& name, uint8_t v) {
        auto* t = g->add_initializer();
        t->set_name(name);
        t->set_data_type(TensorProto_DataType_UINT8);
        t->add_int32_data(static_cast<int32_t>(v));
    };
    auto int8_scalar = [&](const std::string& name, int8_t v) {
        auto* t = g->add_initializer();
        t->set_name(name);
        t->set_data_type(TensorProto_DataType_INT8);
        t->add_int32_data(static_cast<int32_t>(v));
    };

    float_scalar("act_scale", kScale);
    uint8_scalar("act_zp",    128);     // asymmetric: non-zero zero_point
    float_scalar("w_scale",   kScale);
    int8_scalar ("w_zp",      0);
    float_scalar("out_scale", kScale);
    uint8_scalar("out_zp",    128);

    // w_q: int8 [8,4,3,3] all ones
    {
        auto* init = g->add_initializer();
        init->set_name("w_q");
        init->set_data_type(TensorProto_DataType_INT8);
        for (int64_t d : {8, 4, 3, 3}) init->add_dims(d);
        init->set_raw_data(std::string(8 * 4 * 3 * 3, '\x01'));
    }

    auto node = [&](const char* op,
                    std::initializer_list<const char*> ins,
                    std::initializer_list<const char*> outs,
                    const char* name) {
        auto* n = g->add_node();
        n->set_op_type(op);
        n->set_name(name);
        for (const char* i : ins)  n->add_input(i);
        for (const char* o : outs) n->add_output(o);
    };

    node("QuantizeLinear",   {"x",        "act_scale", "act_zp"}, {"x_q"},      "act_q");
    node("DequantizeLinear", {"x_q",      "act_scale", "act_zp"}, {"x_dq"},     "act_dq");
    node("DequantizeLinear", {"w_q",      "w_scale",   "w_zp"},   {"w_dq"},     "w_dq");
    node("Conv",             {"x_dq",     "w_dq"},                {"conv_out"}, "conv");
    node("QuantizeLinear",   {"conv_out", "out_scale", "out_zp"}, {"out_q"},    "out_q");
    node("DequantizeLinear", {"out_q",    "out_scale", "out_zp"}, {"output"},   "out_dq");

    onnx::shape_inference::InferShapes(model);

    std::string buf;
    model.SerializeToString(&buf);
    return buf;
}

// ---------------------------------------------------------------------------
// File I/O helper
// ---------------------------------------------------------------------------

static std::filesystem::path WriteTempModel(const std::string& data,
                                            const std::string& filename) {
    auto path = std::filesystem::temp_directory_path() / filename;
    std::ofstream ofs(path, std::ios::binary);
    ofs.write(data.data(), static_cast<std::streamsize>(data.size()));
    return path;
}

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

class DlaOptionsTest : public ::testing::Test {
 protected:
    void SetUp() override {
        if (trt_test::g_ep_lib_path.empty()) {
            GTEST_SKIP() << "EP library not found, skipping DLA option tests.";
        }
        if (trt_test::g_target != "dla") {
            GTEST_SKIP() << "--target is not 'dla' — pass --target=dla to run DLA tests.";
        }
    }

    void TearDown() override {
        for (const auto& p : temp_files_)
            if (std::filesystem::exists(p))
                std::filesystem::remove(p);
    }

    // Create a session with the TRT EP and given options.
    // disable_cpu_ep_fallback=1 is always set so EP failures surface immediately
    // rather than silently falling back to CPU.
    Ort::Session CreateSession(
        const std::filesystem::path& model_path,
        const std::unordered_map<std::string, std::string>& ep_opts) {
        Ort::SessionOptions so;

        auto devices = trt_test::g_ort_env->GetEpDevices();
        std::vector<Ort::ConstEpDevice> selected;
        for (const auto& d : devices)
            if (std::string(d.EpName()) == trt_test::kEpName) {
                selected.push_back(d);
                break;
            }
        EXPECT_FALSE(selected.empty()) << "No TRT EP device found";
        so.AppendExecutionProvider_V2(*trt_test::g_ort_env, selected, ep_opts);
        so.AddConfigEntry("session.disable_cpu_ep_fallback", "1");

#ifdef _WIN32
        return Ort::Session(*trt_test::g_ort_env, model_path.wstring().c_str(), so);
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

// DLA base options used by Categories 3–6 (mirrors _DLA_BASE_OPTS in Python).
static const std::unordered_map<std::string, std::string> kDlaBaseOpts = {
    {"trt_fp16_enable",                              "1"},
    {"trt_dla_enable",                               "1"},
    {"trt_dla_core",                                 "0"},
    {"trt_dla_gpu_fallback_enable",                  "0"},
    {"trt_dla_adjust_for_dla",                       "1"},
    {"trt_dla_enable_uint8_asymmetric_quantization", "1"},
};

// ---------------------------------------------------------------------------
// Category 1 — DLA sub-options require trt_dla_enable=1
//
// Setting a DLA sub-option without trt_dla_enable=1 must trigger EP validation
// failure in CreateEp.  With disable_cpu_ep_fallback=1 this surfaces as an
// exception rather than silent CPU fallback (the Python equivalent checks that
// the TRT EP is absent from session.get_providers()).
// ---------------------------------------------------------------------------

TEST_F(DlaOptionsTest, DlaGpuFallbackEnableRequiresDlaEnable) {
    auto model = TrackTemp(CreateMinimalFp16ConvModel(), "cat1_gpu_fallback.onnx");
    EXPECT_THROW(CreateSession(model, {{"trt_dla_gpu_fallback_enable", "1"}}),
                 Ort::Exception);
}

TEST_F(DlaOptionsTest, Uint8AsymmetricQuantizationRequiresDlaEnable) {
    auto model = TrackTemp(CreateMinimalFp16ConvModel(), "cat1_uint8_asym.onnx");
    EXPECT_THROW(CreateSession(model, {{"trt_dla_enable_uint8_asymmetric_quantization", "1"}}),
                 Ort::Exception);
}

TEST_F(DlaOptionsTest, AdjustForDlaRequiresDlaEnable) {
    auto model = TrackTemp(CreateMinimalFp16ConvModel(), "cat1_adjust_for_dla.onnx");
    EXPECT_THROW(CreateSession(model, {{"trt_dla_adjust_for_dla", "1"}}),
                 Ort::Exception);
}

#ifdef USE_DLA_TRANSFORMS
TEST_F(DlaOptionsTest, TransformEnableRequiresDlaEnable) {
    auto model = TrackTemp(CreateMinimalFp16ConvModel(), "cat1_transform_enable.onnx");
    EXPECT_THROW(CreateSession(model, {{"trt_dla_transform_enable", "1"}}),
                 Ort::Exception);
}
#endif

// ---------------------------------------------------------------------------
// Category 3 — DLA memory pool limit (hardware required)
// ---------------------------------------------------------------------------

TEST_F(DlaOptionsTest, MemPoolLimitValid) {
    auto opts = kDlaBaseOpts;
    opts["trt_dla_mem_pool_limit"] = std::to_string(1LL << 30);  // 1 GiB
    auto model = TrackTemp(CreateMinimalFp16ConvModel(), "cat3_pool_valid.onnx");
    auto session = CreateSession(model, opts);

    auto input = MakeFp16Tensor({1, 4, 8, 8}, kFp16One);
    const char* in_names[]  = {"x"};
    const char* out_names[] = {"output"};
    auto outputs = session.Run(Ort::RunOptions{}, in_names, &input, 1, out_names, 1);

    ASSERT_EQ(outputs.size(), 1u);
    auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    ASSERT_EQ(shape.size(), 4u);
    EXPECT_EQ(shape[1], 8);
    EXPECT_EQ(shape[2], 6);
    EXPECT_EQ(shape[3], 6);
}

TEST_F(DlaOptionsTest, MemPoolLimitTooSmall) {
    auto opts = kDlaBaseOpts;
    // 256 MiB is below the 512 MiB minimum; the EP must reject this at config time.
    opts["trt_dla_mem_pool_limit"] = std::to_string(256LL << 20);
    auto model = TrackTemp(CreateMinimalFp16ConvModel(), "cat3_pool_too_small.onnx");
    EXPECT_THROW(CreateSession(model, opts), Ort::Exception);
}

// ---------------------------------------------------------------------------
// Category 4 — Static I/O buffers (hardware required)
// ---------------------------------------------------------------------------

TEST_F(DlaOptionsTest, StaticIoBuffersSingleRun) {
    auto opts = kDlaBaseOpts;
    opts["trt_dla_static_io_buffers"] = "1";
    auto model = TrackTemp(CreateMinimalFp16ConvModel(), "cat4_static_io_single.onnx");
    auto session = CreateSession(model, opts);

    auto input = MakeFp16Tensor({1, 4, 8, 8}, kFp16One);
    const char* in_names[]  = {"x"};
    const char* out_names[] = {"output"};
    auto outputs = session.Run(Ort::RunOptions{}, in_names, &input, 1, out_names, 1);

    ASSERT_EQ(outputs.size(), 1u);
    auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    EXPECT_EQ(shape[1], 8);
    EXPECT_EQ(shape[2], 6);
    EXPECT_EQ(shape[3], 6);
}

TEST_F(DlaOptionsTest, StaticIoBuffersMultipleRuns) {
    // All-ones weights -> output = sum of 4*3*3=36 taps * input_value per element.
    // Distinct expected outputs across runs prove skip-unregister does not serve stale buffers.
    auto opts = kDlaBaseOpts;
    opts["trt_dla_static_io_buffers"] = "1";
    auto model = TrackTemp(CreateMinimalFp16ConvModel(), "cat4_static_io_multi.onnx");
    auto session = CreateSession(model, opts);

    const char* in_names[]  = {"x"};
    const char* out_names[] = {"output"};

    struct Case { uint16_t fp16_fill; float expected; };
    const Case cases[] = {
        {kFp16One,  36.0f},
        {kFp16Two,  72.0f},
        {kFp16Zero,  0.0f},
    };

    for (size_t i = 0; i < std::size(cases); ++i) {
        auto input = MakeFp16Tensor({1, 4, 8, 8}, cases[i].fp16_fill);
        auto outputs = session.Run(Ort::RunOptions{}, in_names, &input, 1, out_names, 1);

        ASSERT_EQ(outputs.size(), 1u) << "run " << i;
        auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        int64_t count = 1;
        for (auto d : shape) count *= d;

        const auto* raw = static_cast<const uint16_t*>(outputs[0].GetTensorRawData());
        for (int64_t j = 0; j < count; ++j) {
            float actual = Fp16ToFloat(raw[j]);
            EXPECT_NEAR(actual, cases[i].expected, 1.0f)
                << "run " << i << " element " << j;
        }
    }
}

// ---------------------------------------------------------------------------
// Category 5 — UINT8 asymmetric quantization (hardware required, TRT 10.11+)
// ---------------------------------------------------------------------------

TEST_F(DlaOptionsTest, Uint8AsymmetricQuantizationSucceeds) {
    auto model = TrackTemp(CreateUint8AsymmetricQdqConvModel(), "cat5_uint8_asym_ok.onnx");
    auto session = CreateSession(model, kDlaBaseOpts);

    std::vector<float> input_data(1 * 4 * 8 * 8, 1.0f);
    const std::array<int64_t, 4> shape = {1, 4, 8, 8};
    Ort::MemoryInfo cpu_mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input = Ort::Value::CreateTensor(
        cpu_mem, input_data.data(), input_data.size(), shape.data(), shape.size());

    const char* in_names[]  = {"x"};
    const char* out_names[] = {"output"};
    auto outputs = session.Run(Ort::RunOptions{}, in_names, &input, 1, out_names, 1);
    ASSERT_EQ(outputs.size(), 1u);

    auto out_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    ASSERT_EQ(out_shape.size(), 4u);
    EXPECT_EQ(out_shape[1], 8);
    EXPECT_EQ(out_shape[2], 6);
    EXPECT_EQ(out_shape[3], 6);

    // Each output element ≈ 36/127 ≈ 0.283
    constexpr float kExpected = 36.0f / 127.0f;
    const float* out_data = outputs[0].GetTensorData<float>();
    int64_t count = 1;
    for (auto d : out_shape) count *= d;
    for (int64_t i = 0; i < count; ++i)
        EXPECT_NEAR(out_data[i], kExpected, 0.1f) << "element " << i;
}

TEST_F(DlaOptionsTest, Uint8AsymmetricQuantizationDisabledFails) {
    // Without kENABLE_UINT8_AND_ASYMMETRIC_QUANTIZATION_DLA, TRT rejects UINT8
    // nodes with non-zero zero_point as DLA-incompatible.
    auto opts = kDlaBaseOpts;
    opts["trt_dla_enable_uint8_asymmetric_quantization"] = "0";
    auto model = TrackTemp(CreateUint8AsymmetricQdqConvModel(), "cat5_uint8_asym_fail.onnx");
    EXPECT_THROW(CreateSession(model, opts), Ort::Exception);
}

// ---------------------------------------------------------------------------
// Category 6 — Adjust for DLA (hardware required, TRT 10.16+)
// ---------------------------------------------------------------------------

TEST_F(DlaOptionsTest, AdjustForDlaSucceeds) {
    // kDlaBaseOpts already sets both adjust_for_dla=1 and uint8_asymmetric_quantization=1.
    auto model = TrackTemp(CreateUint8AsymmetricQdqConvModel(), "cat6_adjust_ok.onnx");
    auto session = CreateSession(model, kDlaBaseOpts);

    std::vector<float> input_data(1 * 4 * 8 * 8, 1.0f);
    const std::array<int64_t, 4> shape = {1, 4, 8, 8};
    Ort::MemoryInfo cpu_mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input = Ort::Value::CreateTensor(
        cpu_mem, input_data.data(), input_data.size(), shape.data(), shape.size());

    const char* in_names[]  = {"x"};
    const char* out_names[] = {"output"};
    auto outputs = session.Run(Ort::RunOptions{}, in_names, &input, 1, out_names, 1);
    ASSERT_EQ(outputs.size(), 1u);

    constexpr float kExpected = 36.0f / 127.0f;
    const float* out_data = outputs[0].GetTensorData<float>();
    auto out_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t count = 1;
    for (auto d : out_shape) count *= d;
    for (int64_t i = 0; i < count; ++i)
        EXPECT_NEAR(out_data[i], kExpected, 0.1f) << "element " << i;
}

TEST_F(DlaOptionsTest, AdjustForDlaDisabledFails) {
    // Without kADJUST_FOR_DLA the structural DLA adjustments are absent; engine
    // build must fail even with uint8_asymmetric_quantization enabled.
    auto opts = kDlaBaseOpts;
    opts["trt_dla_adjust_for_dla"] = "0";
    auto model = TrackTemp(CreateUint8AsymmetricQdqConvModel(), "cat6_adjust_fail.onnx");
    EXPECT_THROW(CreateSession(model, opts), Ort::Exception);
}
