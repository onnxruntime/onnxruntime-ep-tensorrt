# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider option validation and functional tests for DLA-related options.

Category 1 — options that require trt_dla_enable=true (no hardware needed):
  trt_dla_gpu_fallback_enable
  trt_dla_enable_uint8_asymmetric_quantization
  trt_dla_adjust_for_dla
  trt_dla_transform_enable

Category 3 — DLA memory pool limit (hardware required):
  trt_dla_mem_pool_limit — valid 1 GiB limit succeeds; value below 512 MiB minimum raises

Category 4 — Static I/O buffers (hardware required):
  trt_dla_static_io_buffers — session creates and multiple consecutive runs produce
  correct, distinct outputs (verifies the skip-unregister path is stable across calls)

Category 5 — UINT8 asymmetric quantization (hardware required, TRT 10.11+):
  trt_dla_enable_uint8_asymmetric_quantization — DLA accepts a QDQ Conv whose
  activations use UINT8 with a non-zero zero_point (asymmetric); session creates
  and inference produces numerically correct output

Category 6 — Adjust for DLA (hardware required, TRT 10.16+):
  trt_dla_adjust_for_dla — enables kADJUST_FOR_DLA parser flag which rewrites
  model constructs into DLA-native forms.  Tested with the same UINT8 asymmetric
  QDQ model as Category 5: with both flags the model compiles; with only
  trt_dla_enable_uint8_asymmetric_quantization=1 but adjust_for_dla=0 the
  engine build fails because the DLA structural adjustments are absent.
"""

from __future__ import annotations

import pytest

import numpy as np

from conftest import RegisteredEp
from ort_helpers import create_session, run_in_large_stack, run_session_once


# ---------------------------------------------------------------------------
# Minimal model (used only to trigger session creation; content is irrelevant
# because validation errors are thrown before TRT parsing begins)
# ---------------------------------------------------------------------------

_DLA_BASE_OPTS = {
    "trt_fp16_enable":                              "1",
    "trt_dla_enable":                               "1",
    "trt_dla_core":                                 "0",
    "trt_dla_gpu_fallback_enable":                  "0",
    "trt_dla_adjust_for_dla":                       "1",
    "trt_dla_enable_uint8_asymmetric_quantization": "1",
}


def _minimal_model() -> bytes:
    import onnx
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    x      = helper.make_tensor_value_info("x",     TensorProto.FLOAT16, [1, 4, 8, 8])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1, 8, 6, 6])
    weight = numpy_helper.from_array(np.ones((8, 4, 3, 3), dtype=np.float16), name="weight")
    node   = helper.make_node("Conv", inputs=["x", "weight"], outputs=["output"])
    graph  = helper.make_graph([node], "minimal", [x], [output], initializer=[weight])
    model  = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = onnx.IR_VERSION
    return model.SerializeToString()


def _uint8_asymmetric_qdq_model() -> bytes:
    """QDQ Conv with UINT8 asymmetric activations (zero_point=128).

    Pattern:
      x (float32) -> Q(UINT8, scale=1/127, zp=128) -> DQ
                  -> Conv <- DQ <- (INT8 weight, scale=1/127, zp=0)
                  -> Q(UINT8, scale=1/127, zp=128) -> DQ -> output (float32)

    The non-zero zero_point (128) is the "asymmetric" part.  TRT rejects this for DLA
    unless kENABLE_UINT8_AND_ASYMMETRIC_QUANTIZATION_DLA is set (TRT 10.11+), which is
    what trt_dla_enable_uint8_asymmetric_quantization enables.

    Expected output with all-ones input:
      dequant(weight) = 1/127 per element; conv sums 4*3*3=36 taps
      -> each output element ≈ 36/127 ≈ 0.283
    """
    import onnx
    import onnx.shape_inference
    from onnx import TensorProto, helper, numpy_helper

    ACT_SCALE = np.float32(1.0 / 127.0)
    ACT_ZP    = np.uint8(128)           # asymmetric: non-zero zero_point
    W_SCALE   = np.float32(1.0 / 127.0)
    W_ZP      = np.int8(0)              # symmetric weight quantization

    x      = helper.make_tensor_value_info("x",      TensorProto.FLOAT, [1, 4, 8, 8])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8, 6, 6])

    act_scale = numpy_helper.from_array(ACT_SCALE,                             name="act_scale")
    act_zp    = numpy_helper.from_array(ACT_ZP,                                name="act_zp")
    w_q       = numpy_helper.from_array(np.ones((8, 4, 3, 3), dtype=np.int8),  name="w_q")
    w_scale   = numpy_helper.from_array(W_SCALE,                               name="w_scale")
    w_zp      = numpy_helper.from_array(W_ZP,                                  name="w_zp")
    out_scale = numpy_helper.from_array(ACT_SCALE,                             name="out_scale")
    out_zp    = numpy_helper.from_array(ACT_ZP,                                name="out_zp")

    act_q  = helper.make_node("QuantizeLinear",   ["x",        "act_scale", "act_zp"],  ["x_q"],       name="act_q")
    act_dq = helper.make_node("DequantizeLinear", ["x_q",      "act_scale", "act_zp"],  ["x_dq"],      name="act_dq")
    w_dq   = helper.make_node("DequantizeLinear", ["w_q",      "w_scale",   "w_zp"],    ["w_dq"],      name="w_dq")
    conv   = helper.make_node("Conv",             ["x_dq",     "w_dq"],                 ["conv_out"],  name="conv")
    out_q  = helper.make_node("QuantizeLinear",   ["conv_out", "out_scale", "out_zp"],  ["out_q"],     name="out_q")
    out_dq = helper.make_node("DequantizeLinear", ["out_q",    "out_scale", "out_zp"],  ["output"],    name="out_dq")

    graph = helper.make_graph(
        [act_q, act_dq, w_dq, conv, out_q, out_dq],
        "uint8_asymmetric_qdq",
        [x], [output],
        initializer=[act_scale, act_zp, w_q, w_scale, w_zp, out_scale, out_zp],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = onnx.IR_VERSION
    model = onnx.shape_inference.infer_shapes(model)
    return model.SerializeToString()


# ---------------------------------------------------------------------------
# Category 1 — options that require trt_dla_enable=true
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("option", [
    "trt_dla_gpu_fallback_enable",
    "trt_dla_enable_uint8_asymmetric_quantization",
    "trt_dla_adjust_for_dla",
    "trt_dla_transform_enable",
])
def test_dla_option_requires_dla_enable(registered_ep: RegisteredEp, option: str):
    """Setting a DLA sub-option without trt_dla_enable=true must prevent the TRT EP from loading.

    The EP raises a validation error in CreateEp which ORT catches and falls
    back to CPU, so no Python exception propagates.  The TRT EP being absent
    from session.get_providers() is the observable proof that validation fired.
    """
    session = create_session(
        registered_ep, _minimal_model(),
        provider_options={option: "1"},
    )
    providers = session.get_providers()
    assert registered_ep.ep_name not in providers, (
        f"{option}=1 without trt_dla_enable=1 should have caused EP validation failure "
        f"and CPU fallback, but TRT EP is still active. providers={providers}"
    )


# ---------------------------------------------------------------------------
# Category 3 — DLA memory pool limit (hardware required)
# ---------------------------------------------------------------------------

def test_dla_mem_pool_limit_valid(registered_ep: RegisteredEp, has_dla) -> None:
    """Explicit 1 GiB DLA memory pool limit succeeds and inference runs correctly."""
    opts = {**_DLA_BASE_OPTS, "trt_dla_mem_pool_limit": str(1 << 30)}  # 1 GiB
    session = run_in_large_stack(
        lambda: create_session(registered_ep, _minimal_model(), provider_options=opts)
    )
    outputs = run_session_once(session, feeds={"x": np.ones((1, 4, 8, 8), dtype=np.float16)})
    assert len(outputs) == 1
    assert outputs[0].shape == (1, 8, 6, 6), f"Unexpected output shape: {outputs[0].shape}"


def test_dla_mem_pool_limit_too_small(registered_ep: RegisteredEp, has_dla) -> None:
    """A DLA memory pool limit below the 512 MiB minimum must cause session creation to fail.

    256 MiB is below the minimum accepted value; the EP rejects this at config time.
    """
    opts = {**_DLA_BASE_OPTS, "trt_dla_mem_pool_limit": str(256 << 20)}  # 256 MiB — below 512 MiB minimum
    with pytest.raises(Exception):
        run_in_large_stack(
            lambda: create_session(registered_ep, _minimal_model(), provider_options=opts)
        )


# ---------------------------------------------------------------------------
# Category 4 — Static I/O buffers (hardware required)
# ---------------------------------------------------------------------------

def test_dla_static_io_buffers_single_run(registered_ep: RegisteredEp, has_dla) -> None:
    """Static I/O buffers mode: session creates and a single inference run succeeds."""
    opts = {**_DLA_BASE_OPTS, "trt_dla_static_io_buffers": "1"}
    session = run_in_large_stack(
        lambda: create_session(registered_ep, _minimal_model(), provider_options=opts)
    )
    outputs = run_session_once(session, feeds={"x": np.ones((1, 4, 8, 8), dtype=np.float16)})
    assert len(outputs) == 1
    assert outputs[0].shape == (1, 8, 6, 6), f"Unexpected output shape: {outputs[0].shape}"


def test_dla_static_io_buffers_multiple_runs(registered_ep: RegisteredEp, has_dla) -> None:
    """Static I/O buffers mode must produce correct, distinct outputs across consecutive runs.

    When trt_dla_static_io_buffers=1 the EP skips clearing DLA tensor addresses between
    runs.  Running with different input values verifies that the skip-unregister path
    does not cause output corruption or stale-buffer reads.
    """
    opts = {**_DLA_BASE_OPTS, "trt_dla_static_io_buffers": "1"}
    session = run_in_large_stack(
        lambda: create_session(registered_ep, _minimal_model(), provider_options=opts)
    )

    # All-ones weights, so output = sum over 4*3*3=36 kernel taps * input value.
    # Expected per-element values: ones->36, twos->72, zeros->0.
    # Distinct expected outputs prove that the skip-unregister path is not serving
    # stale buffers from a previous run.
    cases = [
        (np.ones( (1, 4, 8, 8), dtype=np.float16), np.float16(36.0)),
        (np.full( (1, 4, 8, 8), 2.0, dtype=np.float16), np.float16(72.0)),
        (np.zeros((1, 4, 8, 8), dtype=np.float16), np.float16(0.0)),
    ]
    for i, (feed, expected_val) in enumerate(cases):
        outputs = session.run(None, {"x": feed})
        assert len(outputs) == 1, f"Run {i}: expected 1 output, got {len(outputs)}"
        assert outputs[0].shape == (1, 8, 6, 6), (
            f"Run {i}: unexpected shape {outputs[0].shape}"
        )
        assert np.allclose(outputs[0], expected_val, atol=1.0), (
            f"Run {i}: expected all elements ≈ {expected_val}, got {outputs[0]}"
        )


# ---------------------------------------------------------------------------
# Category 5 — UINT8 asymmetric quantization (hardware required, TRT 10.11+)
# ---------------------------------------------------------------------------

def test_uint8_asymmetric_quantization_succeeds(registered_ep: RegisteredEp, has_dla) -> None:
    """DLA compiles and runs a QDQ Conv with UINT8 asymmetric activations.

    The model uses zero_point=128 (mid-range of [0, 255]) for both input and
    output activations, which is the "asymmetric" case that DLA rejects without
    trt_dla_enable_uint8_asymmetric_quantization=1 (requires TRT 10.11+).

    Expected output with all-ones float32 input:
      dequant(input)  ≈ 1.0  (Q→DQ round-trips cleanly at scale=1/127, zp=128)
      dequant(weight) = 1/127 per INT8 element
      conv sums 4*3*3 = 36 taps → each output element ≈ 36/127 ≈ 0.283
    """
    session = run_in_large_stack(
        lambda: create_session(registered_ep, _uint8_asymmetric_qdq_model(), provider_options=_DLA_BASE_OPTS)
    )
    feeds = {"x": np.ones((1, 4, 8, 8), dtype=np.float32)}
    outputs = session.run(None, feeds)
    assert len(outputs) == 1
    assert outputs[0].shape == (1, 8, 6, 6), f"Unexpected output shape: {outputs[0].shape}"
    expected = np.float32(36.0 / 127.0)  # ≈ 0.283
    assert np.allclose(outputs[0], expected, atol=0.1), (
        f"Output values out of expected range: min={outputs[0].min():.4f}, max={outputs[0].max():.4f}"
    )


def test_uint8_asymmetric_quantization_disabled_fails(registered_ep: RegisteredEp, has_dla) -> None:
    """DLA rejects a UINT8 asymmetric QDQ Conv when the option is disabled.

    Without kENABLE_UINT8_AND_ASYMMETRIC_QUANTIZATION_DLA the TRT ONNX parser
    does not recognise UINT8 nodes with non-zero zero_point as DLA-compatible.
    With GPU fallback also disabled, the engine build must fail.
    """
    opts = {
        "trt_fp16_enable":                              "1",
        "trt_dla_enable":                               "1",
        "trt_dla_core":                                 "0",
        "trt_dla_gpu_fallback_enable":                  "0",
        "trt_dla_adjust_for_dla":                       "1",
        "trt_dla_enable_uint8_asymmetric_quantization": "0",  # disabled — must fail
    }
    with pytest.raises(Exception):
        run_in_large_stack(
            lambda: create_session(registered_ep, _uint8_asymmetric_qdq_model(), provider_options=opts)
        )


# ---------------------------------------------------------------------------
# Category 6 — Adjust for DLA (hardware required, TRT 10.16+)
# ---------------------------------------------------------------------------

def test_dla_adjust_for_dla_succeeds(registered_ep: RegisteredEp, has_dla) -> None:
    """DLA compiles the UINT8 asymmetric QDQ model when both adjust_for_dla and
    uint8_asymmetric_quantization flags are enabled together.

    _DLA_BASE_OPTS already sets both flags; this test confirms their combined effect.
    """
    session = run_in_large_stack(
        lambda: create_session(registered_ep, _uint8_asymmetric_qdq_model(), provider_options=_DLA_BASE_OPTS)
    )
    feeds = {"x": np.ones((1, 4, 8, 8), dtype=np.float32)}
    outputs = session.run(None, feeds)
    assert len(outputs) == 1
    assert outputs[0].shape == (1, 8, 6, 6), f"Unexpected output shape: {outputs[0].shape}"
    expected = np.float32(36.0 / 127.0)  # ≈ 0.283
    assert np.allclose(outputs[0], expected, atol=0.1), (
        f"Output values out of expected range: min={outputs[0].min():.4f}, max={outputs[0].max():.4f}"
    )


def test_dla_adjust_for_dla_disabled_fails(registered_ep: RegisteredEp, has_dla) -> None:
    """DLA rejects the UINT8 asymmetric QDQ model when adjust_for_dla is disabled.

    Even with trt_dla_enable_uint8_asymmetric_quantization=1 the engine build must
    fail because kADJUST_FOR_DLA is absent and the structural DLA adjustments are
    not applied.  GPU fallback is disabled so TRT cannot fall back silently.
    """
    opts = {
        "trt_fp16_enable":                              "1",
        "trt_dla_enable":                               "1",
        "trt_dla_core":                                 "0",
        "trt_dla_gpu_fallback_enable":                  "0",
        "trt_dla_adjust_for_dla":                       "0",  # disabled — must fail
        "trt_dla_enable_uint8_asymmetric_quantization": "1",
    }
    with pytest.raises(Exception):
        run_in_large_stack(
            lambda: create_session(registered_ep, _uint8_asymmetric_qdq_model(), provider_options=opts)
        )
