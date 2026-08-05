# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DLA transform tests for the TensorRT plugin EP.

Tests two DLA transforms in combination on a synthetic A16W8 MatMul subgraph
that replicates the 6-node pattern from PSH_ver_2.0.9.quant.onnx (nodes 102-118).
Dimensions are reduced for fast DLA compilation; scale/ZP values match the real model.

  data (float32) [1, 1, 16, 32]
    -> Q (uint16, scale=5.39e-4, zp=31408)          [activation A16 quantize]
    -> DQ(uint16)                                    [activation dequantize]
    -> MatMul  <-- DQ(weight_uint8 [32, 8],          [weight W8 dequantize]
                       scale=2.50e-3, zp=127)
    -> Q (uint16, scale=1.51e-4, zp=32795)           [output A16 quantize]
    -> DQ(uint16)                                    [output dequantize]
    -> output (float32) [1, 1, 16, 8]

Without transforms: TRT parser rejects the UINT16 zero-point initializers -> throws.
With transforms:
  Step  1  RemoveQDQ                     strips both UINT16 Q+DQ pairs
  Step 12  MatMulToTransposeConvTranspose converts the exposed float MatMul to
                                         Transpose+Conv+Transpose -> DLA accepts.

Required environment variables (both must be set to "1"):
  TRT_EP_HAS_DLA            — DLA hardware is present
  TRT_EP_HAS_DLA_TRANSFORMS — EP was built with -Donnxruntime_ep_tensorrt_DLA_TRANSFORMS=ON
"""

from __future__ import annotations

import numpy as np
import onnx
import onnx.shape_inference
from onnx import TensorProto, helper, numpy_helper

from conftest import RegisteredEp
from ort_helpers import make_session_options, run_in_large_stack


def _make_a16w8_matmul_model() -> bytes:
    """Synthetic A16W8 MatMul subgraph replicating PSH_ver_2.0.9.quant.onnx nodes 102-118.

    Dimensions are reduced ([1,1,16,32] x [32,8]) for fast DLA compilation.
    Scale/ZP values are taken directly from the real PSH model initializers.
    Shape inference is run before serialization so the DLA transforms library
    can identify the UINT16 Q/DQ pairs by tensor dtype in value_info.
    """
    data   = helper.make_tensor_value_info("data",   TensorProto.FLOAT, [1, 1, 16, 32])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 16, 8])

    weight_q  = numpy_helper.from_array(np.full((32, 8), 127, dtype=np.uint8),              name="weight_quantized")
    weight_sc = numpy_helper.from_array(np.array(0.0024978937581181526,  dtype=np.float32), name="weight_scale")
    weight_zp = numpy_helper.from_array(np.array(127,   dtype=np.uint8),                    name="weight_zero_point")
    act_sc    = numpy_helper.from_array(np.array(0.0005390419391915202,  dtype=np.float32), name="act_scale")
    act_zp    = numpy_helper.from_array(np.array(31408, dtype=np.uint16),                   name="act_zero_point")
    out_sc    = numpy_helper.from_array(np.array(0.00015090894885361195, dtype=np.float32), name="out_scale")
    out_zp    = numpy_helper.from_array(np.array(32795, dtype=np.uint16),                   name="out_zero_point")

    weight_dq   = helper.make_node("DequantizeLinear", ["weight_quantized", "weight_scale", "weight_zero_point"], ["weight_dequantized"], name="weight_DequantizeLinear")
    act_q_node  = helper.make_node("QuantizeLinear",   ["data",             "act_scale",    "act_zero_point"],    ["act_quantized"],      name="act_QuantizeLinear")
    act_dq_node = helper.make_node("DequantizeLinear", ["act_quantized",    "act_scale",    "act_zero_point"],    ["act_dequantized"],    name="act_DequantizeLinear")
    mm          = helper.make_node("MatMul",           ["act_dequantized",  "weight_dequantized"],                ["matmul_output"],      name="MatMul")
    out_q_node  = helper.make_node("QuantizeLinear",   ["matmul_output",    "out_scale",    "out_zero_point"],    ["out_quantized"],      name="out_QuantizeLinear")
    out_dq_node = helper.make_node("DequantizeLinear", ["out_quantized",    "out_scale",    "out_zero_point"],    ["output"],             name="out_DequantizeLinear")

    graph = helper.make_graph(
        [weight_dq, act_q_node, act_dq_node, mm, out_q_node, out_dq_node],
        "a16w8_matmul",
        [data], [output],
        initializer=[weight_q, weight_sc, weight_zp, act_sc, act_zp, out_sc, out_zp],
    )
    # opset 21: UINT16 zero-points in QuantizeLinear/DequantizeLinear require opset 21+
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    # Pin to IR version 8 — the DLA transforms library checker supports up to IR 12
    model.ir_version = 8
    model = onnx.shape_inference.infer_shapes(model)
    return model.SerializeToString()


_DLA_BASE_OPTS = {
    "trt_fp16_enable":                              "1",
    "trt_dla_enable":                               "1",
    "trt_dla_core":                                 "0",
    "trt_dla_gpu_fallback_enable":                  "0",
    "trt_dla_adjust_for_dla":                       "1",
    "trt_dla_enable_uint8_asymmetric_quantization": "1",
}

_SESSION_CFG = {"session.disable_cpu_ep_fallback": "1"}


def test_a16w8_matmul_with_transforms_succeeds(registered_ep: RegisteredEp, has_dla, has_dla_transforms):
    model = _make_a16w8_matmul_model()
    opts = {**_DLA_BASE_OPTS, **has_dla_transforms}
    so = make_session_options(registered_ep, provider_options=opts, session_config=_SESSION_CFG)
    session = run_in_large_stack(lambda: registered_ep.ort.InferenceSession(model, sess_options=so))
    feeds = {"data": np.ones((1, 1, 16, 32), dtype=np.float32)}
    outputs = session.run(None, feeds)
    assert len(outputs) == 1
    assert outputs[0].shape == (1, 1, 16, 8), f"Unexpected shape: {outputs[0].shape}"
