# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Basic inference test for the TensorRT plugin EP.

Model: single Conv op, FP16, 4D tensors
  input  "input"  FLOAT16 [1, 4, 8, 8]
  weight "weight" FLOAT16 [8, 4, 3, 3]  (zero initializer)
  output "output" FLOAT16 [1, 8, 6, 6]

Environment variables:
  TRT_EP_HAS_DLA  set to "1" to run with DLA-specific provider options;
                  otherwise runs on GPU.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from conftest import RegisteredEp
from ort_helpers import make_session_options


def _conv_fp16_model() -> bytes:
    import onnx
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    inp    = helper.make_tensor_value_info("input",  TensorProto.FLOAT16, [1, 4, 8, 8])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1, 8, 6, 6])
    weight = numpy_helper.from_array(
        np.ones((8, 4, 3, 3), dtype=np.float16), name="weight"
    )
    node  = helper.make_node("Conv", inputs=["input", "weight"], outputs=["output"])
    graph = helper.make_graph([node], "conv_fp16", [inp], [output], initializer=[weight])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = onnx.IR_VERSION
    return model.SerializeToString()


def test_conv_fp16_inference(registered_ep: RegisteredEp):
    dla = os.environ.get("TRT_EP_HAS_DLA") == "1"

    provider_options: dict[str, str] = {"trt_fp16_enable": "1"}
    if dla:
        provider_options.update({
            "trt_dla_enable":                               "1",
            "trt_dla_core":                                 "0",
            "trt_dla_gpu_fallback_enable":                  "0",
            "trt_dla_adjust_for_dla":                       "1",
            "trt_dla_enable_uint8_asymmetric_quantization": "1",
        })

    so = make_session_options(registered_ep, provider_options=provider_options)
    session = registered_ep.ort.InferenceSession(_conv_fp16_model(), sess_options=so)

    # Non-zero input ensures data flows through the conv.
    feeds = {"input": np.ones((1, 4, 8, 8), dtype=np.float16)}
    outputs = session.run(None, feeds)

    assert len(outputs) == 1, f"Expected 1 output, got {len(outputs)}"

    # Shape check
    assert outputs[0].shape == (1, 8, 6, 6), f"Unexpected output shape: {outputs[0].shape}"

    # Accuracy check: ones input * ones weight → each output = C_in×kH×kW = 4×3×3 = 36.0
    expected = np.full((1, 8, 6, 6), 36.0, dtype=np.float16)
    np.testing.assert_allclose(outputs[0], expected, atol=1.0,
                               err_msg="Conv output does not match expected value of 36.0")
