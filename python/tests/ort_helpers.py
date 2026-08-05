# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, Mapping, TypeVar

import numpy as np
import pytest

_T = TypeVar("_T")


def run_in_large_stack(fn: Callable[[], _T], stack_mb: int = 64) -> _T:
    """Run fn() in a thread with a larger stack and return its result.

    TRT/DLA engine compilation can exhaust the default Python thread stack
    (~1 MB on ARM64 Windows). This helper runs the callable in a new thread
    with a 64 MB stack and re-raises any exception so pytest.raises() works.
    """
    result: list = [None]
    exc: list = [None]

    def _worker() -> None:
        try:
            result[0] = fn()
        except BaseException as e:  # noqa: BLE001
            exc[0] = e

    old_size = threading.stack_size(stack_mb * 1024 * 1024)
    t = threading.Thread(target=_worker, daemon=True)
    threading.stack_size(old_size)
    t.start()
    t.join()
    if exc[0] is not None:
        raise exc[0]
    return result[0]

from conftest import RegisteredEp


def get_trt_ep_devices(registered_ep: RegisteredEp):
    """Return OrtEpDevice entries matching the registered TRT EP name."""
    devices = [
        d for d in registered_ep.ort.get_ep_devices()
        if getattr(d, "ep_name", None) == registered_ep.ep_name
    ]
    if not devices:
        pytest.skip(f"No OrtEpDevice found for {registered_ep.ep_name}")
    return devices


def make_session_options(
    registered_ep: RegisteredEp,
    provider_options: Mapping[str, str] | None = None,
    session_config: Mapping[str, str] | None = None,
):
    """Build SessionOptions with the TRT EP appended via add_provider_for_devices."""
    ort = registered_ep.ort
    so = ort.SessionOptions()

    for key, value in (session_config or {}).items():
        so.add_session_config_entry(str(key), str(value))

    if not hasattr(so, "add_provider_for_devices"):
        pytest.skip("onnxruntime.SessionOptions does not expose add_provider_for_devices")

    so.add_provider_for_devices(
        get_trt_ep_devices(registered_ep),
        {str(k): str(v) for k, v in (provider_options or {}).items()},
    )
    return so


def create_session(
    registered_ep: RegisteredEp,
    model_path_or_bytes,
    provider_options: Mapping[str, str] | None = None,
    session_config: Mapping[str, str] | None = None,
):
    """Create an InferenceSession with the TRT EP and the given options."""
    so = make_session_options(registered_ep, provider_options=provider_options,
                              session_config=session_config)
    return registered_ep.ort.InferenceSession(model_path_or_bytes, sess_options=so)


def _concrete_shape(shape: list, override: list | None = None) -> list[int]:
    if override is not None:
        return list(override)
    return [d if isinstance(d, int) and d > 0 else 1 for d in shape]


def _numpy_dtype(ort_type: str):
    mapping = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(int8)": np.int8,
        "tensor(uint8)": np.uint8,
        "tensor(uint16)": np.uint16,
        "tensor(int16)": np.int16,
        "tensor(uint32)": np.uint32,
        "tensor(uint64)": np.uint64,
        "tensor(bool)": np.bool_,
    }
    dt = mapping.get(ort_type)
    if dt is None:
        pytest.skip(f"No numpy mapping for ORT type {ort_type}")
    return dt


def make_zero_feeds(
    session,
    shape_overrides: Mapping[str, list[int]] | None = None,
) -> dict:
    """Build a dict of zeroed numpy arrays matching the session's input signatures."""
    feeds = {}
    for inp in session.get_inputs():
        shape = _concrete_shape(inp.shape, (shape_overrides or {}).get(inp.name))
        dtype = _numpy_dtype(inp.type)
        feeds[inp.name] = np.zeros(shape, dtype=dtype)
    return feeds


def run_session_once(session, feeds=None, shape_overrides=None):
    if feeds is None:
        feeds = make_zero_feeds(session, shape_overrides)
    return session.run(None, feeds)
