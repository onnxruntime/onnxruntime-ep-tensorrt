# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from dataclasses import dataclass, field
from types import ModuleType

import pytest


@dataclass
class RegisteredEp:
    ort: ModuleType
    ep_name: str
    library_path: str
    registered_by_fixture: bool = field(default=False)


@pytest.fixture(scope="session")
def registered_ep() -> RegisteredEp:
    """Register the TRT EP library for the test session and unregister on teardown."""
    ort = pytest.importorskip("onnxruntime")
    ep_pkg = pytest.importorskip("onnxruntime_ep_tensorrt")

    lib = ep_pkg.get_library_path()
    ep_name = ep_pkg.get_ep_names()[0]

    if not ep_name:
        pytest.skip("onnxruntime_ep_tensorrt.get_ep_names() returned an empty name")
    if not os.path.isfile(lib):
        pytest.skip(f"TRT EP library not found: {lib}")
    if not hasattr(ort, "register_execution_provider_library"):
        pytest.skip("onnxruntime build does not expose register_execution_provider_library")
    if not hasattr(ort, "get_ep_devices"):
        pytest.skip("onnxruntime build does not expose get_ep_devices")

    registered_by_fixture = False
    try:
        ort.register_execution_provider_library(ep_name, lib)
        registered_by_fixture = True
    except Exception as exc:
        # EP may already be registered (e.g. test re-run in the same process).
        # Continue only if a device is actually visible; otherwise the session is unusable.
        devices = [d for d in getattr(ort, "get_ep_devices", lambda: [])()
                   if getattr(d, "ep_name", None) == ep_name]
        if not devices:
            pytest.skip(f"Failed to register TRT EP library: {exc}")

    yield RegisteredEp(ort=ort, ep_name=ep_name, library_path=lib,
                       registered_by_fixture=registered_by_fixture)

    if registered_by_fixture and hasattr(ort, "unregister_execution_provider_library"):
        try:
            ort.unregister_execution_provider_library(ep_name)
        except Exception:
            pass


@pytest.fixture(scope="session")
def has_dla(registered_ep: RegisteredEp) -> None:
    """Skip the test if DLA hardware is not available."""
    if not os.environ.get("TRT_EP_HAS_DLA"):
        pytest.skip("TRT_EP_HAS_DLA not set — no DLA hardware available")


@pytest.fixture(scope="session")
def has_dla_transforms(registered_ep: RegisteredEp) -> dict:
    """Skip if EP not built with DLA transforms; return the provider option to enable them."""
    if os.environ.get("TRT_EP_HAS_DLA_TRANSFORMS") != "1":
        pytest.skip("TRT_EP_HAS_DLA_TRANSFORMS not set — EP not built with USE_DLA_TRANSFORMS")
    return {"trt_dla_transform_enable": "1"}


@pytest.fixture(scope="session")
def has_two_dla_cores(has_dla) -> None:
    """Skip if fewer than 2 DLA cores are available.

    Set TRT_EP_DLA_CORE_COUNT to the number of DLA cores on the device.
    Defaults to 1 if not set.
    """
    count = int(os.environ.get("TRT_EP_DLA_CORE_COUNT", "1"))
    if count < 2:
        pytest.skip("TRT_EP_DLA_CORE_COUNT < 2 — only one DLA core available")
