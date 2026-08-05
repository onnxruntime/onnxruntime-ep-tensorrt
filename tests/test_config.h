// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Shared globals for trt_ep_tests, populated by test_main.cc before any test runs.
// ORT_API_MANUAL_INIT is set as a compile definition for the whole target; callers
// must NOT define or undef it themselves.

#pragma once

#include "onnxruntime_cxx_api.h"

#include <filesystem>
#include <memory>
#include <string>

namespace trt_test {

inline constexpr const char* kEpName = "TRTPluginEP";

// Populated in main(); empty if the EP library could not be found or registered.
extern std::unique_ptr<Ort::Env> g_ort_env;
extern std::filesystem::path     g_ep_lib_path;

// Set from --target=gpu|dla on the command line (default: "gpu").
// DLA-specific tests skip when this is not "dla".
extern std::string g_target;

#ifdef USE_DLA_TRANSFORMS
// Path to dla_transforms.dll located adjacent to the test binary.
// Empty if the file is absent; DLA transforms tests GTEST_FAIL() in that case.
extern std::filesystem::path g_dla_transforms_dll_path;
#endif

}  // namespace trt_test
