// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Custom GTest entry point for trt_ep_tests.
//
// Responsibilities:
//   - Parse --target=gpu|dla from argv before handing off to InitGoogleTest.
//   - Resolve the EP shared library path: binary-adjacent first, then the
//     compile-time EP_LIB_PATH constant baked in by CMake.
//   - Register the TRT EP library with the shared Ort::Env so all test
//     fixtures can reuse it without repeated register/unregister cycles.
//   - When built with USE_DLA_TRANSFORMS: detect dla_transforms.dll adjacent
//     to the binary and record its path for DLA transforms tests.

#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

#include "test_config.h"

#ifdef _WIN32
#include <windows.h>
#endif

namespace trt_test {
std::unique_ptr<Ort::Env> g_ort_env;
std::filesystem::path     g_ep_lib_path;
std::string               g_target = "gpu";
#ifdef USE_DLA_TRANSFORMS
std::filesystem::path     g_dla_transforms_dll_path;
#endif
}  // namespace trt_test

// ---------------------------------------------------------------------------
// EP library resolution
// ---------------------------------------------------------------------------

static std::filesystem::path resolve_ep_lib(const char* argv0) {
  // EP_LIB_PATH is the absolute path baked in at compile time by CMake via
  // -DEP_LIB_PATH="$<TARGET_FILE:onnxruntime_ep_tensorrt>".
  const std::filesystem::path build_path(EP_LIB_PATH);
  // Primary: look for the DLL by filename in the same directory as this binary.
  // A post-build copy step ensures it is placed there.
  const auto local = std::filesystem::absolute(argv0).parent_path() / build_path.filename();
  if (std::filesystem::is_regular_file(local))
    return local;
  // Fallback: use the full compile-time path (e.g. running from source tree).
  return build_path;
}

static void register_ep(Ort::Env& env, const std::filesystem::path& ep_lib) {
  if (!std::filesystem::is_regular_file(ep_lib)) {
    std::cerr << "[setup] EP library not found at " << ep_lib
              << " — tests requiring the EP will be skipped.\n";
    return;
  }
  try {
#ifdef _WIN32
    env.RegisterExecutionProviderLibrary(trt_test::kEpName, ep_lib.wstring().c_str());
#else
    env.RegisterExecutionProviderLibrary(trt_test::kEpName, ep_lib.c_str());
#endif
    std::cout << "[setup] Registered TRT EP from " << ep_lib << "\n";
    trt_test::g_ep_lib_path = ep_lib;
  } catch (const Ort::Exception& ex) {
    std::cerr << "[setup] Failed to register TRT EP: " << ex.what()
              << " — tests requiring the EP will be skipped.\n";
  }
}

// ---------------------------------------------------------------------------
// DLA transforms DLL resolution (Windows only, USE_DLA_TRANSFORMS builds)
// ---------------------------------------------------------------------------

#ifdef USE_DLA_TRANSFORMS
static void resolve_dla_transforms_dll(const char* argv0) {
  const auto local =
      std::filesystem::absolute(argv0).parent_path() / "dla_transforms.dll";
  if (std::filesystem::is_regular_file(local)) {
    trt_test::g_dla_transforms_dll_path = local;
    std::cout << "[setup] Found dla_transforms.dll at " << local << "\n";
  } else {
    std::cerr << "[setup] dla_transforms.dll not found adjacent to test binary"
              << " — DLA transforms tests will fail.\n";
  }
}
#endif

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
  // Strip --target=<value> before InitGoogleTest sees argv, so GTest doesn't
  // reject it as an unknown flag.
  for (int i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], "--target=", 9) == 0) {
      trt_test::g_target = argv[i] + 9;
      for (int j = i; j < argc - 1; ++j) argv[j] = argv[j + 1];
      --argc;
      --i;
    }
  }

  Ort::InitApi();
  trt_test::g_ort_env =
      std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "trt_ep_tests");

  const auto ep_lib = resolve_ep_lib(argv[0]);
  register_ep(*trt_test::g_ort_env, ep_lib);

#ifdef USE_DLA_TRANSFORMS
  resolve_dla_transforms_dll(argv[0]);
#endif

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
