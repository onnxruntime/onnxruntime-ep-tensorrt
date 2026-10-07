// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace trt_ep {

// On Windows, loads the delay-linked TensorRT runtime libraries from the
// EP directory first, then the Windows DLL search path. On other platforms this is a no-op.
//
// Throws std::runtime_error when a required Windows runtime DLL is missing or
// cannot be loaded. Call this before invoking any TensorRT API.
void EnsureTensorRtDependenciesLoaded();

// TensorRT loads nvdla_compiler.dll later with a filename-only LoadLibrary
// call. Preload it with the shared search policy so the later call reuses it.
void EnsureDlaCompilerDependencyLoaded();

// Preloads the cuDLA user-mode runtime, preferring the copy staged by CMake.
void EnsureCuDlaDependencyLoaded();

}  // namespace trt_ep
