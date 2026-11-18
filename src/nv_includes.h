// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

// File to include the required TRT headers with workarounds for warnings we can't fix or not fixed yet.
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4100)  // Ignore warning C4100: unreferenced formal parameter
#pragma warning(disable : 4996)  // Ignore warning C4996: 'nvinfer1::IPluginV2' was declared deprecated
#endif

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>

// TensorRT 11.3 and newer removed the legacy precision builder flags
// (kFP16/kBF16/kINT8). Use the EP's strongly typed path instead, where precision
// comes from the ONNX graph and the builder flags are not available.
#if NV_TENSORRT_MAJOR > 11 || (NV_TENSORRT_MAJOR == 11 && NV_TENSORRT_MINOR >= 3)
#define ORT_TENSORRT_STRONGLY_TYPED
#endif

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

// SubGraph_t and SubGraphCollection_t were removed from NvOnnxParser.h in TRT 11.1.
// Define them here so ORT's internal subgraph tracking still compiles.
#if NV_TENSORRT_MAJOR >= 11
#include <utility>
#include <vector>
using SubGraph_t = std::pair<std::vector<size_t>, bool>;
using SubGraphCollection_t = std::vector<SubGraph_t>;
#endif
