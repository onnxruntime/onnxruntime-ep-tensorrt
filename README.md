# TensorRT Plugin Execution Provider

The TensorRT plugin Execution Provider (EP) implements the [ORT EP plugin ABI](https://onnxruntime.ai/docs/reference/ep-abi.html) introduced in ONNX Runtime 1.23.0, enabling NVIDIA TensorRT acceleration for ONNX models. It is migrated from the in-tree [TensorRT EP](https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/core/providers/tensorrt) and exposes the same set of provider options and features.

Unlike the legacy in-tree EP, this plugin EP is built as a **standalone shared library** (`onnxruntime_ep_tensorrt.dll` / `libonnxruntime_ep_tensorrt.so`) and does **not** need to be compiled together with ONNX Runtime. It only links against the ONNX Runtime shared library (`onnxruntime.dll` / `libonnxruntime.so`).

Supported platforms: **Linux** and **Windows** (Debug / Release).

## Contents

| Path | Description |
|------|-------------|
| `CMakeLists.txt` | Build configuration for the plugin EP and optional unit tests. |
| `src/` | C++ source code for the plugin EP. |
| `tests/` | GTest-based unit tests (basic inference, CUDA graph, engine caching, etc.). |
| `python/` | Python package and example usage script. See [`python/readme.md`](python/readme.md). |
| `csharp/` | C# NuGet package and sample application. See [`csharp/readme.md`](csharp/readme.md). |

## Prerequisites

- **ONNX Runtime** ≥ 1.23.0 (headers + shared library)
- **NVIDIA TensorRT** (10.x or 11.x)
- **CUDA Toolkit** (with `nvcc`)
- **CMake** ≥ 3.25

## Build Instructions

### On Windows

```bash
mkdir build && cd build
cmake -S ../ -B ./ -DCMAKE_BUILD_TYPE=Release ^
  -DTENSORRT_HOME=C:/path/to/TensorRT ^
  -DORT_HOME=C:/path/to/onnxruntime ^
  -DTRT_MAJOR_VERSION=11
cmake --build ./ --config Release
```

### On Linux

```bash
mkdir build && cd build
cmake -S ../ -B ./ -DCMAKE_BUILD_TYPE=Release \
  -DTENSORRT_HOME=/path/to/TensorRT \
  -DORT_HOME=/path/to/onnxruntime \
  -DTRT_MAJOR_VERSION=11 \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build ./ --config Release
```

> **Note:** `ORT_HOME` must contain `include/` and `lib/` subdirectories:
> ```
> ORT_HOME/
> ├── include/
> │   ├── onnxruntime_c_api.h
> │   ├── onnxruntime_ep_c_api.h
> │   ├── onnxruntime_cxx_api.h
> │   └── ...
> └── lib/
>     ├── onnxruntime.dll (or libonnxruntime.so)
>     ├── onnxruntime.lib (Windows only)
>     └── ...
> ```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `ORT_HOME` | auto-download | Path to ONNX Runtime package (include + lib). |
| `TENSORRT_HOME` | *(required)* | Path to TensorRT installation. |
| `TRT_MAJOR_VERSION` | `10` | TensorRT major version (affects library names). |
| `CMAKE_CUDA_ARCHITECTURES` | `86` | Target CUDA architectures (e.g., `80`, `86`, `89`, `90`). |
| `onnxruntime_ep_tensorrt_BUILD_TESTS` | `OFF` | Build unit tests (requires GTest, fetched automatically). |
| `onnxruntime_ep_tensorrt_OBJECT_CACHE` | `ON` | Use sccache/ccache if available. |

## Usage

The plugin EP follows the ORT EP plugin ABI workflow:

1. **Register** the plugin EP library with the ORT environment.
2. **Discover** available EP devices.
3. **Append** the EP to session options with provider-specific options.
4. **Create** an inference session and run the model.
5. **Unregister** the library after all sessions using it have been released.

### C/C++ API

```cpp
#include "onnxruntime_cxx_api.h"

Ort::InitApi();
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MyApp");

// 1. Register the plugin EP library
env.RegisterExecutionProviderLibrary("TRTPluginEP", "path/to/onnxruntime_ep_tensorrt.dll");

// 2. Find the EP device
auto all_devices = env.GetEpDevices();
std::vector<Ort::ConstEpDevice> trt_devices;
for (const auto& d : all_devices) {
  if (std::string(d.EpName()) == "TRTPluginEP") {
    trt_devices.push_back(d);
    break;
  }
}

// 3. Create session with EP options
Ort::SessionOptions session_options;
std::unordered_map<std::string, std::string> ep_options = {
  {"trt_fp16_enable", "1"},
  {"trt_engine_cache_enable", "1"},
  {"trt_engine_cache_path", "./cache"},
};
session_options.AppendExecutionProvider_V2(env, trt_devices, ep_options);

// 4. Run inference
Ort::Session session(env, "model.onnx", session_options);
auto outputs = session.Run(Ort::RunOptions{}, input_names, inputs, num_inputs, output_names, num_outputs);

// 5. Unregister (after all sessions are released)
session = Ort::Session{nullptr};  // release session first
env.UnregisterExecutionProviderLibrary("TRTPluginEP");
```

### Python

Install the helper package (see [`python/readme.md`](python/readme.md) for build instructions):

```python
import numpy as np
import onnxruntime as ort
import onnxruntime_ep_tensorrt as tensorrt_ep

# 1. Register the plugin EP library
ep_lib_path = tensorrt_ep.get_library_path()
ep_name = tensorrt_ep.get_ep_name()     # "TensorRTPluginExecutionProvider"
ort.register_execution_provider_library(ep_name, ep_lib_path)

# 2. Select an EP device
all_devices = ort.get_ep_devices()
trt_devices = [d for d in all_devices if d.ep_name == ep_name]

# 3. Create session with EP options
sess_options = ort.SessionOptions()
ep_options = {
    "trt_fp16_enable": "1",
    "trt_engine_cache_enable": "1",
}
sess_options.add_provider_for_devices(trt_devices, ep_options)

# 4. Run inference
session = ort.InferenceSession("model.onnx", sess_options=sess_options)
output = session.run([], {"input": input_data})

# 5. Unregister
del session
ort.unregister_execution_provider_library(ep_name)
```

### C\#

Install the NuGet package (see [`csharp/readme.md`](csharp/readme.md) for build instructions):

```csharp
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.EP.TensorRT;

// 1. Register the plugin EP library
var env = OrtEnv.Instance();
string epLibPath = TensorRTEp.GetLibraryPath();
string epName = TensorRTEp.GetEpName();
env.RegisterExecutionProviderLibrary(epName, epLibPath);

// 2. Find the EP device
OrtEpDevice? epDevice = env.GetEpDevices()
    .FirstOrDefault(d => d.EpName == epName);

// 3. Create session with EP options
using var sessionOptions = new SessionOptions();
sessionOptions.AppendExecutionProvider(env, new[] { epDevice },
    new Dictionary<string, string> {
        { "trt_fp16_enable", "1" },
    });

// 4. Run inference
using var session = new InferenceSession("model.onnx", sessionOptions);
using var results = session.Run(runOptions, inputNames, inputValues, session.OutputNames);

// 5. Unregister
env.UnregisterExecutionProviderLibrary(epName);
```

### Quick Test with `onnxruntime_perf_test`

For a quick smoke test without writing code, use the ORT perf test tool:

```bash
onnxruntime_perf_test \
  --plugin_ep_libs "TRTPluginEP|path/to/onnxruntime_ep_tensorrt.dll" \
  --plugin_eps TRTPluginEP \
  -r 1 path/to/model.onnx
```

## Provider Options

Provider options are passed as key-value string pairs when creating a session. These are the same options supported by the legacy in-tree TensorRT EP.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `device_id` | int | `0` | CUDA device ID. |
| `trt_max_partition_iterations` | int | `1000` | Maximum iterations for TensorRT graph partitioning. |
| `trt_min_subgraph_size` | int | `1` | Minimum number of nodes in a subgraph to be accelerated by TRT. |
| `trt_max_workspace_size` | size_t | `1073741824` (1 GB) | Maximum workspace size for TensorRT engine building. |
| `trt_fp16_enable` | bool | `0` | Enable FP16 precision. |
| `trt_int8_enable` | bool | `0` | Enable INT8 precision. |
| `trt_bf16_enable` | bool | `0` | Enable BF16 precision. |
| `trt_int8_calibration_table_name` | string | `""` | Path to INT8 calibration table. |
| `trt_int8_use_native_calibration_table` | bool | `0` | Use native TRT calibration table format. |
| `trt_dla_enable` | bool | `0` | Enable DLA (Deep Learning Accelerator). |
| `trt_dla_core` | int | `0` | DLA core to use. |
| `trt_engine_cache_enable` | bool | `0` | Enable TensorRT engine caching. |
| `trt_engine_cache_path` | string | `""` | Directory path for cached engines. |
| `trt_engine_cache_prefix` | string | `""` | Filename prefix for cached engines. |
| `trt_dump_subgraphs` | bool | `0` | Dump subgraphs to files for debugging. |
| `trt_force_sequential_engine_build` | bool | `0` | Build TRT engines sequentially (for debugging). |
| `trt_context_memory_sharing_enable` | bool | `0` | Share context memory across TRT subgraphs. |
| `trt_layer_norm_fp32_fallback` | bool | `0` | Force FP32 for LayerNorm (for accuracy). |
| `trt_timing_cache_enable` | bool | `0` | Enable timing cache to speed up engine building. |
| `trt_timing_cache_path` | string | `""` | Path for the timing cache file. |
| `trt_force_timing_cache` | bool | `0` | Fail if timing cache is not found. |
| `trt_detailed_build_log` | bool | `0` | Print detailed TRT engine build log. |
| `trt_build_heuristics_enable` | bool | `0` | Enable builder heuristics for faster build. |
| `trt_sparsity_enable` | bool | `0` | Enable structured sparsity. |
| `trt_builder_optimization_level` | int | `3` | TRT builder optimization level (0–5). |
| `trt_auxiliary_streams` | int | `-1` | Number of auxiliary streams (-1 = auto). |
| `trt_tactic_sources` | string | `""` | Tactic sources to enable/disable. |
| `trt_extra_plugin_lib_paths` | string | `""` | Semicolon-separated paths to extra TRT plugin libraries. |
| `trt_profile_min_shapes` | string | `""` | Min shapes for optimization profiles (e.g., `input:1x3x224x224`). |
| `trt_profile_max_shapes` | string | `""` | Max shapes for optimization profiles. |
| `trt_profile_opt_shapes` | string | `""` | Optimal shapes for optimization profiles. |
| `trt_cuda_graph_enable` | bool | `0` | Enable CUDA graph capture and replay. |
| `trt_dump_ep_context_model` | bool | `0` | Dump EPContext model with embedded engine. |
| `trt_ep_context_file_path` | string | `""` | Path for the EPContext model file. |
| `trt_ep_context_embed_mode` | int | `0` | EPContext embedding mode. |
| `trt_weight_stripped_engine_enable` | bool | `0` | Enable weight-stripped engine. |
| `trt_onnx_model_folder_path` | string | `""` | Path to original ONNX model folder (for weight-stripped engine). |
| `trt_engine_hw_compatible` | bool | `0` | Build HW-compatible engine. |
| `trt_op_types_to_exclude` | string | `""` | Op types to exclude from TRT acceleration. |

## Building and Running Tests

Unit tests cover basic inference, dynamic shapes, multi-threading, engine caching, EPContext models, CUDA graph capture/replay, and TRT plugin custom ops.

### Build with Tests

```bash
mkdir build && cd build
cmake -S ../ -B ./ -DCMAKE_BUILD_TYPE=Debug \
  -DTENSORRT_HOME=/path/to/TensorRT \
  -DORT_HOME=/path/to/onnxruntime \
  -DTRT_MAJOR_VERSION=11 \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -Donnxruntime_ep_tensorrt_BUILD_TESTS=ON
cmake --build ./ --config Debug
```

### Run Tests

Set the `TRT_EP_LIBRARY_PATH` environment variable to point to the built plugin EP library, then run via CTest or the test binary directly:

```bash
# Via CTest
cd build
export TRT_EP_LIBRARY_PATH=$(pwd)/libonnxruntime_ep_tensorrt.so   # or onnxruntime_ep_tensorrt.dll on Windows
ctest --output-on-failure

# Or run the test binary directly
./trt_ep_tests
```

### Test Cases

| Test | Description |
|------|-------------|
| `FunctionTest` | Basic inference with a simple Add model. |
| `TestSessionOutputs_MultipleOutputs` | Verifies correct output count for multi-output models. |
| `TestSessionOutputs_UnusedNodeOutput` | Handles models with unused node outputs. |
| `DDSOutputTest` | Inference with data-dependent shapes (DDS). |
| `MultiThreadInference` | Multi-threaded inference on a single session. |
| `MnistModelTest` | End-to-end inference on the MNIST model. |
| `EngineCacheTest` | Engine caching with `trt_engine_cache_enable`. |
| `EPContextNode_ForeignSourceSkipped` | Skips EPContext nodes from other EPs. |
| `EPContextNode_NoSourceAttribute_BackwardCompat` | Backward compatibility with legacy EPContext nodes. |
| `SequentialRuns` | Multiple sequential runs for stability. |
| `DynamicInputShapes` | Dynamic shape support with optimization profiles. |
| `TRTPluginsCustomOpTest` | TRT plugin custom op registration. |
| `BasicCudaGraph` | CUDA graph capture, replay, and in-place input update. |
| `WithoutCudaGraph` | Baseline inference without CUDA graph. |
| `MultipleReplays` | Repeated CUDA graph replays for stability. |

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

