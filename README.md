# TensorRT Plugin Execution Provider
This TensorRT plugin EP is originally migrated from the provider-bridge [TensorRT EP](https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/core/providers/tensorrt) and implements the required ORT EP interfaces (including `OrtEpFactory`, `OrtEp`, `OrtNodeComputeInfo`, `OrtDataTransferImpl`, etc.) to interact with ONNX Runtime through the EP ABI introduced in ORT 1.23.0.

TensorRT plugin EP should be built as a shared library and does not need to be built together with ONNX Runtime. It only needs to link against the ONNX Runtime shared library, i.e., `onnxruntime.dll` or `libonnxruntime.so`.

This TensorRT plugin EP can be built on Linux and Windows with "Debug" and "Release" mode.

## Contents
- `CMakeLists.txt`: Build configuration for the TensorRT plugin EP.
- `src`: Contains source code for the TensorRT plugin EP.
- `python`: Contains example code for setting up and using a Python package.

## Build Instructions
### On Windows
```bash
mkdir build;cd build
```
```bash
cmake -S ../ -B ./ -DCMAKE_BUILD_TYPE=Debug -DTENSORRT_HOME=C:/folder/to/trt -DORT_HOME=C:/folder/to/ort
```
```bash
cmake --build ./ --config Debug
```
(Note: The `ORT_HOME` should contain the include and lib folders as below)
```
C:/folder/to/ort
      | ----- lib
      |          | ----- onnxruntime.dll
      |          | ----- onnxruntime.lib
      |          | ----- onnxruntime.pdb
      |          ...
      |
      | ---- include
      |          | ----- onnxruntime_c_api.h
      |          | ----- onnxruntime_ep_c_api.h
      |          | ----- onnxruntime_cxx_api.h
      |          | ----- onnxruntime_cxx_inline_api.h
      |          ...
```

 
### On Linux
```bash
mkdir build;cd build
```
```bash
cmake -S ../ -B ./ -DCMAKE_BUILD_TYPE=Debug -DTENSORRT_HOME=/home/to/trt/ -DORT_HOME=/home/to/ort -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_POSITION_INDEPENDENT_CODE=ON
```
```bash
cmake --build ./ --config Debug
````

## Usage
Please use `onnxruntime_perf_test`
```bash
--plugin_ep_libs "TensorRTEp|C:\repos\onnxruntime-ep-tensorrt\build\Debug\TensorRTEp.dll" --plugin_eps TensorRTEp -r 1 C:\path\to\model
```

