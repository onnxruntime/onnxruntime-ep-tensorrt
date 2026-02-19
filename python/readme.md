# TensorRT Plugin Execution Provider with Python

## Contents
- `onnxruntime_ep_tensorrt`: Contains files for the TensorRT plugin EP Python package. `__init__.py` provides helper functions to get the EP library path and the EP name.
- `setup.py`: Script to generate the Python package wheel.
- `example_usage`: Contains a script showing example usage of the TensorRT plugin EP Python Package.

## Build Instructions

### Build the native plugin EP library

Follow instructions [here](../readme.md#Build-TRT-Plugin-EP) to build the native library.

### Build the Python package

Set the environment variable `TENSORRT_PLUGIN_EP_LIBRARY_PATH` to the path to the native plugin EP shared library. E.g., `tensorrt_plugin_ep.dll` on Windows or `libtensorrt_plugin_ep.so` on Linux.

Run `setup.py` from this directory.

```
python setup.py bdist_wheel
```

The wheel will be generated in the `./dist` directory.

## Run the example usage script

Install the Python package wheel built in the previous step.

```
pip install ./dist/onnxruntime_ep_tensorrt-0.1.0-<version and platform-specific text>.whl
```

Run the Python example usage script.

```
cd example_usage

# install other prerequisites
pip install onnxruntime numpy

python ./example_usage.py
```
