# usage:
# cd build/
# cmake -S ../ -B ./                                                \
#   -DCMAKE_BUILD_TYPE=Debug                                        \
#   -DORT_HOME=/path/to/ort_package/onnxruntime-win-x64-gpu-1.23.0  \
#   -DCMAKE_CUDA_ARCHITECTURES=80                                   \
#   -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc                  \
#   -DTENSORRT_HOME=/path/to/tensorrt/TensorRT-10.3.0.26
#
# (see the result of "nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits")
# cmake --build ./ --config Debug

cmake_minimum_required(VERSION 3.16)
project(TensorRTEp VERSION 1.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Use release mode if not specified
if (NOT CMAKE_BUILD_TYPE)
  message(WARNING "Build type not set, falling back to Release.")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose build configuration. [Debug | Release]", FORCE)
endif(NOT CMAKE_BUILD_TYPE)

if (NOT CMAKE_CUDA_COMPILER)
  message(FATAL_ERROR "Please specify CMAKE_CUDA_COMPILER, e.g. -DCMAKE_CUDA_COMPILER=/path/to/nvcc")
endif(NOT TENSORRT_HOME)

if (NOT TENSORRT_HOME)
  message(FATAL_ERROR "Please specify TENSORRT_HOME, e.g. -DTENSORRT_HOME=/path/to/trt/")
endif(NOT TENSORRT_HOME)

# CMake config to force dynamic debug CRT or dynamic release CRT globally for all dependencies.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

include(FetchContent)
include(${CMAKE_SOURCE_DIR}/cmake/onnxruntime_library_utils.cmake)

# Setup CUDA dependencies
enable_language(CUDA) # via nvcc to get the CUDA tool kit
file(TO_CMAKE_PATH "/usr/local/cuda" CUDAToolkit_ROOT)
find_package(CUDAToolkit REQUIRED)

if(WIN32)
  # Build protobuf as static library, but using dynamic runtime
  set(BUILD_SHARED_LIBS OFF)
endif()

add_definitions(-DONNX_NAMESPACE=onnx)
add_definitions(-DONNX_ML)
add_definitions(-DNOMINMAX)
file(GLOB TENSORRTEP_SRC 
  "./src/*.cc"
  "./src/kernels/*.cc"
  "./src/utils/*.cc"
  "./src/cuda/unary_elementwise_ops_impl.cu"
  "./src/*.h"
  "./src/kernels/*.h"
)
add_library(TensorRTEp SHARED ${TENSORRTEP_SRC})

set_onnxruntime_paths(
  ORT_HOME ${ORT_HOME}
  DEFAULT_ORT_VERSION "1.24.3"
  ORT_INCLUDE_DIR_VAR ORT_INCLUDE_DIR
  ORT_LIBRARY_DIR_VAR ORT_LIBRARY_DIR
)

message(STATUS "ORT_LIBRARY_DIR: ${ORT_LIBRARY_DIR}")
message(STATUS "ORT_INCLUDE_DIR: ${ORT_INCLUDE_DIR}")

# Add protobuf
FetchContent_Declare(
  protobuf
  GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
  GIT_TAG        v21.12  # Use a specific tag or commit
)

if (WIN32)
  # Sometimes, protobuf ignores CMAKE_MSVC_RUNTIME_LIBRARY. To ensure it works:
  set(protobuf_MSVC_STATIC_RUNTIME OFF CACHE BOOL "" FORCE)
endif()

FetchContent_MakeAvailable(protobuf)

# Add ONNX
FetchContent_Declare(
  onnx
  GIT_REPOSITORY https://github.com/onnx/onnx.git
  GIT_TAG        v1.18.0  # Use a specific tag or commit
)
FetchContent_MakeAvailable(onnx)

# Add GSL
FetchContent_Declare(
  gsl
  GIT_REPOSITORY https://github.com/microsoft/GSL.git
  GIT_TAG        v4.0.0  # Use a specific tag or commit
)
FetchContent_MakeAvailable(gsl)

# Add flatbuffers
FetchContent_Declare(
  flatbuffers
  GIT_REPOSITORY https://github.com/google/flatbuffers.git
  GIT_TAG        v23.5.26  # Use a specific tag or commit
)
FetchContent_MakeAvailable(flatbuffers)

set(DEPS_PATH "${CMAKE_BINARY_DIR}/_deps")

if (WIN32) # Windows
  set(ORT_LIB "${ORT_HOME}/lib/onnxruntime.lib")

  set(TRT_LIBS
    "${TENSORRT_HOME}/lib/nvinfer_10.lib" 
    "${TENSORRT_HOME}/lib/nvinfer_plugin_10.lib"
    "${TENSORRT_HOME}/lib/nvonnxparser_10.lib"
  )
  
  set(DEPS_LIBS
    ${DEPS_LIBS}
    "${DEPS_PATH}/protobuf-build/${CMAKE_BUILD_TYPE}/${CMAKE_STATIC_LIBRARY_PREFIX}protobuf${CMAKE_STATIC_LIBRARY_SUFFIX}.lib"
    "${DEPS_PATH}/protobuf-build/${CMAKE_BUILD_TYPE}/${CMAKE_STATIC_LIBRARY_PREFIX}protoc${CMAKE_STATIC_LIBRARY_SUFFIX}.lib"
    "${DEPS_PATH}/flatbuffers-build/${CMAKE_BUILD_TYPE}/flatbuffers.lib"
    "${DEPS_PATH}/onnx-build/${CMAKE_BUILD_TYPE}/onnx.lib"
    "${DEPS_PATH}/onnx-build/${CMAKE_BUILD_TYPE}/onnx_proto.lib"
  )

  set(TENSORRTEP_LINK_FLAG "-DEF:${CMAKE_SOURCE_DIR}/src/tensorrt_execution_provider.def")

else() # Linux
  set(ORT_LIB "${ORT_HOME}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}onnxruntime.so")

  set(TRT_LIBS
    "${TENSORRT_HOME}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}nvinfer.so"
    "${TENSORRT_HOME}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}nvinfer_plugin.so"
    "${TENSORRT_HOME}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}nvonnxparser.so"
  )

  set(DEPS_LIBS
    ${DEPS_LIBS}
    "${DEPS_PATH}/flatbuffers-build/${CMAKE_STATIC_LIBRARY_PREFIX}flatbuffers.a"
    "${DEPS_PATH}/onnx-build/${CMAKE_STATIC_LIBRARY_PREFIX}onnx.a"
    "${DEPS_PATH}/onnx-build/${CMAKE_STATIC_LIBRARY_PREFIX}onnx_proto.a"
    "${DEPS_PATH}/protobuf-build/${CMAKE_STATIC_LIBRARY_PREFIX}protobuf${CMAKE_STATIC_LIBRARY_SUFFIX}.a"
    "${DEPS_PATH}/protobuf-build/${CMAKE_STATIC_LIBRARY_PREFIX}protoc${CMAKE_STATIC_LIBRARY_SUFFIX}.a"
  )
endif()

MESSAGE(STATUS "Looking for following dependencies ...")
MESSAGE(STATUS "ORT lib  : ${ORT_LIB}")
MESSAGE(STATUS "TRT libs : ${TRT_LIBS}")
MESSAGE(STATUS "Deps libs: ${DEPS_LIBS}")

set_property(
  TARGET TensorRTEp APPEND_STRING PROPERTY LINK_FLAGS TENSORRTEP_LINK_FLAG}
)

target_include_directories(TensorRTEp PUBLIC
  "${ORT_HOME}/include"
  "./src/utils"
  "/usr/local/cuda/include"
  "${TENSORRT_HOME}/include"
  "${DEPS_PATH}/flatbuffers-src/include"
  "${DEPS_PATH}/gsl-src/include" # GSL is header-only
  "${DEPS_PATH}/onnx-src"
  "${DEPS_PATH}/onnx-build"
  "${DEPS_PATH}/protobuf-src/src"
)

target_link_libraries(TensorRTEp PUBLIC
  #${DEPS_LIBS}
  protobuf::libprotobuf
  onnx
  flatbuffers
  ${ORT_LIB}
  ${TRT_LIBS}
  CUDA::cudart
)
