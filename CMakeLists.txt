# usage:
# mkdir build
# cd build
# cmake                                                             \
#   -DCMAKE_BUILD_TYPE=Debug                                        \
#   -DORT_HOME=/path/to/ort_package/onnxruntime-win-x64-gpu-1.23.0  \
#   -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc                  \
#   -DTENSORRT_HOME=/path/to/tensorrt/TensorRT-10.3.0.26            \
#   -S .. -B .

cmake_minimum_required(VERSION 3.24)
project(TensorRTEp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Use release mode if not specified
if(NOT CMAKE_BUILD_TYPE)
  message(WARNING "Build type not set, falling back to RelWithDebInfo.")
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose build configuration. [Debug | Release]" FORCE)
endif(NOT CMAKE_BUILD_TYPE)

if(NOT CMAKE_CUDA_COMPILER)
  message(FATAL_ERROR "Please specify CMAKE_CUDA_COMPILER, e.g. -DCMAKE_CUDA_COMPILER=/path/to/nvcc")
endif(NOT CMAKE_CUDA_COMPILER)

if(NOT TENSORRT_HOME)
  message(FATAL_ERROR "Please specify TENSORRT_HOME, e.g. -DTENSORRT_HOME=/path/to/trt/")
endif(NOT TENSORRT_HOME)

# CMake config to force dynamic debug CRT or dynamic release CRT globally for all dependencies.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

option(TensorRTEp_INSTALL "Install TensorRTEp binaries and files" ON)
option(CMAKE_CUDA_ARCHITECTURES "Version of CUDA architectures to enable" 80)

include(FetchContent)
include(${CMAKE_SOURCE_DIR}/cmake/onnxruntime_library_utils.cmake)

# Setup CUDA dependencies
enable_language(CUDA) # via nvcc to get the CUDA tool kit
file(TO_CMAKE_PATH "/usr/local/cuda" CUDAToolkit_ROOT)
find_package(CUDAToolkit REQUIRED)

file(GLOB TensortRTEp_SRC 
  "./src/*.cc"
  "./src/kernels/*.cc"
  "./src/utils/*.cc"
  "./src/cuda/unary_elementwise_ops_impl.cu"
  "./src/*.h"
  "./src/kernels/*.h"
)
add_library(TensorRTEp SHARED ${TensortRTEp_SRC})
target_compile_definitions(TensorRTEp
  PUBLIC -DONNX_NAMESPACE=onnx
  PUBLIC -DONNX_ML
  PUBLIC -DNOMINMAX
)

set_onnxruntime_paths(
  ORT_HOME ${ORT_HOME}
  DEFAULT_ORT_VERSION "1.24.1"
  ORT_INCLUDE_DIR_VAR ORT_INCLUDE_DIR
  ORT_LIBRARY_DIR_VAR ORT_LIBRARY_DIR
)

message(STATUS "ORT_LIBRARY_DIR: ${ORT_LIBRARY_DIR}")
message(STATUS "ORT_INCLUDE_DIR: ${ORT_INCLUDE_DIR}")

# Add protobuf
if(WIN32)
  # Build protobuf as static library, but using dynamic runtime
  set(BUILD_SHARED_LIBS OFF)
endif()

FetchContent_Declare(
  protobuf
  GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
  GIT_TAG        v21.12  # Use a specific tag or commit
)
set(protobuf_BUILD_TESTS OFF CACHE INTERNAL "Disable tests")
set(protobuf_MSVC_STATIC_RUNTIME OFF CACHE INTERNAL "Force build static runtime")
FetchContent_MakeAvailable(protobuf)

# Add ONNX
FetchContent_Declare(
  onnx
  GIT_REPOSITORY https://github.com/onnx/onnx.git
  GIT_TAG        v1.18.0  # Use a specific tag or commit
)
FetchContent_MakeAvailable(onnx)

# Add GSL
set(GSL_INSTALL OFF CACHE INTERNAL "Disable install")
FetchContent_Declare(
  gsl
  GIT_REPOSITORY https://github.com/microsoft/GSL.git
  GIT_TAG        v4.0.0  # Use a specific tag or commit
)
FetchContent_MakeAvailable(gsl)

# Add flatbuffers
set(FLATBUFFERS_INSTALL OFF CACHE INTERNAL "Disable install")
set(FLATBUFFERS_BUILD_TESTS OFF CACHE INTERNAL "Disable tests")
FetchContent_Declare(
  flatbuffers
  GIT_REPOSITORY https://github.com/google/flatbuffers.git
  GIT_TAG        v23.5.26  # Use a specific tag or commit
)
FetchContent_MakeAvailable(flatbuffers)

set(TRT_LIBS
  ${TENSORRT_HOME}/lib/${CMAKE_IMPORT_LIBRARY_PREFIX}nvinfer$<$<BOOL:WIN32>:_10>${CMAKE_IMPORT_LIBRARY_SUFFIX}
  ${TENSORRT_HOME}/lib/${CMAKE_IMPORT_LIBRARY_PREFIX}nvinfer_plugin$<$<BOOL:WIN32>:_10>${CMAKE_IMPORT_LIBRARY_SUFFIX}
  ${TENSORRT_HOME}/lib/${CMAKE_IMPORT_LIBRARY_PREFIX}nvonnxparser$<$<BOOL:WIN32>:_10>${CMAKE_IMPORT_LIBRARY_SUFFIX}
)

if(WIN32) # Windows
  set(ORT_LIB "${ORT_LIBRARY_DIR}/onnxruntime.lib")
  set(TensortRTEp_LINK_FLAG "-DEF:${CMAKE_SOURCE_DIR}/src/tensorrt_execution_provider.def")
else() # Linux
  set(ORT_LIB "${ORT_LIBRARY_DIR}/lib/libonnxruntime.so")
endif()

MESSAGE(STATUS "Looking for following dependencies ...")
MESSAGE(STATUS "ORT lib  : ${ORT_LIB}")
MESSAGE(STATUS "TRT libs : ${TRT_LIBS}")

set_property(
  TARGET TensorRTEp APPEND_STRING PROPERTY LINK_FLAGS ${TensortRTEp_LINK_FLAG}
)

target_include_directories(TensorRTEp PRIVATE
  $<BUILD_INTERFACE:${ORT_INCLUDE_DIR}>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CUDAToolkit_INCLUDE_DIRS}>
  $<BUILD_INTERFACE:${TENSORRT_HOME}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/tensorrtep>
)

target_link_libraries(TensorRTEp PRIVATE
  protobuf::libprotobuf
  onnx
  GSL
  flatbuffers
  ${ORT_LIB}
  ${TRT_LIBS}
  CUDA::cudart
)

configure_file(
  ${PROJECT_SOURCE_DIR}/cmake/TensorRTEpConfigVersion.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TensorRTEpConfigVersion.cmake
  @ONLY
)
configure_file(
  ${PROJECT_SOURCE_DIR}/cmake/TensorRTEpConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TensorRTEpConfig.cmake
  @ONLY
)

if(TensorRTEp_INSTALL)
  # Installation target
  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)

  if(WIN32 AND (CMAKE_CXX_COMPILER_ID MATCHES "MSVC"))
    install(
      FILES "$<TARGET_PDB_FILE:TensorRTEp>"
      CONFIGURATIONS Debug RelWithDebInfo
      DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
  endif()

  install(
    DIRECTORY src/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tensorrtep
    FILES_MATCHING
    PATTERN "*.h"
  )

  install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/TensorRTEpConfig.cmake
          ${CMAKE_CURRENT_BINARY_DIR}/TensorRTEpConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tensorrtep
  )

  install(TARGETS TensorRTEp GSL flatbuffers EXPORT TensorRTEpTargets)
  install(EXPORT TensorRTEpTargets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tensorrtep
    NAMESPACE TensorRTEp::
  )
endif(TensorRTEp_INSTALL)
