# usage:
# mkdir build
# cd build
# cmake                                                             \
#   -DCMAKE_BUILD_TYPE=Debug                                        \
#   -DORT_HOME=/path/to/ort_package/onnxruntime-win-x64-gpu-1.23.0  \
#   -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc                  \
#   -DTENSORRT_HOME=/path/to/tensorrt/TensorRT-10.3.0.26            \
#   -S .. -B .

cmake_minimum_required(VERSION 3.25)
cmake_policy(SET CMP0091 NEW) # Use CMAKE_MSVC_RUNTIME_LIBRARY
cmake_policy(SET CMP0141 NEW) # Use CMAKE_MSVC_DEBUG_INFORMATION_FORMAT
project(ORTTensorRTEp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Use release mode if not specified
if(NOT CMAKE_BUILD_TYPE)
  message(WARNING "Build type not set, falling back to RelWithDebInfo.")
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose build configuration. [Debug | Release]" FORCE)
endif()

if(NOT CMAKE_CUDA_COMPILER)
  message(FATAL_ERROR "Please specify CMAKE_CUDA_COMPILER, e.g. -DCMAKE_CUDA_COMPILER=/path/to/nvcc")
endif()

if(NOT TENSORRT_HOME)
  message(FATAL_ERROR "Please specify TENSORRT_HOME, e.g. -DTENSORRT_HOME=/path/to/trt/")
endif()

# CMake config to force dynamic debug CRT or dynamic release CRT globally for all dependencies.
if(MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

option(ORTTensorRTEp_INSTALL "Install ORTTensorRTEp binaries and files" ON)
option(ORTTensorRTEp_OBJECT_CACHE "Enable object caching using either sccache/ccache" ON)
option(CMAKE_CUDA_ARCHITECTURES "Version of CUDA architectures to enable" 86)

if(ORTTensorRTEp_OBJECT_CACHE)
  find_program(SCCACHE_PROGRAM sccache)
  find_program(CCACHE_PROGRAM ccache)
  # Prioritize sccache on Windows & ccache on non-Windows
  if(WIN32)
    if(SCCACHE_PROGRAM)
      set(CMAKE_C_COMPILER_LAUNCHER "${SCCACHE_PROGRAM}")
      set(CMAKE_CXX_COMPILER_LAUNCHER "${SCCACHE_PROGRAM}")
    elseif(CCACHE_PROGRAM)
      set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
      set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    endif()
  else()
    if(CCACHE_PROGRAM)
      set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
      set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    elseif(SCCACHE_PROGRAM)
      set(CMAKE_C_COMPILER_LAUNCHER "${SCCACHE_PROGRAM}")
      set(CMAKE_CXX_COMPILER_LAUNCHER "${SCCACHE_PROGRAM}")
    endif()
  endif()
endif()

include(FetchContent)
include("${CMAKE_SOURCE_DIR}/cmake/onnxruntime_library_utils.cmake")

if(MSVC)
  # Few flags required for sccache to succeed. More information here:
  # https://github.com/mozilla/sccache/
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "Embedded")

  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    string(REPLACE "/Zi" "/Z7" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
    string(REPLACE "/Zi" "/Z7" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
  elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    string(REPLACE "/Zi" "/Z7" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    string(REPLACE "/Zi" "/Z7" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
  elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    string(REPLACE "/Zi" "/Z7" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
    string(REPLACE "/Zi" "/Z7" CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO}")
  endif()
endif()

# Setup CUDA dependencies
enable_language(CUDA) # via nvcc to get the CUDA tool kit
find_package(CUDAToolkit REQUIRED)

file(GLOB ORTTensorRTEp_SRC CONFIGURE_DEPENDS
  "./src/*.cc"
  "./src/kernels/*.cc"
  "./src/utils/*.cc"
  "./src/cuda/unary_elementwise_ops_impl.cu"
  "./src/*.h"
  "./src/kernels/*.h"
)
add_library(ORTTensorRTEp SHARED ${ORTTensorRTEp_SRC})
target_compile_definitions(ORTTensorRTEp PUBLIC
  -DONNX_NAMESPACE=onnx
  -DONNX_ML
  -DNOMINMAX
)

set_onnxruntime_paths(
  ORT_HOME "${ORT_HOME}"
  DEFAULT_ORT_VERSION "1.25.0-rc.1"
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
  GIT_TAG        e709452ef2bbc1d113faf678c24e6d3467696e83 # v1.18.0 Use a specific tag or commit
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

if(WIN32) # Windows
  set(TRT_LIBS
    "${TENSORRT_HOME}/lib/nvinfer_10.lib"
    "${TENSORRT_HOME}/lib/nvinfer_plugin_10.lib"
    "${TENSORRT_HOME}/lib/nvonnxparser_10.lib"
  )
  set(ORT_LIBS "${ORT_LIBRARY_DIR}/onnxruntime.lib")

  target_link_options(ORTTensorRTEp PRIVATE "-DEF:${CMAKE_SOURCE_DIR}/src/tensorrt_execution_provider.def")
else()
  set(TRT_LIBS
    "${TENSORRT_HOME}/libnvinfer.so.10"
    "${TENSORRT_HOME}/libnvinfer_plugin.so.10"
    "${TENSORRT_HOME}/libnvonnxparser.so.10"
  )
  set(ORT_LIBS "${ORT_LIBRARY_DIR}/libonnxruntime.so")
endif()

message(STATUS "ORT libs: ${ORT_LIBS}")
message(STATUS "TRT libs: ${TRT_LIBS}")

target_include_directories(ORTTensorRTEp PRIVATE
  "$<BUILD_INTERFACE:${ORT_INCLUDE_DIR}>"
  "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>"
  "$<BUILD_INTERFACE:${CUDAToolkit_INCLUDE_DIRS}>"
  "$<BUILD_INTERFACE:${TENSORRT_HOME}/include>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/orttrtep>"
)

target_link_libraries(ORTTensorRTEp PRIVATE
  ${ORT_LIBS}
  ${TRT_LIBS}
  CUDA::cudart
  flatbuffers
  GSL
  onnx
  protobuf::libprotobuf
)

configure_file(
  "${PROJECT_SOURCE_DIR}/cmake/ORTTensorRTEpConfigVersion.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/ORTTensorRTEpConfigVersion.cmake"
  @ONLY
)
configure_file(
  "${PROJECT_SOURCE_DIR}/cmake/ORTTensorRTEpConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/ORTTensorRTEpConfig.cmake"
  @ONLY
)

if(ORTTensorRTEp_INSTALL)
  # Installation target
  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)

  if(WIN32 AND (CMAKE_CXX_COMPILER_ID MATCHES "MSVC"))
    install(
      FILES "$<TARGET_PDB_FILE:ORTTensorRTEp>"
      CONFIGURATIONS Debug RelWithDebInfo
      DESTINATION "${CMAKE_INSTALL_BINDIR}"
    )
  endif()

  install(
    DIRECTORY src/
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/orttrtep"
    FILES_MATCHING
    PATTERN "*.h"
  )

  install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/ORTTensorRTEpConfig.cmake"
          "${CMAKE_CURRENT_BINARY_DIR}/ORTTensorRTEpConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/orttrtep"
  )

  install(TARGETS ORTTensorRTEp GSL flatbuffers EXPORT ORTTensorRTEpTargets)
  install(EXPORT ORTTensorRTEpTargets
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/orttrtep"
    NAMESPACE ORTTensorRTEp::
  )
endif()
