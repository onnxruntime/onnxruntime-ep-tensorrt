# usage:
# mkdir build
# cd build
# cmake                                                             \
#   -DCMAKE_BUILD_TYPE=Debug                                        \
#   -DORT_HOME=/path/to/ort_package/onnxruntime-win-x64-gpu-1.23.0  \
#   -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc                  \
#   -DTENSORRT_HOME=/path/to/tensorrt/TensorRT-11.0.9               \
#   -DTRT_MAJOR_VERSION=11                                          \
#   -S .. -B .

cmake_minimum_required(VERSION 3.25)
cmake_policy(SET CMP0091 NEW) # Use CMAKE_MSVC_RUNTIME_LIBRARY
cmake_policy(SET CMP0141 NEW) # Use CMAKE_MSVC_DEBUG_INFORMATION_FORMAT
project(onnxruntime_ep_tensorrt VERSION 1.0.0 LANGUAGES CXX)

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

if(NOT TRT_MAJOR_VERSION)
  set(TRT_MAJOR_VERSION 10)
endif()
message(STATUS "TRT major version: ${TRT_MAJOR_VERSION}")

# Use the static MSVC runtime consistently across the EP and its dependencies.
if(MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

option(onnxruntime_ep_tensorrt_INSTALL "Install onnxruntime_ep_tensorrt binaries and files" ON)
option(onnxruntime_ep_tensorrt_OBJECT_CACHE "Enable object caching using either sccache/ccache" ON)
set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING "CUDA architectures to enable")

if(onnxruntime_ep_tensorrt_OBJECT_CACHE)
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

# Some Windows CUDA toolkits resolve cudart from lib/x64 even when MSVC
# targets ARM64. Use the compiler architecture; the host processor may be x64.
if(WIN32 AND (CMAKE_CXX_COMPILER_ARCHITECTURE_ID STREQUAL "ARM64" OR
              CMAKE_GENERATOR_PLATFORM STREQUAL "ARM64" OR
              CMAKE_EXE_LINKER_FLAGS MATCHES "[Aa][Rr][Mm]64"))
  get_target_property(_cudart_location CUDA::cudart IMPORTED_LOCATION)
  file(TO_CMAKE_PATH "${_cudart_location}" _cudart_location)
  if(_cudart_location MATCHES "/x64/")
    string(REPLACE "/x64/" "/arm64/" _cudart_arm64 "${_cudart_location}")
    if(NOT EXISTS "${_cudart_arm64}")
      message(FATAL_ERROR "ARM64 CUDA runtime library not found: ${_cudart_arm64}")
    endif()
    set_target_properties(CUDA::cudart PROPERTIES IMPORTED_LOCATION "${_cudart_arm64}")
    message(STATUS "ARM64 CUDA runtime: ${_cudart_arm64}")
  endif()
endif()

file(GLOB onnxruntime_ep_tensorrt_SRC CONFIGURE_DEPENDS
  "./src/*.cc"
  "./src/kernels/*.cc"
  "./src/utils/*.cc"
  "./src/cuda/unary_elementwise_ops_impl.cu"
  "./src/*.h"
  "./src/kernels/*.h"
)
add_library(onnxruntime_ep_tensorrt SHARED ${onnxruntime_ep_tensorrt_SRC})
target_compile_definitions(onnxruntime_ep_tensorrt PUBLIC
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
  # Build protobuf as a static library, matching the project's static CRT.
  set(BUILD_SHARED_LIBS OFF)
endif()

FetchContent_Declare(
  protobuf
  GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
  GIT_TAG        v21.12  # Use a specific tag or commit
)
set(protobuf_BUILD_TESTS OFF CACHE INTERNAL "Disable tests")
set(protobuf_MSVC_STATIC_RUNTIME ON CACHE INTERNAL "Match the project's static MSVC runtime")

# Cross-compilation needs a protoc executable that runs on the build host.
# Use -DPROTOC_EXECUTABLE=/path/to/host/protoc (matching protobuf v21.12).
if(PROTOC_EXECUTABLE)
  get_filename_component(PROTOC_EXECUTABLE "${PROTOC_EXECUTABLE}" ABSOLUTE)
  if(NOT EXISTS "${PROTOC_EXECUTABLE}" OR IS_DIRECTORY "${PROTOC_EXECUTABLE}")
    message(FATAL_ERROR "PROTOC_EXECUTABLE is not a file: ${PROTOC_EXECUTABLE}")
  endif()
  execute_process(COMMAND "${PROTOC_EXECUTABLE}" --version
    RESULT_VARIABLE _protoc_result OUTPUT_VARIABLE _protoc_version
    ERROR_VARIABLE _protoc_error OUTPUT_STRIP_TRAILING_WHITESPACE TIMEOUT 10)
  if(NOT _protoc_result STREQUAL "0")
    message(FATAL_ERROR "Host protoc cannot run: ${PROTOC_EXECUTABLE}: ${_protoc_error}")
  endif()
  set(protobuf_BUILD_PROTOC_BINARIES OFF CACHE BOOL "Use the supplied host protoc" FORCE)
endif()
FetchContent_MakeAvailable(protobuf)

if(PROTOC_EXECUTABLE AND NOT TARGET protobuf::protoc)
  add_executable(protobuf::protoc IMPORTED GLOBAL)
  set_target_properties(protobuf::protoc PROPERTIES IMPORTED_LOCATION "${PROTOC_EXECUTABLE}")
  message(STATUS "Using host protoc: ${PROTOC_EXECUTABLE} (${_protoc_version})")
endif()

# Add ONNX
set(ONNX_USE_MSVC_STATIC_RUNTIME ON CACHE INTERNAL "Match the project's static MSVC runtime")
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
  set(TRT_RUNTIME_DLL_NAME "nvinfer_${TRT_MAJOR_VERSION}.dll")
  set(TRT_PLUGIN_DLL_NAME "nvinfer_plugin_${TRT_MAJOR_VERSION}.dll")
  set(TRT_ONNX_PARSER_DLL_NAME "nvonnxparser_${TRT_MAJOR_VERSION}.dll")
  set(TRT_DLA_COMPILER_DLL_NAME "nvdla_compiler.dll")

  set(TRT_LIBS
    "${TENSORRT_HOME}/lib/nvinfer_${TRT_MAJOR_VERSION}.lib"
    "${TENSORRT_HOME}/lib/nvinfer_plugin_${TRT_MAJOR_VERSION}.lib"
    "${TENSORRT_HOME}/lib/nvonnxparser_${TRT_MAJOR_VERSION}.lib"
  )
  set(ORT_LIBS "${ORT_LIBRARY_DIR}/onnxruntime.lib")

  target_compile_definitions(onnxruntime_ep_tensorrt PRIVATE
    TRT_RUNTIME_DLL_NAME="${TRT_RUNTIME_DLL_NAME}"
    TRT_PLUGIN_DLL_NAME="${TRT_PLUGIN_DLL_NAME}"
    TRT_ONNX_PARSER_DLL_NAME="${TRT_ONNX_PARSER_DLL_NAME}"
    TRT_DLA_COMPILER_DLL_NAME="${TRT_DLA_COMPILER_DLL_NAME}"
  )

  target_link_options(onnxruntime_ep_tensorrt PRIVATE
    "-DEF:${CMAKE_SOURCE_DIR}/src/tensorrt_execution_provider.def"
    "/DELAYLOAD:${TRT_RUNTIME_DLL_NAME}"
    "/DELAYLOAD:${TRT_PLUGIN_DLL_NAME}"
    "/DELAYLOAD:${TRT_ONNX_PARSER_DLL_NAME}"
  )

  # The EP-relative delay loader requires the complete TensorRT runtime set to
  # be co-located with the EP. Include every runtime DLL because TensorRT loads
  # builder/resource libraries dynamically in addition to its PE imports.
  file(TO_CMAKE_PATH "${TENSORRT_HOME}" _tensorrt_home_cmake_path)
  file(GLOB TRT_RUNTIME_DLLS CONFIGURE_DEPENDS
    "${_tensorrt_home_cmake_path}/bin/*.dll"
    "${_tensorrt_home_cmake_path}/lib/*.dll"
    "${_tensorrt_home_cmake_path}/*.dll"
  )
  if(NOT TRT_RUNTIME_DLLS)
    message(FATAL_ERROR
      "No TensorRT runtime DLLs found under ${TENSORRT_HOME}. "
      "Expected DLLs under bin/, lib/, or the TensorRT root.")
  endif()

  # cuDLA for Windows ARM64 is distributed in the DTK EP package rather than
  # the CUDA toolkit or TensorRT SDK. Allow an explicit path and recognize the
  # standard sibling layouts used by DTK drops.
  set(CUDLA_DLL_PATH "" CACHE FILEPATH
      "Path to the target-architecture cudla.dll to stage beside the EP")
  if(CMAKE_CXX_COMPILER_ARCHITECTURE_ID STREQUAL "ARM64" OR
     CMAKE_GENERATOR_PLATFORM STREQUAL "ARM64" OR
     CMAKE_EXE_LINKER_FLAGS MATCHES "[Aa][Rr][Mm]64")
    if(CUDLA_DLL_PATH AND NOT EXISTS "${CUDLA_DLL_PATH}")
      message(FATAL_ERROR "CUDLA_DLL_PATH does not exist: ${CUDLA_DLL_PATH}")
    endif()

    if(NOT CUDLA_DLL_PATH)
      get_filename_component(_tensorrt_parent "${_tensorrt_home_cmake_path}" DIRECTORY)
      get_filename_component(_tensorrt_grandparent "${_tensorrt_parent}" DIRECTORY)
      find_file(_detected_cudla_dll
        NAMES cudla.dll
        HINTS
          "${_tensorrt_home_cmake_path}/NV_TENSORRT_ARM64_EP_PACKAGE"
          "${_tensorrt_parent}/NV_TENSORRT_ARM64_EP_PACKAGE"
          "${_tensorrt_grandparent}/NV_TENSORRT_ARM64_EP_PACKAGE"
        NO_DEFAULT_PATH
      )
      if(_detected_cudla_dll)
        set(CUDLA_DLL_PATH "${_detected_cudla_dll}" CACHE FILEPATH
            "Path to the target-architecture cudla.dll to stage beside the EP" FORCE)
      endif()
    endif()

    if(CUDLA_DLL_PATH)
      get_filename_component(CUDLA_DLL_NAME "${CUDLA_DLL_PATH}" NAME)
      target_compile_definitions(onnxruntime_ep_tensorrt PRIVATE
        CUDLA_DLL_NAME="${CUDLA_DLL_NAME}"
      )
      list(APPEND TRT_RUNTIME_DLLS "${CUDLA_DLL_PATH}")
      list(REMOVE_DUPLICATES TRT_RUNTIME_DLLS)
      message(STATUS "App-local cuDLA runtime: ${CUDLA_DLL_PATH}")
    else()
      message(WARNING
        "cudla.dll was not found for the ARM64 target. Set CUDLA_DLL_PATH to "
        "the cudla.dll from NV_TENSORRT_ARM64_EP_PACKAGE to run DLA without PATH.")
    endif()
  endif()

else()
  set(TRT_LIBS
    "${TENSORRT_HOME}/libnvinfer.so.${TRT_MAJOR_VERSION}"
    "${TENSORRT_HOME}/libnvinfer_plugin.so.${TRT_MAJOR_VERSION}"
    "${TENSORRT_HOME}/libnvonnxparser.so.${TRT_MAJOR_VERSION}"
  )
  set(ORT_LIBS "${ORT_LIBRARY_DIR}/libonnxruntime.so")
endif()

message(STATUS "ORT libs: ${ORT_LIBS}")
message(STATUS "TRT libs: ${TRT_LIBS}")

target_include_directories(onnxruntime_ep_tensorrt PRIVATE
  "$<BUILD_INTERFACE:${ORT_INCLUDE_DIR}>"
  "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>"
  "$<BUILD_INTERFACE:${CUDAToolkit_INCLUDE_DIRS}>"
  "$<BUILD_INTERFACE:${TENSORRT_HOME}/include>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime_ep_tensorrt>"
)

target_link_libraries(onnxruntime_ep_tensorrt PRIVATE
  ${ORT_LIBS}
  ${TRT_LIBS}
  CUDA::cudart
  flatbuffers
  GSL
  onnx
  protobuf::libprotobuf
)

if(WIN32)
  target_link_libraries(onnxruntime_ep_tensorrt PRIVATE delayimp)

  add_custom_command(TARGET onnxruntime_ep_tensorrt POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${TRT_RUNTIME_DLLS}
      "$<TARGET_FILE_DIR:onnxruntime_ep_tensorrt>/"
    COMMAND_EXPAND_LISTS
    COMMENT "Copying TensorRT runtime DLLs next to the plugin EP"
  )
endif()

configure_file(
  "${PROJECT_SOURCE_DIR}/cmake/onnxruntime_ep_tensorrt_config_version.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime_ep_tensorrt_config_version.cmake"
  @ONLY
)
configure_file(
  "${PROJECT_SOURCE_DIR}/cmake/onnxruntime_ep_tensorrt_config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime_ep_tensorrt_config.cmake"
  @ONLY
)

# ── Tests ──────────────────────────────────────────────────────────────────────
option(onnxruntime_ep_tensorrt_BUILD_TESTS "Build unit tests" OFF)

if(onnxruntime_ep_tensorrt_BUILD_TESTS)
  enable_testing()

  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
  )
  set(gtest_force_shared_crt OFF CACHE BOOL "Match the project's static MSVC runtime" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  add_executable(trt_ep_tests
    tests/cuda_graph_test.cc
    tests/tensorrt_basic_test.cc
    tests/tensorrt_dla_test.cc
    tests/tensorrt_ep_context_test.cc
  )
  target_compile_definitions(trt_ep_tests PRIVATE
    -DONNX_NAMESPACE=onnx
    -DONNX_ML
    -DNOMINMAX
  )
  target_include_directories(trt_ep_tests PRIVATE
    "$<BUILD_INTERFACE:${ORT_INCLUDE_DIR}>"
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>"
    "$<BUILD_INTERFACE:${CUDAToolkit_INCLUDE_DIRS}>"
  )
  target_link_libraries(trt_ep_tests PRIVATE
    ${ORT_LIBS}
    CUDA::cudart
    onnx
    protobuf::libprotobuf
    GTest::gtest_main
  )

  # Copy testdata to the build directory so tests can find it
  if(EXISTS "${CMAKE_SOURCE_DIR}/tests/testdata")
    add_custom_command(TARGET trt_ep_tests POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/tests/testdata"
        "$<TARGET_FILE_DIR:trt_ep_tests>/testdata"
      COMMENT "Copying testdata to build output directory"
    )
  endif()

  include(GoogleTest)
  gtest_discover_tests(trt_ep_tests
    PROPERTIES ENVIRONMENT "TESTDATA_DIR=$<TARGET_FILE_DIR:trt_ep_tests>/testdata"
    DISCOVERY_MODE PRE_TEST
  )
endif()

if(onnxruntime_ep_tensorrt_INSTALL)
  # Installation target
  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)

  if(WIN32 AND (CMAKE_CXX_COMPILER_ID MATCHES "MSVC"))
    install(
      FILES "$<TARGET_PDB_FILE:onnxruntime_ep_tensorrt>"
      CONFIGURATIONS Debug RelWithDebInfo
      DESTINATION "${CMAKE_INSTALL_BINDIR}"
    )
  endif()

  install(
    DIRECTORY src/
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime_ep_tensorrt"
    FILES_MATCHING
    PATTERN "*.h"
  )

  install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime_ep_tensorrt_config.cmake"
          "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime_ep_tensorrt_config_version.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/onnxruntime_ep_tensorrt"
  )

  if(WIN32)
    install(FILES ${TRT_RUNTIME_DLLS} DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()

  install(TARGETS onnxruntime_ep_tensorrt GSL flatbuffers EXPORT onnxruntime_ep_tensorrt_targets)
  install(EXPORT onnxruntime_ep_tensorrt_targets
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/onnxruntime_ep_tensorrt"
    NAMESPACE onnxruntime_ep_tensorrt::
  )
endif()
