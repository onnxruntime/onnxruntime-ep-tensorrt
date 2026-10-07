// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "windows_dependency_loader.h"

#ifdef _WIN32

#include <windows.h>
#include <delayimp.h>

#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "utils/path_string.h"

#ifndef TRT_RUNTIME_DLL_NAME
#define TRT_RUNTIME_DLL_NAME "nvinfer.dll"
#endif

#ifndef TRT_PLUGIN_DLL_NAME
#define TRT_PLUGIN_DLL_NAME "nvinfer_plugin.dll"
#endif

#ifndef TRT_ONNX_PARSER_DLL_NAME
#define TRT_ONNX_PARSER_DLL_NAME "nvonnxparser.dll"
#endif

#ifndef TRT_DLA_COMPILER_DLL_NAME
#define TRT_DLA_COMPILER_DLL_NAME "nvdla_compiler.dll"
#endif

#ifndef CUDLA_DLL_NAME
#define CUDLA_DLL_NAME "cudla.dll"
#endif

namespace {

constexpr const char* kRequiredTensorRtDlls[]{
    TRT_RUNTIME_DLL_NAME,
    TRT_PLUGIN_DLL_NAME,
    TRT_ONNX_PARSER_DLL_NAME,
};

std::once_flag g_load_dependencies_once;
std::mutex g_loaded_modules_mutex;
std::vector<std::pair<std::string, HMODULE>> g_loaded_modules;

bool EqualsIgnoreCase(std::string_view lhs, std::string_view rhs) {
  return lhs.size() == rhs.size() &&
         _strnicmp(lhs.data(), rhs.data(), lhs.size()) == 0;
}

bool IsManagedDependency(const char* dll_name) {
  if (dll_name == nullptr) {
    return false;
  }

  std::string_view filename{dll_name};
  const size_t separator = filename.find_last_of("\\/");
  if (separator != std::string_view::npos) {
    filename.remove_prefix(separator + 1);
  }

  for (const char* required_dll : kRequiredTensorRtDlls) {
    if (EqualsIgnoreCase(filename, required_dll)) {
      return true;
    }
  }

  if (EqualsIgnoreCase(filename, TRT_DLA_COMPILER_DLL_NAME)) {
    return true;
  }

  if (EqualsIgnoreCase(filename, CUDLA_DLL_NAME)) {
    return true;
  }

  return false;
}

std::wstring GetEpDirectory() {
  HMODULE ep_module = nullptr;
  const auto address_in_ep = reinterpret_cast<LPCWSTR>(&GetEpDirectory);
  if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                              GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                          address_in_ep, &ep_module)) {
    throw std::runtime_error("TensorRT EP could not identify its own module");
  }

  std::vector<wchar_t> path(512);
  for (;;) {
    const DWORD length = GetModuleFileNameW(ep_module, path.data(), static_cast<DWORD>(path.size()));
    if (length == 0) {
      throw std::runtime_error("TensorRT EP could not determine its module path");
    }

    if (length < path.size() - 1) {
      path.resize(length);
      break;
    }

    if (path.size() >= 32768) {
      throw std::runtime_error("TensorRT EP module path exceeds the Windows maximum path length");
    }
    path.resize(path.size() * 2);
  }

  std::wstring module_path(path.begin(), path.end());
  const size_t separator = module_path.find_last_of(L"\\/");
  if (separator == std::wstring::npos) {
    throw std::runtime_error("TensorRT EP module path does not contain a directory");
  }

  return module_path.substr(0, separator);
}

std::string WindowsErrorMessage(DWORD error) {
  wchar_t* message = nullptr;
  const DWORD length = FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER |
                                          FORMAT_MESSAGE_FROM_SYSTEM |
                                          FORMAT_MESSAGE_IGNORE_INSERTS,
                                      nullptr, error, 0,
                                      reinterpret_cast<wchar_t*>(&message), 0, nullptr);
  if (length == 0 || message == nullptr) {
    return "Windows error " + std::to_string(error);
  }

  std::wstring text(message, length);
  LocalFree(message);
  while (!text.empty() && (text.back() == L'\r' || text.back() == L'\n' || text.back() == L' ')) {
    text.pop_back();
  }

  return ToUTF8String(text) + " (Windows error " + std::to_string(error) + ")";
}

HMODULE FindLoadedModule(std::string_view dll_name) {
  std::lock_guard<std::mutex> lock(g_loaded_modules_mutex);
  for (const auto& [name, module] : g_loaded_modules) {
    if (EqualsIgnoreCase(name, dll_name)) {
      return module;
    }
  }
  return nullptr;
}

HMODULE LoadManagedDependency(const char* dll_name) {
  std::string_view filename{dll_name};
  const size_t separator = filename.find_last_of("\\/");
  if (separator != std::string_view::npos) {
    filename.remove_prefix(separator + 1);
  }

  if (HMODULE existing = FindLoadedModule(filename); existing != nullptr) {
    return existing;
  }

  const std::wstring full_path = GetEpDirectory() + L"\\" + ToWideString(filename);
  const DWORD attributes = GetFileAttributesW(full_path.c_str());
  const bool app_local = attributes != INVALID_FILE_ATTRIBUTES &&
                         (attributes & FILE_ATTRIBUTE_DIRECTORY) == 0;
  const std::wstring load_path = app_local ? full_path : ToWideString(filename);
  // Prefer an explicitly staged SDK. If absent, use the normal Windows DLL
  // search policy, including PATH, as applications using TensorRT expect.
  HMODULE module = LoadLibraryExW(load_path.c_str(), nullptr,
      app_local ? LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32 : 0);
  if (module == nullptr) {
    const DWORD error = GetLastError();
    throw std::runtime_error("TensorRT EP failed to load dependency '" +
                             ToUTF8String(load_path) + "': " + WindowsErrorMessage(error));
  }

  {
    std::lock_guard<std::mutex> lock(g_loaded_modules_mutex);
    g_loaded_modules.emplace_back(filename, module);
  }
  return module;
}

FARPROC WINAPI DelayLoadNotifyHook(unsigned notification, PDelayLoadInfo delay_info) {
  if (notification != dliNotePreLoadLibrary || delay_info == nullptr ||
      !IsManagedDependency(delay_info->szDll)) {
    return nullptr;
  }

  try {
    return reinterpret_cast<FARPROC>(LoadManagedDependency(delay_info->szDll));
  } catch (...) {
    // Preserve the standard delay-load failure after both supported search
    // locations have been considered.
    RaiseException(VcppException(ERROR_SEVERITY_ERROR, ERROR_MOD_NOT_FOUND), 0, 0, nullptr);
    return nullptr;
  }
}

}  // namespace

extern "C" {
const PfnDliHook __pfnDliNotifyHook2 = DelayLoadNotifyHook;
}

namespace trt_ep {

void EnsureTensorRtDependenciesLoaded() {
  std::call_once(g_load_dependencies_once, []() {
    for (const char* dll_name : kRequiredTensorRtDlls) {
      LoadManagedDependency(dll_name);
    }
  });
}

void EnsureDlaCompilerDependencyLoaded() {
  LoadManagedDependency(TRT_DLA_COMPILER_DLL_NAME);
}

void EnsureCuDlaDependencyLoaded() {
  LoadManagedDependency(CUDLA_DLL_NAME);
}

}  // namespace trt_ep

#else

namespace trt_ep {

void EnsureTensorRtDependenciesLoaded() {}

void EnsureDlaCompilerDependencyLoaded() {}

void EnsureCuDlaDependencyLoaded() {}

}  // namespace trt_ep

#endif
