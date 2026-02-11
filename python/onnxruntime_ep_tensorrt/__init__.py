from __future__ import annotations
import pathlib
import sys
import os
import ctypes
import onnxruntime as ort

if sys.platform == "win32":
    ort_dir = os.path.dirname(os.path.abspath(ort.__file__))
    dll_path = os.path.join(ort_dir, "capi", "onnxruntime.dll")

    # When the application calls ort.register_execution_provider_library() with the path to the plugin EP DLL,
    # ORT internally uses LoadLibraryExW() to load that DLL. Since the plugin EP depends on onnxruntime.dll,
    # the operating system will attempt to locate and load onnxruntime.dll first.
    #
    # On Windows, LoadLibraryExW() searches the directory containing the plugin EP DLL before searching system directories.
    # Because onnxruntime.dll is not located in the plugin EP’s directory, Windows ends up loading the copy from a 
    # system directory instead — which is not the correct version.
    #
    # To ensure the plugin EP uses the correct onnxruntime.dll bundled with the ONNX Runtime package, 
    # we load that DLL explicitly before loading the plugin EP DLL.
    ctypes.WinDLL(dll_path)

__all__ = ['get_library_path', 'get_ep_name', 'get_ep_names']

module_dir = pathlib.Path(__file__).parent

def get_library_path() -> str:
    candidate_paths = [
        module_dir / 'tensorrt_plugin_ep.dll',
        module_dir / 'libtensorrt_plugin_ep.so',
    ]

    paths = [p for p in candidate_paths if p.is_file()]

    assert len(paths) == 1, f"Did not find exactly one library path."

    return str(paths[0])

def get_ep_name() -> str:
    return "TensorRTPluginExecutionProvider"

def get_ep_names() -> list[str]:
    return [get_ep_name()]