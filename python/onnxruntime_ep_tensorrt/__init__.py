from __future__ import annotations

import pathlib

__all__ = ["get_ep_name", "get_ep_names", "get_library_path"]

module_dir = pathlib.Path(__file__).parent


def get_library_path() -> str:
    candidate_paths = [
        module_dir / "ORTTensorRTEp.dll",
        module_dir / "ORTTensorRTEp.so",
    ]

    paths = [p for p in candidate_paths if p.is_file()]

    assert len(paths) == 1, "Did not find exactly one library path."

    return str(paths[0])


def get_ep_name() -> str:
    return "TensorRTPluginExecutionProvider"


def get_ep_names() -> list[str]:
    return [get_ep_name()]
