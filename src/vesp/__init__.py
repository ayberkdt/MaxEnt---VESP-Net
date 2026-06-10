"""Experimental discrete VESP gravity surrogate framework.

The convenience re-exports below resolve lazily (PEP 562): importing ``vesp`` -- or any
torch-free subpackage such as ``vesp.ui`` -- does not pull ``torch`` until one of these names
is actually used. All internal code imports from the concrete submodules directly.
"""

from __future__ import annotations

__all__ = [
    "DiscreteVESP",
    "MultiShellDiscreteVESP",
    "SourceGeometry",
    "SourceSet",
    "fibonacci_sphere",
    "make_shell_sources",
]

_LAZY = {
    "DiscreteVESP": "vesp.core.models",
    "MultiShellDiscreteVESP": "vesp.core.models",
    "SourceGeometry": "vesp.core.sources",
    "SourceSet": "vesp.core.sources",
    "fibonacci_sphere": "vesp.core.sources",
    "make_shell_sources": "vesp.core.sources",
}


def __getattr__(name: str):
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module 'vesp' has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_name), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
