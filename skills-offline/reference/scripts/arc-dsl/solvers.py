"""Lazy compatibility loader for one-file-per-task ARC-DSL solvers."""

from importlib import import_module
from pathlib import Path


_SOLVER_PACKAGE = "solver_scripts"
__all__ = tuple(
    sorted(path.stem for path in Path(__file__).with_name(_SOLVER_PACKAGE).glob("solve_*.py"))
)


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{_SOLVER_PACKAGE}.{name}")
    solver = getattr(module, name)
    globals()[name] = solver
    return solver


def get_solver_names():
    return __all__
