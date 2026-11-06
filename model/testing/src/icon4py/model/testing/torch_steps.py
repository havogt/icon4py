# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""
The global steps on torch tensors, eager or under `torch.compile`, for the benchmarks.

The torch device is `ICON4PY_BENCH_TORCH_DEVICE` (default `cuda` when available). Under
`torch.compile` the graph must be whole unless `ICON4PY_BENCH_TORCH_FULLGRAPH=0`, and
`ICON4PY_BENCH_TORCH_EXPLAIN=1` prints `torch._dynamo.explain` of the first call. Eager torch is
timed only with `ICON4PY_BENCH_TORCH_EAGER=1`. A component in `COMPILE_BLOCKED` is not compiled
unless named in `ICON4PY_BENCH_TORCH_COMPILE_FORCE` (comma separated).
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable
from typing import Any

import gt4py.next as gtx
import numpy as np
from gt4py.next import common as gtx_common

from icon4py.model.common import utils as common_utils
from icon4py.model.common.utils import data_allocation as data_alloc


#: the components `torch.compile` cannot build yet, with the reason
COMPILE_BLOCKED: dict[str, str] = {}


def compile_blocked(component: str) -> str | None:
    """Why `component` is not compiled, or None."""
    forced = os.environ.get("ICON4PY_BENCH_TORCH_COMPILE_FORCE", "").split(",")
    return None if component in forced else COMPILE_BLOCKED.get(component)


def eager_timed() -> bool:
    return os.environ.get("ICON4PY_BENCH_TORCH_EAGER", "0") == "1"


def _is_host_or_cupy_array(obj: Any) -> bool:
    module = type(obj).__module__.split(".")[0]
    return module in ("numpy", "cupy") and not isinstance(obj, np.generic) and hasattr(obj, "shape")


def to_array_ns(obj: Any, xp: Any) -> Any:  # noqa: PLR0911 [too-many-return-statements]
    """
    `obj` with every field, connectivity and NumPy or CuPy array in `xp`, member by member; a
    CuPy array goes through host memory.
    """
    if isinstance(obj, gtx_common.Connectivity):
        return gtx.as_connectivity(
            obj.domain,
            obj.codomain,
            xp.asarray(obj.asnumpy()),
            skip_value=obj.skip_value,
            allocator=xp,
        )
    if isinstance(obj, gtx.Field):
        return gtx.as_field(obj.domain, xp.asarray(obj.asnumpy()), allocator=xp)
    if _is_host_or_cupy_array(obj):
        return xp.asarray(data_alloc.as_numpy(obj))
    if isinstance(obj, common_utils.PredictorCorrectorPair):
        return common_utils.PredictorCorrectorPair(*(to_array_ns(x, xp) for x in obj))
    if isinstance(obj, tuple):
        converted = [to_array_ns(x, xp) for x in obj]
        # a NamedTuple takes its fields positionally
        return type(obj)(*converted) if hasattr(obj, "_fields") else tuple(converted)
    if isinstance(obj, dict):
        return {k: to_array_ns(v, xp) for k, v in obj.items()}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.replace(
            obj,
            **{
                f.name: to_array_ns(getattr(obj, f.name), xp)
                for f in dataclasses.fields(obj)
                if f.init
            },
        )
    return obj


def _foreign_leaves(obj: Any, xp: Any, path: str, init_only: bool) -> list[str]:  # noqa: PLR0911 [too-many-return-statements]
    if isinstance(obj, gtx.Field):
        return _foreign_leaves(obj.ndarray, xp, f"{path}.ndarray", init_only)
    if isinstance(obj, (tuple, list)):
        return [
            p for i, x in enumerate(obj) for p in _foreign_leaves(x, xp, f"{path}[{i}]", init_only)
        ]
    if isinstance(obj, dict):
        return [
            p for k, v in obj.items() for p in _foreign_leaves(v, xp, f"{path}[{k!r}]", init_only)
        ]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return [
            p
            for f in dataclasses.fields(obj)
            for p in _foreign_leaves(getattr(obj, f.name), xp, f"{path}.{f.name}", init_only)
        ]
    if isinstance(obj, np.generic):
        return [] if init_only else [f"{path}: numpy scalar {type(obj).__name__}"]
    if not hasattr(obj, "shape"):
        return []
    if xp.__name__ == "torch":
        if not isinstance(obj, xp.Tensor):
            return [f"{path}: {type(obj).__module__}.{type(obj).__name__}, no torch tensor"]
        if obj.device.type != device(xp).type:
            return [f"{path}: torch tensor on {obj.device}, not {device(xp)}"]
        return []
    import jax  # type: ignore[import-not-found]  # noqa: PLC0415 [import-outside-top-level]

    if not isinstance(obj, jax.Array):
        return [f"{path}: {type(obj).__module__}.{type(obj).__name__}, no JAX array"]
    return []


def converted(obj: Any, xp: Any, path: str, *, init_only: bool = False) -> Any:
    """
    `to_array_ns(obj, xp)`, checked: every array in it is an `xp` array, a torch tensor on the
    torch device; no NumPy or CuPy array and no NumPy scalar is left. Raises naming the paths.

    With `init_only`, for what only a component's `__init__` reads, NumPy scalars are allowed: the
    component turns those it passes to its step into Python scalars, and the check of the step's
    inputs names any that it does not.
    """
    result = to_array_ns(obj, xp)
    if foreign := _foreign_leaves(result, xp, path, init_only):
        raise TypeError(
            f"{len(foreign)} arrays not converted to {xp.__name__}:\n  " + "\n  ".join(foreign[:20])
        )
    return result


def device(torch: Any) -> Any:
    default = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(os.environ.get("ICON4PY_BENCH_TORCH_DEVICE", default))


def step(run: Callable[..., Any], torch: Any, *, compiled: bool) -> Callable[..., Any]:
    """
    `run` eager or under `torch.compile`, synchronised after every call.

    `torch.compile` is entered outside any device mode (pytorch#156162, #140884); the fields
    `run` closes over and takes must already be on the device.
    """
    target = device(torch)
    called = run
    if compiled:
        called = torch.compile(
            run, fullgraph=os.environ.get("ICON4PY_BENCH_TORCH_FULLGRAPH", "1") == "1"
        )
    explain = compiled and os.environ.get("ICON4PY_BENCH_TORCH_EXPLAIN", "0") == "1"

    def synchronised(*args: Any) -> Any:
        nonlocal explain
        if explain:
            explain = False
            try:
                explanation = torch._dynamo.explain(run)(*args)
            except Exception as error:
                print(f"TORCH_EXPLAIN fails: {type(error).__name__}: {str(error).splitlines()[0]}")
            else:
                reasons = [str(r.reason).splitlines()[0] for r in explanation.break_reasons]
                print(
                    f"TORCH_EXPLAIN graphs {explanation.graph_count}, "
                    f"graph breaks {explanation.graph_break_count}, reasons {reasons}"
                )
        out = called(*args)
        if target.type == "cuda":
            torch.cuda.synchronize()
        return out

    return synchronised


def compare(
    computed: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    label: str,
    relative: float = 1e-10,
) -> list[str]:
    """
    The names whose max |difference| exceeds `relative` times their max |reference|.

    Relative to the field's scale, not pointwise: the vertical solves cancel to many orders below
    a field's maximum, where rounding of a different exp/pow or fusion order is relatively large.
    """
    assert computed.keys() == reference.keys(), (sorted(computed), sorted(reference))
    failures = []
    for name in sorted(reference):
        diff = float(np.nanmax(np.abs(computed[name] - reference[name])))
        scale = float(np.nanmax(np.abs(reference[name])))
        nan = int(np.isnan(computed[name]).sum()), int(np.isnan(reference[name]).sum())
        print(
            f"CHECK {label} {name}: max abs {diff:.3e}, max |ref| {scale:.3e}, "
            f"rel {diff / scale if scale else 0.0:.3e}, nan {nan[0]}/{nan[1]}"
        )
        if nan != (0, 0) or diff > relative * scale:
            failures.append(name)
    return failures
