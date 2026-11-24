# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""
A granule's `run` on GT4Py's eager embedded torch or JAX backend, under torch.compile or jax.jit.

The compiled `run` has the signature of the granule's and, like it, updates the fields of the
states it is given. Everything else, e.g. `dtime` or `initial_run` and a state's other leaves, is
a constant of the compiled program: one compilation per set of values.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
import logging
import typing
from collections.abc import Callable
from types import ModuleType
from typing import Any

import gt4py.next as gtx
import numpy as np
from gt4py.next import backend as gtx_backend, common as gtx_common
from gt4py.next.ffront import decorator as gtx_decorator

from icon4py.model.common import model_backends, state


def to_array_ns(obj: Any, xp: Any) -> Any:
    """`obj` with every field and connectivity in it on the arrays of `xp`, member by member."""
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
    if isinstance(obj, tuple):
        converted = [to_array_ns(x, xp) for x in obj]
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


def with_connectivities_on(grid: Any, xp: Any) -> Any:
    """`grid` with its connectivities on the arrays of `xp`."""
    return dataclasses.replace(
        grid, connectivities={k: to_array_ns(v, xp) for k, v in grid.connectivities.items()}
    )


def _is_state(value: Any) -> bool:
    return (
        dataclasses.is_dataclass(value)
        and not isinstance(value, type)
        and any(isinstance(getattr(value, f.name), gtx.Field) for f in dataclasses.fields(value))
    )


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


# An argument of `run` as a template: a state's fields refer to their position in the list of the
# distinct fields, which keeps a field that two states share (e.g. a component's in-place input
# and output) one field in the trace; everything else is a constant of the compiled program.
_Template = tuple[tuple[str, Any], ...]


def _template(arguments: dict[str, Any], fields: list[gtx.Field]) -> _Template:
    positions: dict[int, int] = {}

    def leaf(value: Any) -> Any:
        if isinstance(value, gtx.Field):
            if id(value) not in positions:
                positions[id(value)] = len(fields)
                fields.append(value)
            return ("field", positions[id(value)])
        return ("constant", _python_scalar(value))

    return tuple(
        (
            name,
            (
                type(value),
                tuple((f.name, leaf(getattr(value, f.name))) for f in dataclasses.fields(value)),
            )
            if _is_state(value)
            else leaf(value),
        )
        for name, value in arguments.items()
    )


def _rebuilt(template: _Template, fields: list[gtx.Field]) -> dict[str, Any]:
    def leaf(value: tuple[str, Any]) -> Any:
        kind, payload = value
        return fields[payload] if kind == "field" else payload

    return {
        name: kind_or_type(**{n: leaf(v) for n, v in payload})
        if isinstance(kind_or_type, type)
        else leaf((kind_or_type, payload))
        for name, (kind_or_type, payload) in template
    }


def _library(backend: Any) -> str:
    # the eager embedded backends only: the jit backends compile each program themselves
    if type(backend) is gtx_backend.EmbeddedBackend:
        return typing.cast(ModuleType, backend.allocator).__name__.split(".")[0]
    raise ValueError(f"Compiled runs need an eager embedded backend, got '{backend}'.")


def compiled_run(granule: Any, backend: Any) -> Callable[..., None]:
    """`granule.run` under `torch.compile` or `jax.jit`, by the array library of `backend`."""
    compile_ = {"torch": _torch_compile, "jax": _jax_jit}.get(library := _library(backend))
    if compile_ is None:
        raise ValueError(f"No compiled run on '{library}'.")
    signature = inspect.signature(granule.run)
    cache: dict[_Template, Callable[[list[gtx.Field]], None]] = {}

    def run(*args: Any, **kwargs: Any) -> None:
        arguments = signature.bind(*args, **kwargs).arguments
        fields: list[gtx.Field] = []
        template = _template(arguments, fields)
        if template not in cache:
            # read the declarations of the states before the trace, see `state._declarations`
            for value in arguments.values():
                if isinstance(value, state.State):
                    state.fields(value)
            cache[template] = compile_(granule, template)
        cache[template](fields)

    return run


_programs: dict[int, Callable[..., Any]] = {}


@functools.cache
def _nonstrict_program_call() -> Callable[..., tuple]:
    import torch  # type: ignore[import-not-found]  # noqa: PLC0415 [import-outside-top-level]

    @torch.compiler.nonstrict_trace
    def call(program_id: int, args: tuple, kwargs: dict[str, Any]) -> tuple:
        _programs[program_id](*args, **kwargs)
        # a `None` output is not supported by `nonstrict_trace`
        return ()

    return call


class _NonstrictProgram:
    """An embedded program bound by `setup_program`, called through `torch.compiler.nonstrict_trace`."""

    def __init__(self, bound: functools.partial) -> None:
        self._program = bound.func
        self._args = bound.args
        self._keywords = bound.keywords
        _programs[id(self._program)] = self._program
        self._call = _nonstrict_program_call()

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        # the bound fields and connectivities are arguments of the traced function: closed over by
        # it, a connectivity is not supported (GT4Py ADR 0029)
        self._call(id(self._program), (*self._args, *args), {**self._keywords, **kwargs})


def _call_embedded_programs_nonstrictly(granule: Any) -> None:
    # Dynamo cannot trace embedded GT4Py code, which has to run in a `nonstrict_trace` call
    for name, value in vars(granule).items():
        if (
            isinstance(value, functools.partial)
            and isinstance(value.func, gtx_decorator.Program)
            and model_backends.is_embedded(value.func.backend)
        ):
            setattr(granule, name, _NonstrictProgram(value))


def _torch_compile(granule: Any, template: _Template) -> Callable[[list[gtx.Field]], None]:
    import torch  # noqa: PLC0415 [import-outside-top-level]

    _call_embedded_programs_nonstrictly(granule)
    # the granules log in `run`, which Dynamo cannot trace
    torch._dynamo.config.ignore_logging_functions.add(logging.Logger.debug)
    # GT4Py puts the domains in the pytree context, which takes no symbolic ints: a recompilation
    # for other grid sizes must not make them dynamic
    return torch.compile(
        lambda fields: granule.run(**_rebuilt(template, fields)), fullgraph=True, dynamic=False
    )


def _jax_jit(granule: Any, template: _Template) -> Callable[[list[gtx.Field]], None]:
    import jax  # type: ignore[import-not-found]  # noqa: PLC0415 [import-outside-top-level]

    # The programs the granule binds with `setup_program` share the grid's offset provider dict.
    # A connectivity closed over by a jax trace is not supported (GT4Py ADR 0029), so the
    # connectivities are jit arguments and replace the concrete ones during the trace.
    offset_provider = granule._grid.connectivities
    concrete_offset_provider = dict(offset_provider)
    # the granule's own fields get traced buffers in the trace, which must not outlive it
    scratch_fields = [value for value in vars(granule).values() if isinstance(value, gtx.Field)]

    def traced_run(connectivities: dict, fields: list[gtx.Field]) -> list[Any]:
        concrete_buffers = [field.ndarray for field in scratch_fields]
        offset_provider.update(connectivities)
        try:
            granule.run(**_rebuilt(template, fields))
        finally:
            offset_provider.update(concrete_offset_provider)
            for field, buffer in zip(scratch_fields, concrete_buffers, strict=True):
                object.__setattr__(field, "_ndarray", buffer)
        # JAX's `out` assignment gives the fields new buffers
        return [field.ndarray for field in fields]

    jitted = jax.jit(traced_run)

    def run(fields: list[gtx.Field]) -> None:
        for field, buffer in zip(fields, jitted(concrete_offset_provider, fields), strict=True):
            object.__setattr__(field, "_ndarray", buffer)

    return run
