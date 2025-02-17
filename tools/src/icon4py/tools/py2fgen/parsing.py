# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import importlib
import types
import typing
from inspect import signature, unwrap
from types import ModuleType, NoneType
from typing import Any, Callable, List, Sequence, Union

import numpy as np
from gt4py.next.type_system import (
    type_specifications as gtx_type_specifications,
    type_translation as gtx_type_translation,
)

from icon4py.tools.py2fgen.template import CffiPlugin, Func, FuncParameter
from icon4py.tools.py2fgen.utils import parse_type_spec


def parse(module_name: str, functions: list[str], plugin_name: str) -> CffiPlugin:
    module = importlib.import_module(module_name)
    parsed_functions = [_parse_function(module, f) for f in functions]

    return CffiPlugin(
        module_name=module_name,
        plugin_name=plugin_name,
        functions=parsed_functions,
    )


def _parse_function(module: ModuleType, function_name: str) -> Func:
    func = unwrap(getattr(module, function_name))
    params = _parse_params(func)
    return Func(name=function_name, args=params)


def _is_optional_type_hint(type_hint: Any) -> bool:
    return typing.get_origin(type_hint) is Union and typing.get_args(type_hint)[1] is NoneType


def _unpack_optional_type_hint(type_hint: Any) -> tuple[Any, bool]:
    if _is_optional_type_hint(type_hint):
        return typing.get_args(type_hint)[0], True
    else:
        return type_hint, False


def _canonical_type(type_hint: Any) -> Any:
    canonical_type = (
        typing.get_origin(type_hint)
        if isinstance(type_hint, types.GenericAlias) or type(type_hint).__module__ == "typing"
        else type_hint
    )
    return canonical_type, typing.get_args(type_hint)


def _from_ndarray(
    type_hint: Any, args: Sequence[Any]
) -> tuple[int, gtx_type_specifications.ScalarKind]:
    rank = len(typing.get_args(args[0]))
    dtype_type_hint = _canonical_type(args[1])[0]
    if dtype_type_hint is np.dtype:
        dtype = typing.get_args(args[1])[0]
    dtype = gtx_type_translation.from_type_hint(dtype).kind
    return rank, dtype


def _parse_params(func: Callable) -> List[FuncParameter]:
    sig_params = signature(func, follow_wrapped=False).parameters
    params = []
    for s, param in sig_params.items():
        non_optional_type, is_optional = _unpack_optional_type_hint(param.annotation)
        canonical_type, args = _canonical_type(non_optional_type)
        if canonical_type is np.ndarray:
            rank, dtype = _from_ndarray(canonical_type, args)

            def _py_field_renderer(arg):
                return f"{arg.name} = wrapper_utils.as_numpy(ffi, {arg.name}, ts.ScalarKind.{arg.d_type.name}, {','.join(arg.size_args)})"

            meta = {"py_renderer": _py_field_renderer}
            use_device = False
        else:
            dims, dtype = parse_type_spec(gtx_type_translation.from_type_hint(non_optional_type))
            rank = len(dims) if len(dims) > 0 else None
            meta = {"dimensions": dims if len(dims) > 0 else None}

            def _py_field_renderer(arg):
                return f"{arg.name} = wrapper_utils.as_field(ffi, xp, {arg.name}, ts.ScalarKind.{arg.d_type.name}, {arg.domain}, {arg.is_optional})"

            meta["py_renderer"] = _py_field_renderer
            use_device = True

        params.append(
            FuncParameter(
                name=s,
                rank=rank,
                d_type=dtype,
                is_optional=is_optional,
                use_device=use_device,
                meta=meta,
            )
        )

    return params
