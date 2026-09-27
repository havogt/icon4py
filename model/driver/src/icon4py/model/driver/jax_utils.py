# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""Conversion of numpy-backed driver states to JAX, for running the global steps under jax.jit."""

from __future__ import annotations

import dataclasses
import types
from typing import Any

import gt4py.next as gtx
from gt4py.next import common as gtx_common

import icon4py.model.common.utils as common_utils
from icon4py.model.common.grid import base as grid_base


def import_jax() -> types.ModuleType:
    """Import jax with 64-bit floats enabled; this must happen before any jax array is created."""
    import jax  # type: ignore[import-not-found]  # noqa: PLC0415 [import-outside-top-level] # jax is an optional dependency

    jax.config.update("jax_enable_x64", True)
    return jax


def without_skip_values(obj: Any) -> Any:
    """A neighbor table with its skip values replaced by a valid neighbor of the same entry."""
    if not gtx_common.is_neighbor_table(obj) or obj.skip_value is None:
        return obj
    table = grid_base._replace_skip_values(obj.domain.dims, obj.asnumpy().copy())
    return gtx.as_connectivity(obj.domain, obj.codomain, table)  # type: ignore[arg-type]  # NDArray is a union


def to_jax(obj: Any) -> Any:
    """
    Copy the fields and connectivities in `obj` to JAX arrays.

    Tuples, predictor-corrector pairs and dataclasses are converted member by member; anything
    else, scalars included, is returned as is.
    """
    jnp = import_jax().numpy
    if isinstance(obj, gtx_common.Connectivity):
        return gtx.as_connectivity(
            obj.domain,
            obj.codomain,
            jnp.asarray(obj.asnumpy()),
            skip_value=obj.skip_value,
            allocator=jnp,
        )
    if isinstance(obj, gtx.Field):
        return gtx.as_field(obj.domain, jnp.asarray(obj.asnumpy()), allocator=jnp)
    if isinstance(obj, common_utils.PredictorCorrectorPair):
        return common_utils.PredictorCorrectorPair(*(to_jax(x) for x in obj))
    if isinstance(obj, tuple):
        return tuple(to_jax(x) for x in obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.replace(
            obj,
            **{f.name: to_jax(getattr(obj, f.name)) for f in dataclasses.fields(obj) if f.init},
        )
    return obj
