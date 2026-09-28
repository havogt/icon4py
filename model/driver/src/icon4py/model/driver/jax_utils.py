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
import socket
import types
from typing import Any

import gt4py.next as gtx
import numpy as np
from gt4py.next import common as gtx_common

import icon4py.model.common.utils as common_utils


def import_jax() -> types.ModuleType:
    """Import jax with 64-bit floats enabled; this must happen before any jax array is created."""
    import jax  # type: ignore[import-not-found]  # noqa: PLC0415 [import-outside-top-level] # jax is an optional dependency

    jax.config.update("jax_enable_x64", True)
    return jax


def initialize_distributed(comm: Any) -> None:
    """
    Join every rank of the MPI communicator `comm` into one multi-process JAX runtime.

    Must run before JAX creates any array. The collectives between the CPU devices of the
    processes go through gloo.
    """
    jax = import_jax()
    jax.config.update("jax_cpu_collectives_implementation", "gloo")
    address = None
    if comm.rank == 0:
        with socket.socket() as sock:
            sock.bind(("", 0))
            address = f"{socket.gethostname()}:{sock.getsockname()[1]}"
    jax.distributed.initialize(
        coordinator_address=comm.bcast(address, root=0),
        num_processes=comm.size,
        process_id=comm.rank,
        local_device_ids=[0],
    )


def rank_mesh(axis_name: str) -> Any:
    """
    A 1-D mesh over the devices of all processes, device `i` belonging to process (rank) `i`.

    `jax.make_mesh` refuses devices spread over several nodes.
    """
    jax = import_jax()
    devices = sorted(jax.devices(), key=lambda device: device.process_index)
    if [device.process_index for device in devices] != list(range(len(devices))):
        raise ValueError("The mesh over the ranks needs exactly one device per process.")
    return jax.sharding.Mesh(np.array(devices), (axis_name,))


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
