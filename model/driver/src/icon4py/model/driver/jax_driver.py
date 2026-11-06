# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""
The driver time loop with the single-field-operator global steps under jax.jit.

The setup is the numpy one of `driver.Icon4pyDriver`; the states are converted to JAX once and
threaded through the stateless `SolveNonhydroGlobal`, `DiffusionGlobal` and
`TracerAdvectionGlobal` steps, and `physics_global.PhysicsGlobal` after them. No output.

On a distributed grid the steps return their outputs on the owned points only. Between two jitted
steps the driver builds new local fields from them, with the halo filled by a halo exchange on the
host or, from CuPy buffers, on the device (`driver_utils.HaloExchange`); with `jit_time_step` the
whole time step is one jitted function, and the exchange a `buffer_callback` inside it, on XLA's
stream on a GPU, whatever `halo_exchange` says. The halo must be deep enough for one step:
`driver_utils.JAX_EXTRA_HALO_RINGS`, with physics `driver_utils.JAX_PHYSICS_EXTRA_HALO_RINGS`.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import datetime
import functools
import logging
from collections.abc import Callable
from typing import Any

import gt4py.next as gtx
import numpy as np
from gt4py.next import common as gtx_common

import icon4py.model.common.utils as common_utils
from icon4py.model.atmosphere.dycore import dycore_states, solve_nonhydro_global
from icon4py.model.common import dimension as dims
from icon4py.model.common.decomposition import definitions as decomposition_defs
from icon4py.model.common.grid import base as grid_base
from icon4py.model.common.states import (
    nonhydro_states,
    prognostic_state as prognostics,
    tracer_prep_adv_states,
    tracer_states,
)
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.driver import driver, driver_states, driver_utils, jax_utils, spmd_layout


log = logging.getLogger(__name__)

_PROGNOSTICS = ("rho", "w", "vn", "exner", "theta_v")
_TENDENCY_PAIRS = ("normal_wind_advective_tendency", "vertical_wind_advective_tendency")
_SUBSTEP_GROUPS = ("prognostic", "diagnostic", "intermediate", "prep_adv")
_MESH_AXIS = "rank"

#: The state fields the global dynamical core and diffusion steps read outside the owned points,
#: from `required_indices` of `_solve_nonhydro_global_step` and `_diffusion_global_step`. A
#: predictor-corrector pair counts as one field: the driver swaps its elements between the
#: exchange and the step.
_DYCORE_HALO_READS = frozenset(
    {
        "prognostic.vn",
        "prognostic.w",
        "prognostic.rho",
        "prognostic.exner",
        "prognostic.theta_v",
        "diagnostic.tangential_wind",
        "diagnostic.contravariant_correction_at_cells_on_half_levels",
        "diagnostic.theta_v_at_cells_on_half_levels",
        "diagnostic.perturbed_exner_at_cells_on_model_levels",
        "diagnostic.rho_at_cells_on_half_levels",
        "diagnostic.exner_tendency_due_to_slow_physics",
        "diagnostic.normal_wind_tendency_due_to_slow_physics_process",
        "intermediate.tangential_wind_on_half_levels",
        "intermediate.contravariant_correction_at_edges_on_model_levels",
        "diagnostic.normal_wind_advective_tendency",
        "diagnostic.vertical_wind_advective_tendency",
    }
)
_DIFFUSION_HALO_READS = frozenset({"prognostic.vn", "prognostic.w", "prognostic.theta_v"})
#: The dynamical core's fluxes for tracer advection, by the name `TracerPrepAdvState` gives them.
_TRACER_PREP_ADV = {
    "vn_traj": "vn_traj",
    "mass_flx_me": "mass_flx_me",
    "mass_flx_ic": "dynamical_vertical_mass_flux_at_cells_on_half_levels",
}
_TRACER_ADVECTION_HALO_READS = frozenset(f"prep_adv.{name}" for name in _TRACER_PREP_ADV.values())


def _shallow_dict(obj: Any) -> dict[str, Any]:
    return {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}


def _diagnostic_to_dict(state: nonhydro_states.DiagnosticStateNonHydro) -> dict[str, Any]:
    d = _shallow_dict(state)
    for name in _TENDENCY_PAIRS:
        d[name] = tuple(d[name])
    return d


def _diagnostic_from_dict(d: dict[str, Any]) -> nonhydro_states.DiagnosticStateNonHydro:
    fields: dict[str, Any] = {
        k: common_utils.PredictorCorrectorPair(*v) if k in _TENDENCY_PAIRS else v
        for k, v in d.items()
    }
    return nonhydro_states.DiagnosticStateNonHydro(**fields)


def _lift_constants(value: Any) -> Any:
    """
    The fields in `value`, as a pytree that `_bind_constants` puts back; None if there are none.

    Fields, tuples and dicts of fields, and the fields of dataclasses that `dataclasses.replace`
    can rebuild are lifted; a grid is not, its connectivities are the offset provider's.
    """
    if isinstance(value, gtx.Field):
        return value
    if isinstance(value, tuple) and value and all(isinstance(v, gtx.Field) for v in value):
        return value
    if isinstance(value, dict) and value and all(isinstance(v, gtx.Field) for v in value.values()):
        return dict(value)
    if (
        dataclasses.is_dataclass(value)
        and not isinstance(value, (type, grid_base.Grid))
        and _rebuildable(value)
    ):
        lifted = {
            f.name: lifted
            for f in dataclasses.fields(value)
            if (lifted := _lift_constants(getattr(value, f.name))) is not None
        }
        return lifted or None
    return None


def _rebuildable(value: Any) -> bool:
    if not all(f.init for f in dataclasses.fields(value)):
        return False
    try:
        dataclasses.replace(value)
    except (TypeError, ValueError):
        return False
    return True


def _bind_constants(value: Any, lifted: Any) -> Any:
    if isinstance(lifted, dict) and dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.replace(
            value, **{k: _bind_constants(getattr(value, k), v) for k, v in lifted.items()}
        )
    return lifted


def _with_constants(granule: Any, constants: dict[str, Any]) -> Any:
    """A shallow copy of `granule` with the lifted attributes `constants` bound."""
    bound = copy.copy(granule)
    for name, lifted in constants.items():
        setattr(bound, name, _bind_constants(getattr(granule, name), lifted))
    return bound


def _constants_of(granule: Any) -> dict[str, Any]:
    return {
        name: lifted
        for name, value in vars(granule).items()
        if (lifted := _lift_constants(value)) is not None
    }


_PHYSICS_PROCESSES = ("muphys", "surface", "tmx")


def _physics_constants_of(physics: Any) -> dict[str, Any]:
    """`_constants_of` the physics and of each of its processes."""
    return {
        "physics": _constants_of(physics),
        **{
            process: _constants_of(granule)
            for process in _PHYSICS_PROCESSES
            if (granule := getattr(physics, process)) is not None
        },
    }


def _with_physics_constants(physics: Any, constants: dict[str, Any]) -> Any:
    bound = _with_constants(physics, constants["physics"])
    for process in _PHYSICS_PROCESSES:
        if (granule := getattr(physics, process)) is not None:
            setattr(bound, process, _with_constants(granule, constants[process]))
    return bound


def _granule_constants(
    granules: driver_utils.GlobalGranules, as_arguments: bool, *, spmd: bool
) -> dict[str, Any]:
    """
    The lifted static fields of each granule, None for a granule that closes over them.

    Passed as arguments, the static fields are not constants of the jitted programs.
    """
    constants = {
        f.name: (
            None
            if not as_arguments or (granule := getattr(granules, f.name)) is None
            else _physics_constants_of(granule)
            if f.name == "physics"
            else _constants_of(granule)
        )
        for f in dataclasses.fields(granules)
    }
    if spmd and (
        closed_over := [
            name
            for name, lifted in constants.items()
            if getattr(granules, name) is not None and lifted is None
        ]
    ):
        # each rank would embed its own static fields in the program: the ranks lower different
        # programs, and the collectives in them wait for each other forever
        raise ValueError(
            "One SPMD program needs the static fields of every granule passed as arguments; "
            f"closed over: {closed_over}."
        )
    return constants


def _unfilled_local(owned: Any, num_local: int, finite_padding: bool) -> Any:
    jax = jax_utils.import_jax()
    shape = (num_local - owned.shape[0], *owned.shape[1:])
    padding = (
        jax.numpy.broadcast_to(owned[0], shape)
        if finite_padding
        else jax.numpy.full(shape, jax.numpy.nan, owned.dtype)
    )
    return jax.numpy.concatenate([owned, padding])


def _cell_halo_rings(grid: grid_base.Grid, owned: Any) -> int:
    """How many rings of vertex-neighbour cells around the owned cells the local grid holds."""
    c2v = grid.connectivities["C2V"].asnumpy()
    v2c = grid.connectivities["V2C"].asnumpy()
    reached = np.asarray(owned, dtype=bool).copy()
    rings = 0
    while not reached.all():
        vertices = np.unique(c2v[reached])
        cells = v2c[vertices[vertices >= 0]]
        ring = np.zeros_like(reached)
        ring[cells[cells >= 0]] = True
        ring &= ~reached
        if not ring.any():
            break
        reached |= ring
        rings += 1
    return rings


def exchange_collective(
    owned: Any, send: Any, recv: Any, num_local: int, finite_padding: bool = False
) -> Any:
    """
    The local rows of the padded layout from the `owned` ones, the halo received from the owners.

    Runs inside `shard_map` over the mesh axis of the ranks; `send` and `recv` are this rank's
    `spmd_layout.PaddedLayout` tables of the dimension. The halo rows the exchange leaves unfilled
    are NaN, or with `finite_padding` copies of row 0: never read, but NaN there turns into NaN in
    gradients.
    """
    jax = jax_utils.import_jax()
    local = _unfilled_local(owned, num_local, finite_padding)
    received = jax.lax.all_to_all(owned[send], _MESH_AXIS, 0, 0)
    return local.at[recv].set(received, mode="drop")


def exchange_coloured(
    owned: Any,
    send_round: Any,
    recv_round: Any,
    rounds: tuple[tuple[tuple[int, int], ...], ...],
    num_local: int,
    *,
    finite_padding: bool = False,
) -> Any:
    """
    `exchange_collective` in the rounds of pairwise swaps of `spmd_layout.PaddedLayout.rounds`.

    A rank without a partner in a round receives zeros from `ppermute`; its row of `recv_round`
    drops them all.
    """
    jax = jax_utils.import_jax()
    local = _unfilled_local(owned, num_local, finite_padding)
    for k, perm in enumerate(rounds):
        received = jax.lax.ppermute(owned[send_round[k]], _MESH_AXIS, perm=perm)
        local = local.at[recv_round[k]].set(received, mode="drop")
    return local


class JaxIcon4pyDriver(driver.Icon4pyDriver):
    def __init__(
        self,
        *,
        global_granules: driver_utils.GlobalGranules,
        halo_exchange: driver_utils.HaloExchange = driver_utils.HaloExchange.HOST,
        exchange_read_fields_only: bool = False,
        constants_as_arguments: bool = True,
        jit_time_step: bool = False,
        layout: spmd_layout.PaddedLayout | None = None,
        finite_halo_padding: bool = False,
        spmd_transport: driver_utils.SpmdTransport = driver_utils.SpmdTransport.ALL_TO_ALL,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._distributed = not kwargs["process_props"].is_single_rank()
        self._num_local = {dim: self.grid.size[dim] for dim in dims.horizontal_dims()}
        if self._distributed:
            info = self.decomposition_info
            if not (
                info.halo_levels(dims.CellDim)
                == decomposition_defs.DecompositionFlag.EXTRA_HALO_LEVEL
            ).any():
                raise ValueError(
                    f"The JAX driver needs {driver_utils.JAX_EXTRA_HALO_RINGS} extra halo rings "
                    "on a distributed grid."
                )
            self._num_owned = {}
            for dim in dims.horizontal_dims():
                owner_mask = self._xp.asarray(info.owner_mask(dim))
                num_owned = int(owner_mask.sum())
                assert owner_mask[:num_owned].all(), f"the owned {dim.value}s are not first"
                self._num_owned[dim] = num_owned
        self._layout = layout
        # NaN in the halo rows no exchange fills makes a missing exchange show; for gradients they
        # must be finite
        self.finite_halo_padding = finite_halo_padding
        self.spmd_transport = driver_utils.SpmdTransport(spmd_transport)
        if layout is not None:
            if not (jit_time_step and constants_as_arguments):
                raise ValueError(
                    "One SPMD program needs the whole time step jitted and the static fields "
                    "passed as arguments."
                )
            self._num_owned = dict(layout.padded_owned)
            self._num_local = dict(layout.padded_local)
        self._traced_tables: dict[str, Any] | None = None
        if self.io_monitor is not None or self.tendencies is not None:
            raise NotImplementedError("The JAX driver does not write output or apply tendencies.")
        if self.config.graupel is not None:
            raise NotImplementedError("The JAX driver does not run the NWP graupel scheme.")
        if global_granules.physics is not None and self._distributed:
            rings = _cell_halo_rings(
                self.grid, self._xp.asarray(self.decomposition_info.owner_mask(dims.CellDim))
            )
            if rings < 1 + driver_utils.JAX_PHYSICS_EXTRA_HALO_RINGS:
                raise ValueError(
                    f"The JAX driver's physics needs {driver_utils.JAX_PHYSICS_EXTRA_HALO_RINGS} extra "
                    f"halo rings on a distributed grid; this one has {rings - 1}."
                )
        self._tracer_names = (
            self.config.tracer_config.active_names if self.config.tracer_config is not None else ()
        )
        if self._tracer_names and global_granules.tracer_advection is None:
            raise ValueError("Active tracers need a tracer advection configuration.")
        self.global_granules = global_granules
        self.halo_exchange = driver_utils.HaloExchange(halo_exchange)
        self.exchange_read_fields_only = exchange_read_fields_only
        self._constants = _granule_constants(
            global_granules, constants_as_arguments, spmd=layout is not None
        )
        # the Godunov splitting order, flipped once per time step for all tracers as in ICON
        self._even_timestep = False
        self._traces = 0
        self.jit_time_step = jit_time_step
        if jit_time_step and self._distributed and layout is None:
            from mpi4py import MPI  # noqa: PLC0415 [import-outside-top-level]

            # XLA may run the exchange callbacks on its own thread
            if MPI.Query_thread() < MPI.THREAD_SERIALIZED:
                log.warning(
                    "MPI is initialized below MPI_THREAD_SERIALIZED, but the halo exchanges inside "
                    "the jitted time step may call MPI from another thread than the main one."
                )
        self._stream: Any = None
        self._in_flight: tuple[Any, list] | None = None
        self.on_step_end: Callable[[int, prognostics.PrognosticState], None] | None = None
        self._jax = jax_utils.import_jax()
        self._jitted: dict[tuple, Callable] = {}

    def _with_halo(self, tree: dict[str, Any], reads: frozenset[str] | None = None) -> dict:
        """
        New local fields from the owned points of the fields in `tree`, with the halo exchanged.

        `tree` maps state groups to their fields. With `reads`, only the fields named in it (as
        `group.field`) are exchanged, the other ones get a NaN halo; local-size fields not in
        `reads` are passed through. Anything else is passed through.
        """
        if not self._distributed:
            return tree
        jax = self._jax
        paths_and_leaves, treedef = jax.tree_util.tree_flatten_with_path(
            tree, is_leaf=lambda x: isinstance(x, gtx.Field)
        )
        leaves = [leaf for _, leaf in paths_and_leaves]
        read = [
            reads is None or f"{path[0].key}.{path[1].key}" in reads for path, _ in paths_and_leaves
        ]
        traced = any(
            isinstance(leaf, gtx.Field) and isinstance(leaf.ndarray, jax.core.Tracer)
            for leaf in leaves
        )
        with_halo = self._with_halo_traced if traced else self._with_halo_eager
        return jax.tree_util.tree_unflatten(treedef, with_halo(leaves, read))

    def _is_horizontal(self, leaf: Any) -> bool:
        return isinstance(leaf, gtx.Field) and leaf.domain.dims[0] in self._num_owned

    def _local_domain(self, field: gtx.Field) -> gtx_common.Domain:
        dim = field.domain.dims[0]
        return field.domain.replace(
            dim, gtx_common.NamedRange(dim, gtx_common.unit_range((0, self._num_local[dim])))
        )

    def _nan_padded(self, field: gtx.Field) -> gtx.Field:
        """
        An owned-size field extended to the local size with a NaN halo; others as they are.

        With `finite_halo_padding` the halo repeats the last owned row instead: NaN there would turn
        into NaN in gradients.
        """
        dim = field.domain.dims[0]
        num_owned, num_local = self._num_owned[dim], self._num_local[dim]
        if field.ndarray.shape[0] != num_owned:
            return field
        jnp = self._jax.numpy
        padding = [(0, num_local - num_owned)] + [(0, 0)] * (field.ndarray.ndim - 1)
        return gtx.as_field(
            self._local_domain(field),
            jnp.pad(field.ndarray, padding, mode="edge")
            if self.finite_halo_padding
            else jnp.pad(field.ndarray, padding, constant_values=jnp.nan),
            allocator=jnp,
        )

    def _with_halo_eager(self, leaves: list, read: list[bool]) -> list:
        jax = self._jax
        mode = self.halo_exchange
        on_device = mode != driver_utils.HaloExchange.HOST
        xp = data_alloc.array_ns(on_device)
        stream = self._device_stream() if mode == driver_utils.HaloExchange.DEVICE_STREAM else None
        new_leaves = list(leaves)
        with stream if stream is not None else contextlib.nullcontext():
            buffers: dict[int, Any] = {}
            # views of the JAX arrays the device copies read from; they must outlive the copies
            views = []
            for i, leaf in enumerate(leaves):
                if not self._is_horizontal(leaf):
                    continue
                if not read[i]:
                    new_leaves[i] = self._nan_padded(leaf)
                    continue
                num_owned = self._num_owned[leaf.domain.dims[0]]
                # NaN marks every halo point the exchange leaves unfilled
                buffer = xp.full(
                    self._local_domain(leaf).shape, xp.nan, dtype=leaf.dtype.scalar_type
                )
                if on_device:
                    views.append(xp.from_dlpack(leaf.ndarray[:num_owned]))
                    buffer[:num_owned] = views[-1]
                else:
                    buffer[:num_owned] = xp.asarray(leaf.ndarray[:num_owned])
                buffers[i] = buffer
            if stream is not None:
                self._hold_until_done(stream.record(), views)
            elif on_device:
                # GHEX starts after the default stream only; the copies are on CuPy's current stream
                xp.cuda.get_current_stream().synchronize()
            exchange_stream = {"stream": stream if stream is not None else decomposition_defs.BLOCK}
            for dim in self._num_owned:
                dim_buffers = [b for i, b in buffers.items() if leaves[i].domain.dims[0] == dim]
                if dim_buffers:
                    self.exchange.exchange(
                        dim, *dim_buffers, **(exchange_stream if on_device else {})
                    )
            if on_device and stream is None:
                # GHEX unpacks on its own streams; JAX must not see a buffer before they are done
                xp.cuda.runtime.deviceSynchronize()
            # with `stream` current, DLPack makes JAX wait for the exchange on `stream`
            to_jax = jax.numpy.from_dlpack if on_device else jax.numpy.asarray
            for i, buffer in buffers.items():
                new_leaves[i] = gtx.as_field(
                    self._local_domain(leaves[i]), to_jax(buffer), allocator=jax.numpy
                )
        return new_leaves

    def _with_halo_traced(self, leaves: list, read: list[bool]) -> list:
        """`_with_halo` inside a jitted function: all exchanged fields go through one callback."""
        from jax.experimental import (  # type: ignore[import-not-found]  # noqa: PLC0415 [import-outside-top-level]
            buffer_callback,
        )

        jax = self._jax
        new_leaves = list(leaves)
        exchanged = []
        for i, leaf in enumerate(leaves):
            if not self._is_horizontal(leaf):
                continue
            if read[i]:
                exchanged.append(i)
            else:
                new_leaves[i] = self._nan_padded(leaf)
        if not exchanged:
            return new_leaves
        if self._layout is not None:
            for i in exchanged:
                new_leaves[i] = gtx.as_field(
                    self._local_domain(leaves[i]),
                    self._exchange_collective(leaves[i]),
                    allocator=jax.numpy,
                )
            return new_leaves
        field_dims = tuple(leaves[i].domain.dims[0] for i in exchanged)
        owned = [leaves[i].ndarray[: self._num_owned[dim]] for i, dim in zip(exchanged, field_dims)]
        exchange = buffer_callback.buffer_callback(
            functools.partial(
                self._exchange_in_callback, field_dims, jax.default_backend() == "gpu"
            ),
            [
                jax.ShapeDtypeStruct((self._num_local[dim], *array.shape[1:]), array.dtype)
                for dim, array in zip(field_dims, owned)
            ],
            # every rank has to issue every exchange
            has_side_effect=True,
        )
        for i, array in zip(exchanged, exchange(*owned)):
            new_leaves[i] = gtx.as_field(self._local_domain(leaves[i]), array, allocator=jax.numpy)
        return new_leaves

    def _exchange_collective(self, field: gtx.Field) -> Any:
        dim = field.domain.dims[0]
        assert self._traced_tables is not None
        tables = self._traced_tables[dim.value]
        owned = field.ndarray[: self._num_owned[dim]]
        if self.spmd_transport == driver_utils.SpmdTransport.PPERMUTE:
            assert self._layout is not None
            return exchange_coloured(
                owned,
                tables["send"],
                tables["recv"],
                self._layout.rounds,
                self._num_local[dim],
                finite_padding=self.finite_halo_padding,
            )
        return exchange_collective(
            owned, tables["send"], tables["recv"], self._num_local[dim], self.finite_halo_padding
        )

    def _exchange_in_callback(
        self,
        field_dims: tuple[gtx.Dimension, ...],
        on_gpu: bool,
        context: Any,
        outs: list,
        *owned: Any,
    ) -> None:
        from jax.experimental import buffer_callback  # noqa: PLC0415 [import-outside-top-level]

        # an exchange that ran at another stage too would not be matched by the other ranks
        if context.stage != buffer_callback.ExecutionStage.EXECUTE:
            return
        xp = data_alloc.array_ns(on_gpu)
        stream = xp.cuda.ExternalStream(context.stream) if on_gpu else None
        with stream if stream is not None else contextlib.nullcontext():
            buffers = []
            for dim, out, array in zip(field_dims, outs, owned):
                buffer = xp.asarray(out)
                num_owned = self._num_owned[dim]
                buffer[:num_owned] = xp.asarray(array)
                # NaN marks every halo point the exchange leaves unfilled
                buffer[num_owned:] = xp.nan
                buffers.append(buffer)
            for dim in self._num_owned:
                dim_buffers = [b for d, b in zip(field_dims, buffers) if d == dim]
                if not dim_buffers:
                    continue
                self.exchange.exchange(
                    dim,
                    *dim_buffers,
                    stream=stream if stream is not None else decomposition_defs.BLOCK,
                )

    def _reads(self, fields: frozenset[str]) -> frozenset[str] | None:
        return fields if self.exchange_read_fields_only else None

    def _device_stream(self) -> Any:
        if self._stream is None:
            self._stream = data_alloc.array_ns(True).cuda.Stream(non_blocking=True)
        return self._stream

    def _hold_until_done(self, event: Any, views: list) -> None:
        """Keep `views` alive until `event` has happened, which the next call makes sure of."""
        if self._in_flight is not None:
            self._in_flight[0].synchronize()
        self._in_flight = (event, views)

    def _jitted_substep(self, key: tuple) -> Callable:
        if key not in self._jitted:
            (
                at_first_substep,
                at_last_substep,
                at_initial_timestep,
                dtime,
                ndyn_substeps_var,
                second_order_divdamp_factor,
            ) = key
            solve_nonhydro = self.global_granules.solve_nonhydro
            assert solve_nonhydro is not None

            def substep(
                constants: dict | None,
                prognostic: dict,
                diagnostic: dict,
                intermediate: dict,
                prep_adv: dict,
            ) -> tuple[dict, dict, dict, dict]:
                self._traces += 1
                solver = (
                    solve_nonhydro
                    if constants is None
                    else _with_constants(solve_nonhydro, constants)
                )
                new_prognostic, new_diagnostic, new_intermediate, new_prep_adv = solver.time_step(
                    diagnostic_state_nh=_diagnostic_from_dict(diagnostic),
                    prognostic_state=prognostics.PrognosticState(**prognostic),
                    intermediate_state=solve_nonhydro_global.IntermediateState(**intermediate),
                    prep_adv=dycore_states.PrepAdvection(**prep_adv),
                    second_order_divdamp_factor=second_order_divdamp_factor,
                    dtime=dtime,
                    ndyn_substeps_var=ndyn_substeps_var,
                    at_initial_timestep=at_initial_timestep,
                    prepare_fluxes_for_advection=self.config.tracer_advection is not None,
                    at_first_substep=at_first_substep,
                    at_last_substep=at_last_substep,
                )
                return (
                    {k: getattr(new_prognostic, k) for k in _PROGNOSTICS},
                    _diagnostic_to_dict(new_diagnostic),
                    new_intermediate._asdict(),
                    _shallow_dict(new_prep_adv),
                )

            self._jitted[key] = self._jax.jit(substep)
        return self._jitted[key]

    def _jitted_diffusion(self, dtime: float) -> Callable:
        key = ("diffusion", dtime)
        if key not in self._jitted:
            diffusion = self.global_granules.diffusion
            assert diffusion is not None

            def run(constants: dict | None, prognostic: dict) -> dict:
                self._traces += 1
                bound = diffusion if constants is None else _with_constants(diffusion, constants)
                new = bound.run(
                    prognostic_state=prognostics.PrognosticState(**prognostic), dtime=dtime
                )
                return {k: getattr(new, k) for k in _PROGNOSTICS}

            self._jitted[key] = self._jax.jit(run)
        return self._jitted[key]

    def _jitted_tracer_advection(self, dtime: float, even_timestep: bool) -> Callable:
        key = ("tracer_advection", dtime, even_timestep)
        if key not in self._jitted:
            advection = self.global_granules.tracer_advection
            assert advection is not None

            def run(
                constants: dict | None,
                rho_now: gtx.Field,
                rho_new: gtx.Field,
                prep_adv: dict,
                tracers: dict,
            ) -> dict:
                self._traces += 1
                bound = advection if constants is None else _with_constants(advection, constants)
                tracer_prep_adv = tracer_prep_adv_states.TracerPrepAdvState(
                    **{name: prep_adv[field] for name, field in _TRACER_PREP_ADV.items()}
                )
                airmass_now = bound.air_mass(rho_now)
                airmass_new = bound.air_mass(rho_new)
                return {
                    name: bound.run(
                        p_tracer_now=tracer,
                        airmass_now=airmass_now,
                        airmass_new=airmass_new,
                        prep_adv=tracer_prep_adv,
                        dtime=dtime,
                        even_timestep=even_timestep,
                    )[0]
                    for name, tracer in tracers.items()
                }

            self._jitted[key] = self._jax.jit(run)
        return self._jitted[key]

    def _jitted_physics(self, dtime: float) -> Callable:
        key = ("physics", dtime)
        if key not in self._jitted:
            physics = self.global_granules.physics
            assert physics is not None

            def run(
                constants: dict | None, prognostic: dict, tracers: dict, perturbed_exner: gtx.Field
            ) -> tuple:
                self._traces += 1
                bound = (
                    physics if constants is None else _with_physics_constants(physics, constants)
                )
                return bound.run(
                    prognostic=prognostic,
                    tracers=tracers,
                    perturbed_exner=perturbed_exner,
                    dtime=dtime,
                )

            self._jitted[key] = self._jax.jit(run)
        return self._jitted[key]

    def _time_step(  # noqa: PLR0917 [too-many-positional-arguments]  # positional through jax.jit
        self,
        constants: dict[str, Any],
        prognostic: dict,
        diagnostic: dict,
        intermediate: dict,
        prep_adv: dict,
        tracers: dict,
        *,
        key: tuple,
    ) -> tuple[dict, dict, dict, dict, dict]:
        (
            at_initial_timestep,
            substep_dtime,
            ndyn_substeps_var,
            second_order_divdamp_factor,
            dtime,
            even_timestep,
        ) = key
        rho_now = prognostic["rho"]
        if self._layout is not None:
            self._traced_tables = constants["halo_tables"]
        diagnostic_state = _diagnostic_from_dict(diagnostic)
        for dyn_substep in range(ndyn_substeps_var):
            if not self.jit_time_step:
                self._compute_statistics(dyn_substep, prognostics.PrognosticState(**prognostic))
            at_first_substep = self._is_first_substep(dyn_substep)
            at_last_substep = dyn_substep == ndyn_substeps_var - 1
            self._update_time_levels_for_velocity_tendencies(
                diagnostic_state,
                at_first_substep=at_first_substep,
                at_initial_timestep=at_initial_timestep,
            )
            substep = self._jitted_substep(
                (
                    at_first_substep,
                    at_last_substep,
                    at_initial_timestep,
                    substep_dtime,
                    ndyn_substeps_var,
                    second_order_divdamp_factor,
                )
            )
            state = self._with_halo(
                dict(
                    zip(
                        _SUBSTEP_GROUPS,
                        substep(
                            constants["solve_nonhydro"],
                            prognostic,
                            _diagnostic_to_dict(diagnostic_state),
                            intermediate,
                            prep_adv,
                        ),
                    )
                ),
                reads=self._reads(
                    _DYCORE_HALO_READS
                    | (
                        _DIFFUSION_HALO_READS
                        | (_TRACER_ADVECTION_HALO_READS if tracers else frozenset())
                        if at_last_substep
                        else frozenset()
                    )
                ),
            )
            prognostic, diagnostic, intermediate, prep_adv = (
                state[group] for group in _SUBSTEP_GROUPS
            )
            diagnostic_state = _diagnostic_from_dict(diagnostic)

        if (
            self.global_granules.diffusion is not None
            and self.global_granules.diffusion.config.apply_to_horizontal_wind
        ):
            prognostic = self._with_halo(
                {"prognostic": self._jitted_diffusion(dtime)(constants["diffusion"], prognostic)},
                reads=self._reads(_DYCORE_HALO_READS),
            )["prognostic"]
        if tracers:
            tracers = self._with_halo(
                {
                    "tracers": self._jitted_tracer_advection(dtime, even_timestep)(
                        constants["tracer_advection"],
                        rho_now,
                        prognostic["rho"],
                        prep_adv,
                        tracers,
                    )
                }
            )["tracers"]
        if self.global_granules.physics is not None:
            prognostic, tracers, perturbed_exner = self._jitted_physics(dtime)(
                constants["physics"],
                prognostic,
                tracers,
                diagnostic_state.perturbed_exner_at_cells_on_model_levels,
            )
            # the physics computed the halo from a halo it read beyond, and the next step reads it
            state = self._with_halo(
                {
                    "prognostic": prognostic,
                    "tracers": tracers,
                    "diagnostic": {"perturbed_exner_at_cells_on_model_levels": perturbed_exner},
                },
                reads=self._reads(
                    _DYCORE_HALO_READS | frozenset(f"tracers.{name}" for name in tracers)
                ),
            )
            prognostic, tracers = state["prognostic"], state["tracers"]
            diagnostic_state = dataclasses.replace(
                diagnostic_state,
                perturbed_exner_at_cells_on_model_levels=state["diagnostic"][
                    "perturbed_exner_at_cells_on_model_levels"
                ],
            )
        return prognostic, _diagnostic_to_dict(diagnostic_state), intermediate, prep_adv, tracers

    def _jitted_time_step(self, key: tuple) -> Callable:
        if ("time_step", *key) not in self._jitted:
            closed_over = self._constants["solve_nonhydro"] is None
            self._jitted[("time_step", *key)] = self._jax.jit(
                functools.partial(self._time_step, key=key),
                # folding the operations on closed-over static fields across a whole time step
                # takes XLA minutes
                compiler_options=(
                    {"xla_disable_hlo_passes": "constant_folding"} if closed_over else None
                ),
            )
        return self._jitted[("time_step", *key)]

    def _jitted_spmd_time_step(self, key: tuple, args: tuple) -> tuple[Callable, Any]:
        """
        The jitted time step as one SPMD program over all ranks, on the flattened `args`.

        Returns the program and the tree structure of the state it returns.
        """
        jax = self._jax
        leaves, treedef = jax.tree_util.tree_flatten(args)
        state_treedef = jax.tree_util.tree_structure(args[1:])
        specs = [
            jax.sharding.PartitionSpec()
            if leaf.ndim == 0
            else jax.sharding.PartitionSpec(_MESH_AXIS)
            for leaf in leaves
        ]
        num_constants = len(jax.tree_util.tree_leaves(args[0]))
        if ("spmd_time_step", *key) not in self._jitted:

            def body(*leaves: Any) -> list:
                constants, *state = jax.tree_util.tree_unflatten(treedef, leaves)
                # The connectivities differ between the ranks, so domain inference must not read
                # a table: without the handle to one it takes every neighbor as present, which
                # the padded layout makes true.
                constants = jax.tree_util.tree_map(
                    lambda c: (
                        type(c)(c.domain, c.ndarray, c.codomain, c.skip_value, None)  # type: ignore[call-arg]  # the JAX connectivity has a table handle
                        if isinstance(c, gtx_common.Connectivity)
                        else c
                    ),
                    constants,
                    is_leaf=lambda x: isinstance(x, gtx_common.Connectivity),
                )
                prognostic, diagnostic, intermediate, prep_adv, tracers = self._time_step(
                    constants, *state, key=key
                )
                diagnostic["max_vertical_cfl"] = jax.lax.pmax(
                    diagnostic["max_vertical_cfl"], _MESH_AXIS
                )
                return jax.tree_util.tree_leaves(
                    (prognostic, diagnostic, intermediate, prep_adv, tracers)
                )

            self._jitted[("spmd_time_step", *key)] = jax.jit(
                jax.shard_map(
                    body,
                    mesh=self._mesh,
                    in_specs=tuple(specs),
                    out_specs=specs[num_constants:],
                    # gt4py's embedded scan starts its carry from a value that does not vary
                    # between the ranks, while its body's output does
                    check_vma=False,
                )
            )
        return self._jitted[("spmd_time_step", *key)], state_treedef

    def _to_global(self, tree: Any) -> Any:
        """Process-local arrays to arrays over all ranks: split along the first axis, 0-d replicated."""
        jax = self._jax
        sharding = jax.sharding.NamedSharding(self._mesh, jax.sharding.PartitionSpec(_MESH_AXIS))
        replicated = jax.sharding.NamedSharding(self._mesh, jax.sharding.PartitionSpec())

        assert self._layout is not None
        num_ranks = self._layout.num_ranks

        def to_global(x: Any) -> Any:
            x = self._xp.asarray(x)
            if x.ndim == 0:
                return jax.make_array_from_process_local_data(replicated, x, ())
            return jax.make_array_from_process_local_data(
                sharding, x, (num_ranks * x.shape[0], *x.shape[1:])
            )

        return jax.tree_util.tree_map(to_global, tree)

    def _local(self, tree: Any) -> Any:
        """This rank's part of `tree`: its shard of the arrays over all ranks."""
        if self._layout is None:
            return tree
        return self._jax.tree_util.tree_map(lambda x: x.addressable_shards[0].data, tree)

    def _prepare(self, ds: driver_states.DriverStates) -> tuple[dict, dict, dict, dict, dict, dict]:
        """The state of `ds` and the constants as the jitted steps take them."""
        assert ds.solve_nonhydro_diagnostic is not None
        assert ds.prep_advection_prognostic is not None
        solve_nonhydro = self.global_granules.solve_nonhydro
        assert solve_nonhydro is not None
        state: dict[str, Any] = {
            "prognostic": {k: getattr(ds.prognostics.current, k) for k in _PROGNOSTICS},
            "diagnostic": ds.solve_nonhydro_diagnostic,
            "prep_adv": ds.prep_advection_prognostic,
            "tracers": {name: getattr(ds.tracers.current, name) for name in self._tracer_names},
        }
        if self._layout is not None:
            state = driver_utils.pad_fields(state, self._layout)
        state = {
            "prognostic": {k: jax_utils.to_jax(v) for k, v in state["prognostic"].items()},
            "diagnostic": _diagnostic_to_dict(jax_utils.to_jax(state["diagnostic"])),
            "prep_adv": _shallow_dict(jax_utils.to_jax(state["prep_adv"])),
            "tracers": {k: jax_utils.to_jax(v) for k, v in state["tracers"].items()},
        }
        intermediate = solve_nonhydro.initial_intermediate_state()._asdict()
        constants = self._constants
        if self._layout is not None:
            self._mesh = jax_utils.rank_mesh(_MESH_AXIS)
            # the halos of the initial state are the ones the setup computed
            state, intermediate = self._to_global((state, intermediate))
            coloured = self.spmd_transport == driver_utils.SpmdTransport.PPERMUTE
            layout = self._layout
            constants = self._to_global(
                {
                    **self._constants,
                    "halo_tables": {
                        dim.value: {
                            "send": (layout.send_round if coloured else layout.send_index)[dim],
                            "recv": (layout.recv_round if coloured else layout.recv_index)[dim],
                        }
                        for dim in layout.padded_local
                    },
                }
            )
        else:
            state = self._with_halo(state)
        return (
            state["prognostic"],
            state["diagnostic"],
            intermediate,
            state["prep_adv"],
            state["tracers"],
            constants,
        )

    def time_integration(self, ds: driver_states.DriverStates) -> None:
        assert self.config.nonhydrostatic is not None
        time_vars = self.model_time_variables

        prognostic, diagnostic, intermediate, prep_adv, tracers, constants = self._prepare(ds)
        diagnostic_state = _diagnostic_from_dict(diagnostic)

        wall_clock_starting_time = datetime.datetime.now()
        for time_step in range(time_vars.n_time_steps):
            log.info(
                f"simulation date : {time_vars.simulation_current_datetime}, at timestep : {time_step}, "
                f"elapsed wall clock time: {(datetime.datetime.now() - wall_clock_starting_time).total_seconds()}"
            )
            time_vars.advance_simulation_datetime()
            second_order_divdamp_factor = float(self._second_order_divdamp_factor())

            key = (
                time_vars.is_first_step_in_simulation,
                float(time_vars.substep_timestep),
                time_vars.ndyn_substeps_var,
                second_order_divdamp_factor,
                float(time_vars.dtime_in_seconds),
                self._even_timestep,
            )
            args = (
                constants,
                prognostic,
                _diagnostic_to_dict(diagnostic_state),
                intermediate,
                prep_adv,
                tracers,
            )
            if self._layout is not None:
                spmd_step, state_treedef = self._jitted_spmd_time_step(key, args)
                prognostic, diagnostic, intermediate, prep_adv, tracers = (
                    self._jax.tree_util.tree_unflatten(
                        state_treedef, spmd_step(*self._jax.tree_util.tree_leaves(args))
                    )
                )
            else:
                if self.jit_time_step:
                    self._compute_statistics(0, prognostics.PrognosticState(**prognostic))
                    step = self._jitted_time_step(key)
                else:
                    step = functools.partial(self._time_step, key=key)
                prognostic, diagnostic, intermediate, prep_adv, tracers = step(*args)
            diagnostic_state = _diagnostic_from_dict(diagnostic)
            self._jax.block_until_ready(prognostic["vn"].ndarray)

            time_vars.is_first_step_in_simulation = False
            if tracers:
                self._even_timestep = not self._even_timestep
            self._adjust_ndyn_substeps_var(diagnostic_state)
            if self.on_step_end is not None:
                self.on_step_end(time_step, prognostics.PrognosticState(**self._local(prognostic)))

        ds.prognostics.first = prognostics.PrognosticState(**self._local(prognostic))  # type: ignore[method-assign]  # Pair.first is a named_property with a setter
        ds.tracers.first = tracer_states.TracerState(**self._local(tracers))  # type: ignore[method-assign]
        log.info(
            f"JAX time loop: {time_vars.n_time_steps} steps in "
            f"{(datetime.datetime.now() - wall_clock_starting_time).total_seconds():.1f} s, "
            f"{len(self._jitted)} jitted functions, {self._traces} traces"
        )
