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
threaded through the stateless `SolveNonhydroGlobal` and `DiffusionGlobal` steps. No tracer
transport, no physics, no output.

On a distributed grid the steps return their outputs on the owned points only. Between two jitted
steps the driver builds new local fields from them, with the halo filled by a halo exchange on the
host or, from CuPy buffers, on the device (`driver_utils.HaloExchange`); with `jit_time_step` the
whole time step is one jitted function, and the exchange a `buffer_callback` inside it, on XLA's
stream on a GPU, whatever `halo_exchange` says. The halo must be deep enough for one step:
`driver_utils.JAX_EXTRA_HALO_RINGS`.
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
from gt4py.next import common as gtx_common

import icon4py.model.common.utils as common_utils
from icon4py.model.atmosphere.dycore import dycore_states, solve_nonhydro_global
from icon4py.model.common import dimension as dims
from icon4py.model.common.decomposition import definitions as decomposition_defs
from icon4py.model.common.grid import base as grid_base
from icon4py.model.common.states import nonhydro_states, prognostic_state as prognostics
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.driver import driver, driver_states, driver_utils, jax_utils


log = logging.getLogger(__name__)

_PROGNOSTICS = ("rho", "w", "vn", "exner", "theta_v")
_TENDENCY_PAIRS = ("normal_wind_advective_tendency", "vertical_wind_advective_tendency")
_SUBSTEP_GROUPS = ("prognostic", "diagnostic", "intermediate", "prep_adv")

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


class JaxIcon4pyDriver(driver.Icon4pyDriver):
    def __init__(
        self,
        *,
        global_granules: driver_utils.GlobalGranules,
        halo_exchange: driver_utils.HaloExchange = driver_utils.HaloExchange.HOST,
        exchange_read_fields_only: bool = False,
        constants_as_arguments: bool = True,
        jit_time_step: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._distributed = not kwargs["process_props"].is_single_rank()
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
        if self.io_monitor is not None or self.tendencies is not None:
            raise NotImplementedError("The JAX driver does not write output or apply tendencies.")
        if self.config.tracer_config is not None and self.config.tracer_config.nactive > 0:
            raise NotImplementedError("The JAX driver does not transport tracers.")
        self.global_granules = global_granules
        self.halo_exchange = driver_utils.HaloExchange(halo_exchange)
        self.exchange_read_fields_only = exchange_read_fields_only
        # passed as arguments, the static fields are not constants of the jitted programs
        self._constants = {
            name: _constants_of(granule) if constants_as_arguments else None
            for name, granule in (
                ("solve_nonhydro", global_granules.solve_nonhydro),
                ("diffusion", global_granules.diffusion),
            )
        }
        self._traces = 0
        self.jit_time_step = jit_time_step
        if jit_time_step and self._distributed:
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
            dim, gtx_common.NamedRange(dim, gtx_common.unit_range((0, self.grid.size[dim])))
        )

    def _nan_padded(self, field: gtx.Field) -> gtx.Field:
        """An owned-size field extended to the local size with a NaN halo; others as they are."""
        dim = field.domain.dims[0]
        num_owned, num_local = self._num_owned[dim], self.grid.size[dim]
        if field.ndarray.shape[0] != num_owned:
            return field
        jnp = self._jax.numpy
        return gtx.as_field(
            self._local_domain(field),
            jnp.pad(
                field.ndarray,
                [(0, num_local - num_owned)] + [(0, 0)] * (field.ndarray.ndim - 1),
                constant_values=jnp.nan,
            ),
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
        field_dims = tuple(leaves[i].domain.dims[0] for i in exchanged)
        owned = [leaves[i].ndarray[: self._num_owned[dim]] for i, dim in zip(exchanged, field_dims)]
        exchange = buffer_callback.buffer_callback(
            functools.partial(
                self._exchange_in_callback, field_dims, jax.default_backend() == "gpu"
            ),
            [
                jax.ShapeDtypeStruct((self.grid.size[dim], *array.shape[1:]), array.dtype)
                for dim, array in zip(field_dims, owned)
            ],
            # every rank has to issue every exchange
            has_side_effect=True,
        )
        for i, array in zip(exchanged, exchange(*owned)):
            new_leaves[i] = gtx.as_field(self._local_domain(leaves[i]), array, allocator=jax.numpy)
        return new_leaves

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

    def _time_step(
        self,
        constants: dict[str, Any],
        prognostic: dict,
        diagnostic: dict,
        intermediate: dict,
        prep_adv: dict,
        *,
        key: tuple,
    ) -> tuple[dict, dict, dict, dict]:
        (
            at_initial_timestep,
            substep_dtime,
            ndyn_substeps_var,
            second_order_divdamp_factor,
            dtime,
        ) = key
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
                    _DYCORE_HALO_READS | _DIFFUSION_HALO_READS
                    if at_last_substep
                    else _DYCORE_HALO_READS
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
        return prognostic, _diagnostic_to_dict(diagnostic_state), intermediate, prep_adv

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

    def time_integration(self, ds: driver_states.DriverStates) -> None:
        assert self.config.nonhydrostatic is not None
        assert ds.solve_nonhydro_diagnostic is not None
        assert ds.prep_advection_prognostic is not None
        solve_nonhydro = self.global_granules.solve_nonhydro
        assert solve_nonhydro is not None
        time_vars = self.model_time_variables

        state = self._with_halo(
            {
                "prognostic": {
                    k: jax_utils.to_jax(getattr(ds.prognostics.current, k)) for k in _PROGNOSTICS
                },
                "diagnostic": _diagnostic_to_dict(jax_utils.to_jax(ds.solve_nonhydro_diagnostic)),
                "prep_adv": _shallow_dict(jax_utils.to_jax(ds.prep_advection_prognostic)),
            }
        )
        prognostic, diagnostic, prep_adv = (
            state["prognostic"],
            state["diagnostic"],
            state["prep_adv"],
        )
        diagnostic_state = _diagnostic_from_dict(diagnostic)
        intermediate = solve_nonhydro.initial_intermediate_state()._asdict()

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
            )
            if self.jit_time_step:
                self._compute_statistics(0, prognostics.PrognosticState(**prognostic))
                step = self._jitted_time_step(key)
            else:
                step = functools.partial(self._time_step, key=key)
            prognostic, diagnostic, intermediate, prep_adv = step(
                self._constants,
                prognostic,
                _diagnostic_to_dict(diagnostic_state),
                intermediate,
                prep_adv,
            )
            diagnostic_state = _diagnostic_from_dict(diagnostic)
            self._jax.block_until_ready(prognostic["vn"].ndarray)

            time_vars.is_first_step_in_simulation = False
            self._adjust_ndyn_substeps_var(diagnostic_state)
            if self.on_step_end is not None:
                self.on_step_end(time_step, prognostics.PrognosticState(**prognostic))

        ds.prognostics.first = prognostics.PrognosticState(**prognostic)  # type: ignore[method-assign]  # Pair.first is a named_property with a setter
        log.info(
            f"JAX time loop: {time_vars.n_time_steps} steps in "
            f"{(datetime.datetime.now() - wall_clock_starting_time).total_seconds():.1f} s, "
            f"{len(self._jitted)} jitted functions, {self._traces} traces"
        )
