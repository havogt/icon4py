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
host or, from CuPy buffers, on the device (`driver_utils.HaloExchange`). The halo must be deep
enough for one step: `driver_utils.JAX_EXTRA_HALO_RINGS`.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime
import logging
from collections.abc import Callable
from typing import Any

import gt4py.next as gtx
from gt4py.next import common as gtx_common

import icon4py.model.common.utils as common_utils
from icon4py.model.atmosphere.dycore import dycore_states, solve_nonhydro_global
from icon4py.model.common import dimension as dims
from icon4py.model.common.decomposition import definitions as decomposition_defs
from icon4py.model.common.states import nonhydro_states, prognostic_state as prognostics
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.driver import driver, driver_states, driver_utils, jax_utils


log = logging.getLogger(__name__)

_PROGNOSTICS = ("rho", "w", "vn", "exner", "theta_v")
_TENDENCY_PAIRS = ("normal_wind_advective_tendency", "vertical_wind_advective_tendency")


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


class JaxIcon4pyDriver(driver.Icon4pyDriver):
    def __init__(
        self,
        *,
        global_granules: driver_utils.GlobalGranules,
        halo_exchange: driver_utils.HaloExchange = driver_utils.HaloExchange.HOST,
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
        self._stream: Any = None
        self._in_flight: tuple[Any, list] | None = None
        self.on_step_end: Callable[[int, prognostics.PrognosticState], None] | None = None
        self._jax = jax_utils.import_jax()
        self._jitted: dict[tuple, Callable] = {}

    def _with_halo(self, tree: Any) -> Any:
        """
        New local fields from the owned points of the fields in `tree`, with the halo exchanged.

        Works on owned-size and local-size fields alike; anything else is passed through.
        """
        if not self._distributed:
            return tree
        jax = self._jax
        leaves, treedef = jax.tree_util.tree_flatten(
            tree, is_leaf=lambda x: isinstance(x, gtx.Field)
        )
        mode = self.halo_exchange
        on_device = mode != driver_utils.HaloExchange.HOST
        xp = data_alloc.array_ns(on_device)
        stream = self._device_stream() if mode == driver_utils.HaloExchange.DEVICE_STREAM else None
        with stream if stream is not None else contextlib.nullcontext():
            buffers: dict[int, tuple[gtx_common.Domain, Any]] = {}
            # views of the JAX arrays the device copies read from; they must outlive the copies
            views = []
            for i, leaf in enumerate(leaves):
                if isinstance(leaf, gtx.Field) and leaf.domain.dims[0] in self._num_owned:
                    dim = leaf.domain.dims[0]
                    num_owned, num_local = self._num_owned[dim], self.grid.size[dim]
                    # NaN marks every halo point the exchange leaves unfilled
                    buffer = xp.full(
                        (num_local, *leaf.ndarray.shape[1:]), xp.nan, dtype=leaf.dtype.scalar_type
                    )
                    if on_device:
                        views.append(xp.from_dlpack(leaf.ndarray[:num_owned]))
                        buffer[:num_owned] = views[-1]
                    else:
                        buffer[:num_owned] = xp.asarray(leaf.ndarray[:num_owned])
                    domain = leaf.domain.replace(
                        dim, gtx_common.NamedRange(dim, gtx_common.unit_range((0, num_local)))
                    )
                    buffers[i] = (domain, buffer)
            if stream is not None:
                self._hold_until_done(stream.record(), views)
            elif on_device:
                # GHEX starts after the default stream only; the copies are on CuPy's current stream
                xp.cuda.get_current_stream().synchronize()
            exchange_stream = {"stream": stream if stream is not None else decomposition_defs.BLOCK}
            for dim in self._num_owned:
                dim_buffers = [b for d, b in buffers.values() if d.dims[0] == dim]
                if dim_buffers:
                    self.exchange.exchange(
                        dim, *dim_buffers, **(exchange_stream if on_device else {})
                    )
            if on_device and stream is None:
                # GHEX unpacks on its own streams; JAX must not see a buffer before they are done
                xp.cuda.runtime.deviceSynchronize()
            # with `stream` current, DLPack makes JAX wait for the exchange on `stream`
            to_jax = jax.numpy.from_dlpack if on_device else jax.numpy.asarray
            new_leaves = [
                gtx.as_field(buffers[i][0], to_jax(buffers[i][1]), allocator=jax.numpy)
                if i in buffers
                else leaf
                for i, leaf in enumerate(leaves)
            ]
        return jax.tree_util.tree_unflatten(treedef, new_leaves)

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
                prognostic: dict, diagnostic: dict, intermediate: dict, prep_adv: dict
            ) -> tuple[dict, dict, dict, dict]:
                new_prognostic, new_diagnostic, new_intermediate, new_prep_adv = (
                    solve_nonhydro.time_step(
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

            def run(prognostic: dict) -> dict:
                new = diffusion.run(
                    prognostic_state=prognostics.PrognosticState(**prognostic), dtime=dtime
                )
                return {k: getattr(new, k) for k in _PROGNOSTICS}

            self._jitted[key] = self._jax.jit(run)
        return self._jitted[key]

    def time_integration(self, ds: driver_states.DriverStates) -> None:
        assert self.config.nonhydrostatic is not None
        assert ds.solve_nonhydro_diagnostic is not None
        assert ds.prep_advection_prognostic is not None
        solve_nonhydro = self.global_granules.solve_nonhydro
        assert solve_nonhydro is not None
        time_vars = self.model_time_variables

        prognostic, diagnostic, prep_adv = self._with_halo(
            (
                {k: jax_utils.to_jax(getattr(ds.prognostics.current, k)) for k in _PROGNOSTICS},
                _diagnostic_to_dict(jax_utils.to_jax(ds.solve_nonhydro_diagnostic)),
                _shallow_dict(jax_utils.to_jax(ds.prep_advection_prognostic)),
            )
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

            for dyn_substep in range(time_vars.ndyn_substeps_var):
                self._compute_statistics(dyn_substep, prognostics.PrognosticState(**prognostic))
                at_first_substep = self._is_first_substep(dyn_substep)
                self._update_time_levels_for_velocity_tendencies(
                    diagnostic_state,
                    at_first_substep=at_first_substep,
                    at_initial_timestep=time_vars.is_first_step_in_simulation,
                )
                substep = self._jitted_substep(
                    (
                        at_first_substep,
                        self._is_last_substep(dyn_substep),
                        time_vars.is_first_step_in_simulation,
                        float(time_vars.substep_timestep),
                        time_vars.ndyn_substeps_var,
                        second_order_divdamp_factor,
                    )
                )
                prognostic, diagnostic, intermediate, prep_adv = self._with_halo(
                    substep(
                        prognostic, _diagnostic_to_dict(diagnostic_state), intermediate, prep_adv
                    )
                )
                diagnostic_state = _diagnostic_from_dict(diagnostic)

            if (
                self.global_granules.diffusion is not None
                and self.global_granules.diffusion.config.apply_to_horizontal_wind
            ):
                prognostic = self._with_halo(
                    self._jitted_diffusion(float(time_vars.dtime_in_seconds))(prognostic)
                )
            self._jax.block_until_ready(prognostic["vn"].ndarray)

            time_vars.is_first_step_in_simulation = False
            self._adjust_ndyn_substeps_var(diagnostic_state)
            if self.on_step_end is not None:
                self.on_step_end(time_step, prognostics.PrognosticState(**prognostic))

        ds.prognostics.first = prognostics.PrognosticState(**prognostic)  # type: ignore[method-assign]  # Pair.first is a named_property with a setter
        log.info(
            f"JAX time loop: {time_vars.n_time_steps} steps in "
            f"{(datetime.datetime.now() - wall_clock_starting_time).total_seconds():.1f} s, "
            f"{len(self._jitted)} compiled step variants"
        )
