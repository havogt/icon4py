# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""
The JAX driver's time step as pure functions of pytrees, on a single rank.

`IconStatic` holds the static fields as pytree leaves and the granules as metadata, so under
jax.jit the fields are arguments and never constants of the program. `IconState` is the state
one step reads and returns. A time step is `dynamics_step`, the fixed number of dynamical
substeps, diffusion and tracer advection, then `physics_step`; the physics is also open per
process group, on `ColumnState` and `ColumnTendencies`:

    col = dyn2phy(static, state)
    tend_muphys, _ = muphys(static, col)
    col = apply_column_tendencies(col, tend_muphys, static.dtime)
    fluxes = surface(static, surface_input(static, col))
    tend_tmx, _ = tmx(static, col, fluxes)
    col = apply_column_tendencies(col, tend_tmx, static.dtime)
    state = phy2dyn(static, state, col, sum_column_tendencies(tend_muphys, tend_tmx))

which is `physics_step`.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

from icon4py.model.atmosphere.dycore import dycore_states, solve_nonhydro_global
from icon4py.model.atmosphere.subgrid_scale_physics.surface import surface_states
from icon4py.model.common import model_backends
from icon4py.model.common.decomposition import definitions as decomposition_defs
from icon4py.model.common.metrics import metrics_attributes
from icon4py.model.common.states import (
    prognostic_state as prognostics,
    surface_flux_state,
    tracer_prep_adv_states,
)
from icon4py.model.driver import (
    config as driver_config,
    driver,
    driver_utils,
    jax_driver,
    jax_utils,
    physics_global,
)
from icon4py.model.driver.physics_global import (
    ColumnState,
    ColumnTendencies,
    apply_column_tendencies,
    sum_column_tendencies,
)


__all__ = [
    "ColumnState",
    "ColumnTendencies",
    "IconState",
    "IconStatic",
    "apply_column_tendencies",
    "dyn2phy",
    "dynamics_step",
    "init_from_grid_file",
    "muphys",
    "phy2dyn",
    "physics_step",
    "sum_column_tendencies",
    "surface",
    "surface_input",
    "time_step",
    "tmx",
]

_jax = jax_utils.import_jax()


class _Setup:
    """The granules and the configuration: pytree metadata, equal only to itself."""

    def __init__(
        self, granules: driver_utils.GlobalGranules, config: driver_config.ExperimentConfig
    ) -> None:
        self.granules = granules
        self.config = config


@dataclasses.dataclass(frozen=True)
class IconStatic:
    """
    The static fields of the granules, by granule, and what a time step is.

    `z_mc` and `z_ifc` are the heights of the cell centres on the full and the half levels [m].
    """

    constants: dict[str, Any]
    z_mc: Any
    z_ifc: Any
    setup: _Setup
    dtime: float
    substep_dtime: float
    ndyn_substeps: int
    second_order_divdamp_factor: float


@dataclasses.dataclass(frozen=True)
class IconState:
    """
    The prognostics, the tracers and what the next step reads of this one.

    `carry` holds the dynamical core's diagnostic, intermediate and advection states. Its
    `carry["diagnostic"]["max_vertical_cfl"]` is the maximum vertical CFL number of the step that
    returned the state, for the caller to check against `vertical_cfl_threshold`: the number of
    substeps is fixed.
    `at_initial_timestep` and `even_timestep`, the tracers' splitting order, are metadata: a
    jitted step is traced once per value.
    """

    prognostic: dict[str, Any]
    tracers: dict[str, Any]
    carry: dict[str, dict[str, Any]]
    at_initial_timestep: bool
    even_timestep: bool


_jax.tree_util.register_dataclass(
    IconStatic,
    data_fields=["constants", "z_mc", "z_ifc"],
    meta_fields=["setup", "dtime", "substep_dtime", "ndyn_substeps", "second_order_divdamp_factor"],
)
_jax.tree_util.register_dataclass(
    IconState,
    data_fields=["prognostic", "tracers", "carry"],
    meta_fields=["at_initial_timestep", "even_timestep"],
)
for _cls in (
    ColumnState,
    ColumnTendencies,
    surface_states.SurfaceInputState,
    surface_flux_state.SurfaceFluxState,
):
    _jax.tree_util.register_dataclass(_cls)


def _bound(static: IconStatic, name: str) -> Any:
    granule = getattr(static.setup.granules, name)
    return jax_driver._with_constants(granule, static.constants[name])


def _physics(static: IconStatic) -> physics_global.PhysicsGlobal:
    physics = static.setup.granules.physics
    if physics is None:
        raise ValueError("The experiment configures no physics.")
    return jax_driver._with_physics_constants(physics, static.constants["physics"])


def init_from_grid_file(
    grid_file: pathlib.Path, config: driver_config.ExperimentConfig
) -> tuple[IconStatic, IconState]:
    """The static fields and the initial state of the experiment `config` on one rank."""
    assert config.nonhydrostatic is not None
    if (
        config.nonhydrostatic.divdamp_order == dycore_states.DivergenceDampingOrder.COMBINED
        and config.driver.apply_extra_second_order_divdamp
    ):
        raise NotImplementedError(
            "The second-order divergence damping of the spin-up changes from step to step."
        )
    process_props = decomposition_defs.SingleNodeProcessProperties()
    allocator = model_backends.get_allocator(None)
    grid_manager = driver_utils.create_grid_manager(
        grid_file_path=grid_file,
        vertical_grid_config=config.vertical_grid,
        allocator=allocator,
        process_props=process_props,
    )
    icon_driver = driver.initialize_driver(
        config=config,
        grid_manager=grid_manager,
        process_props=process_props,
        backend=None,
        jax=True,
        jax_jit_time_step=True,
    )
    assert isinstance(icon_driver, jax_driver.JaxIcon4pyDriver)
    ds = driver.initialize_driver_states(icon_driver, allocator, validate_granules=False)
    prognostic, diagnostic, intermediate, prep_adv, tracers, constants = icon_driver._prepare(ds)
    granules = icon_driver.global_granules
    time_vars = icon_driver.model_time_variables
    metrics = icon_driver.static_field_factories.metrics
    static = IconStatic(
        constants=constants,
        z_mc=jax_utils.to_jax(metrics.get(metrics_attributes.Z_MC)),
        z_ifc=jax_utils.to_jax(metrics.get(metrics_attributes.CELL_HEIGHT_ON_HALF_LEVEL)),
        setup=_Setup(granules, icon_driver.config),
        dtime=float(time_vars.dtime_in_seconds),
        substep_dtime=float(time_vars.substep_timestep),
        ndyn_substeps=int(time_vars.ndyn_substeps_var),
        second_order_divdamp_factor=float(icon_driver._second_order_divdamp_factor()),
    )
    state = IconState(
        prognostic=prognostic,
        tracers=tracers,
        carry={"diagnostic": diagnostic, "intermediate": intermediate, "prep_adv": prep_adv},
        at_initial_timestep=bool(time_vars.is_first_step_in_simulation),
        even_timestep=False,
    )
    return static, state


def dynamics_step(static: IconStatic, state: IconState) -> IconState:
    """
    The dynamical substeps, diffusion and tracer advection of one time step.

    `ndyn_substeps` is fixed: where the driver adds substeps for a vertical CFL number above
    `vertical_cfl_threshold`, the returned state's `max_vertical_cfl` is above it.
    """
    granules = static.setup.granules
    config = static.setup.config
    solver = _bound(static, "solve_nonhydro")
    prognostic = state.prognostic
    tracers = state.tracers
    diagnostic = jax_driver._diagnostic_from_dict(state.carry["diagnostic"])
    # the substeps take the maximum with it: it starts from zero each step, as after
    # `Icon4pyDriver._adjust_ndyn_substeps_var`
    diagnostic = dataclasses.replace(
        diagnostic, max_vertical_cfl=_jax.numpy.zeros_like(diagnostic.max_vertical_cfl)
    )
    intermediate = state.carry["intermediate"]
    prep_adv = state.carry["prep_adv"]
    rho_now = prognostic["rho"]
    num_substeps = static.ndyn_substeps
    for substep in range(num_substeps):
        at_first_substep = substep == 0
        # `Icon4pyDriver._update_time_levels_for_velocity_tendencies`
        if not (state.at_initial_timestep and at_first_substep):
            diagnostic.vertical_wind_advective_tendency.swap()
        if not at_first_substep:
            diagnostic.normal_wind_advective_tendency.swap()
        new_prognostic, diagnostic, new_intermediate, new_prep_adv = solver.time_step(
            diagnostic_state_nh=diagnostic,
            prognostic_state=prognostics.PrognosticState(**prognostic),
            intermediate_state=solve_nonhydro_global.IntermediateState(**intermediate),
            prep_adv=dycore_states.PrepAdvection(**prep_adv),
            second_order_divdamp_factor=static.second_order_divdamp_factor,
            dtime=static.substep_dtime,
            ndyn_substeps_var=num_substeps,
            at_initial_timestep=state.at_initial_timestep,
            prepare_fluxes_for_advection=config.tracer_advection is not None,
            at_first_substep=at_first_substep,
            at_last_substep=substep == num_substeps - 1,
        )
        prognostic = {k: getattr(new_prognostic, k) for k in jax_driver._PROGNOSTICS}
        intermediate = new_intermediate._asdict()
        prep_adv = jax_driver._shallow_dict(new_prep_adv)

    if granules.diffusion is not None and granules.diffusion.config.apply_to_horizontal_wind:
        new = _bound(static, "diffusion").run(
            prognostic_state=prognostics.PrognosticState(**prognostic), dtime=static.dtime
        )
        prognostic = {k: getattr(new, k) for k in jax_driver._PROGNOSTICS}
    if tracers:
        advection = _bound(static, "tracer_advection")
        tracer_prep_adv = tracer_prep_adv_states.TracerPrepAdvState(
            **{name: prep_adv[field] for name, field in jax_driver._TRACER_PREP_ADV.items()}
        )
        airmass_now = advection.air_mass(rho_now)
        airmass_new = advection.air_mass(prognostic["rho"])
        tracers = {
            name: advection.run(
                p_tracer_now=tracer,
                airmass_now=airmass_now,
                airmass_new=airmass_new,
                prep_adv=tracer_prep_adv,
                dtime=static.dtime,
                even_timestep=state.even_timestep,
            )[0]
            for name, tracer in tracers.items()
        }
    return IconState(
        prognostic=prognostic,
        tracers=tracers,
        carry={
            "diagnostic": jax_driver._diagnostic_to_dict(diagnostic),
            "intermediate": intermediate,
            "prep_adv": prep_adv,
        },
        at_initial_timestep=False,
        even_timestep=not state.even_timestep if tracers else state.even_timestep,
    )


def dyn2phy(static: IconStatic, state: IconState) -> ColumnState:
    return _physics(static).dyn2phy(state.prognostic, state.tracers)


def muphys(static: IconStatic, col: ColumnState) -> tuple[ColumnTendencies, dict[str, Any]]:
    """The tendencies of temperature and moisture, and the precipitation fluxes."""
    return _physics(static).run_muphys(col, static.dtime)


def surface_input(static: IconStatic, col: ColumnState) -> surface_states.SurfaceInputState:
    return _physics(static).surface_input(col)


def surface(
    static: IconStatic, input_state: surface_states.SurfaceInputState
) -> surface_flux_state.SurfaceFluxState:
    return _physics(static).run_surface(input_state)


def tmx(
    static: IconStatic, col: ColumnState, fluxes: surface_flux_state.SurfaceFluxState
) -> tuple[ColumnTendencies, dict[str, Any]]:
    """The tendencies of temperature, qv, qc, qi, u, v and w, and the tmx diagnostics."""
    return _physics(static).run_tmx(col, fluxes, static.dtime)


def phy2dyn(
    static: IconStatic, state: IconState, col: ColumnState, tend: ColumnTendencies
) -> IconState:
    """`state` with the advanced column `col` and the wind tendencies of `tend`, at constant volume."""
    diagnostic = state.carry["diagnostic"]
    prognostic, tracers, perturbed_exner = _physics(static).phy2dyn(
        state.prognostic,
        diagnostic["perturbed_exner_at_cells_on_model_levels"],
        col,
        tend,
        static.dtime,
    )
    return dataclasses.replace(
        state,
        prognostic=prognostic,
        tracers=tracers,
        carry={
            **state.carry,
            "diagnostic": {
                **diagnostic,
                "perturbed_exner_at_cells_on_model_levels": perturbed_exner,
            },
        },
    )


def physics_step(static: IconStatic, state: IconState) -> IconState:
    """The physics of one time step, `physics_global.PhysicsGlobal.run`."""
    diagnostic = state.carry["diagnostic"]
    prognostic, tracers, perturbed_exner = _physics(static).run(
        prognostic=state.prognostic,
        tracers=state.tracers,
        perturbed_exner=diagnostic["perturbed_exner_at_cells_on_model_levels"],
        dtime=static.dtime,
    )
    return dataclasses.replace(
        state,
        prognostic=prognostic,
        tracers=tracers,
        carry={
            **state.carry,
            "diagnostic": {
                **diagnostic,
                "perturbed_exner_at_cells_on_model_levels": perturbed_exner,
            },
        },
    )


def time_step(static: IconStatic, state: IconState) -> IconState:
    """One time step of the JAX driver: `dynamics_step`, then `physics_step` with physics."""
    state = dynamics_step(static, state)
    if static.setup.granules.physics is not None and static.setup.granules.physics.has_processes:
        state = physics_step(static, state)
    return state
