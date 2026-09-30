# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import dataclasses
import os
import types

import numpy as np
import pytest

from icon4py.model.atmosphere.dycore import (
    dycore_states,
    solve_nonhydro as solve_nh,
    solve_nonhydro_global,
)
from icon4py.model.common import dimension as dims, model_backends, model_options
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.common.states import prognostic_state as prognostics
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import definitions as test_defs, test_utils

from .. import utils
from ..fixtures import *  # noqa: F403
from .test_benchmark_solve_nonhydro import (
    _PROGNOSTIC_FIELDS,
    _SOLVE_NONHYDRO_GLOBAL_STATIC_PARAMS,
    _to_jax,
)


_STEP = pytest.mark.parametrize(
    "istep_init, substep_init, istep_exit, substep_exit, at_initial_timestep", [(1, 1, 2, 1, True)]
)
_EXPERIMENT = pytest.mark.parametrize(
    "experiment_description, step_date_init, step_date_exit",
    [
        (
            test_defs.Experiments.EXCLAIM_APE,
            "2000-01-01T00:00:02.000",
            "2000-01-01T00:00:02.000",
        ),
    ],
)


def _solver_inputs(  # noqa: PLR0917 [too-many-positional-arguments]
    experiment, icon_grid, sp, grid_savepoint, metrics_savepoint, interpolation_savepoint, backend
):
    config = experiment.config.nonhydrostatic
    solver_kwargs = dict(
        grid=icon_grid,
        config=config,
        params=solve_nh.NonHydrostaticParams(config),
        metric_state_nonhydro=utils.construct_metric_state(metrics_savepoint, grid_savepoint),
        interpolation_state=utils.construct_interpolation_state(interpolation_savepoint),
        vertical_params=utils.create_vertical_params(
            experiment.config.vertical_grid, grid_savepoint
        ),
        edge_geometry=grid_savepoint.construct_edge_geometry(),
        cell_geometry=grid_savepoint.construct_cell_geometry(),
    )
    prognostic_state = utils.create_prognostic_states(sp).current
    prep_adv = dycore_states.PrepAdvection(
        vn_traj=sp.vn_traj(),
        mass_flx_me=sp.mass_flx_me(),
        dynamical_vertical_mass_flux_at_cells_on_half_levels=sp.mass_flx_ic(),
        dynamical_vertical_volumetric_flux_at_cells_on_half_levels=data_alloc.zero_field(
            icon_grid, dims.CellDim, dims.KHalfDim, allocator=backend
        ),
    )
    diagnostic_state_nh = utils.construct_diagnostics(sp, icon_grid, backend)
    return solver_kwargs, prognostic_state, prep_adv, diagnostic_state_nh


def _step_kwargs(experiment, sp, substep_init, at_initial_timestep):
    return dict(
        second_order_divdamp_factor=sp.divdamp_fac_o2(),
        dtime=sp.get_metadata("dtime").get("dtime"),
        ndyn_substeps_var=experiment.config.driver.ndyn_substeps,
        at_initial_timestep=at_initial_timestep,
        prepare_fluxes_for_advection=sp.get_metadata("prep_adv").get("prep_adv"),
        at_first_substep=substep_init == 1,
        at_last_substep=substep_init == experiment.config.driver.ndyn_substeps,
    )


def _check_against_savepoints(  # noqa: PLR0917 [too-many-positional-arguments]
    sp, prognostic_state, next_state, next_diagnostic_state, exit_sp, final_sp
):
    for name, reference in (
        ("rho", sp.rho_now),
        ("w", sp.w_now),
        ("vn", sp.vn_now),
        ("exner", sp.exner_now),
        ("theta_v", sp.theta_v_now),
    ):
        assert test_utils.dallclose(
            getattr(prognostic_state, name).asnumpy(), reference().asnumpy()
        )
    checks = (
        ("theta_v", next_state.theta_v, final_sp.theta_v_new, {}),
        ("exner", next_state.exner, final_sp.exner_new, {}),
        ("vn", next_state.vn, exit_sp.vn_new, {"rtol": 1e-12, "atol": 1e-13}),
        ("rho", next_state.rho, exit_sp.rho_new, {}),
        ("w", next_state.w, exit_sp.w_new, {"atol": 8e-14}),
        (
            "exner_dynamical_increment",
            next_diagnostic_state.exner_dynamical_increment,
            exit_sp.exner_dyn_incr,
            {"atol": 1e-14},
        ),
    )
    for name, actual, reference, _ in checks:
        diff = np.abs(actual.asnumpy() - reference().asnumpy())
        print(
            f"{name}: max abs diff {diff.max():.3e}, max |ref| {np.abs(reference().asnumpy()).max():.3e}"
        )
    for name, actual, reference, tolerances in checks:
        assert test_utils.dallclose(actual.asnumpy(), reference().asnumpy(), **tolerances), name


@pytest.mark.embedded_only
@pytest.mark.datatest
@_STEP
@_EXPERIMENT
def test_run_solve_nonhydro_global_single_step(  # noqa: PLR0917 [too-many-positional-arguments]
    istep_init,
    substep_init,
    istep_exit,
    substep_exit,
    at_initial_timestep,
    step_date_init,
    step_date_exit,
    experiment,
    icon_grid,
    savepoint_nonhydro_init,
    grid_savepoint,
    metrics_savepoint,
    interpolation_savepoint,
    savepoint_nonhydro_exit,
    savepoint_nonhydro_step_final,
    backend,
):
    sp = savepoint_nonhydro_init
    solver_kwargs, prognostic_state, prep_adv, diagnostic_state_nh = _solver_inputs(
        experiment,
        icon_grid,
        sp,
        grid_savepoint,
        metrics_savepoint,
        interpolation_savepoint,
        backend,
    )
    solver = solve_nonhydro_global.SolveNonhydroGlobal(
        **solver_kwargs, allocator=model_backends.get_allocator(backend), backend=backend
    )
    next_state, next_diagnostic_state, _, _ = solver.time_step(
        diagnostic_state_nh=diagnostic_state_nh,
        prognostic_state=prognostic_state,
        intermediate_state=solver.initial_intermediate_state(),
        prep_adv=prep_adv,
        **_step_kwargs(experiment, sp, substep_init, at_initial_timestep),
    )
    _check_against_savepoints(
        sp,
        prognostic_state,
        next_state,
        next_diagnostic_state,
        savepoint_nonhydro_exit,
        savepoint_nonhydro_step_final,
    )


@pytest.mark.datatest
@_STEP
@_EXPERIMENT
@pytest.mark.parametrize("execution", ["backend", "jax"])
def test_run_solve_nonhydro_global_single_step_fused(  # noqa: PLR0917 [too-many-positional-arguments]
    istep_init,
    substep_init,
    istep_exit,
    substep_exit,
    at_initial_timestep,
    step_date_init,
    step_date_exit,
    experiment,
    icon_grid,
    savepoint_nonhydro_init,
    grid_savepoint,
    metrics_savepoint,
    interpolation_savepoint,
    savepoint_nonhydro_exit,
    savepoint_nonhydro_step_final,
    backend,
    backend_like,
    execution,
):
    """
    The savepoint check of the fused substep as the benchmarks run it: on `--backend` with static
    domains and static integer and boolean arguments (`ICON4PY_BENCH_DACE_DISABLE_SPLITTING=1`
    switches off the dace map splitting), or under `jax.jit` on JAX arrays.
    """
    sp = savepoint_nonhydro_init
    solver_kwargs, prognostic_state, prep_adv, diagnostic_state_nh = _solver_inputs(
        experiment,
        icon_grid,
        sp,
        grid_savepoint,
        metrics_savepoint,
        interpolation_savepoint,
        backend,
    )
    step_kwargs = _step_kwargs(experiment, sp, substep_init, at_initial_timestep)
    if execution == "backend":
        if backend is None:
            pytest.skip("needs --backend")
        backend_descriptor = backend_like
        if os.environ.get("ICON4PY_BENCH_DACE_DISABLE_SPLITTING", "0") == "1":
            assert isinstance(backend_like, dict)
            backend_descriptor = {**backend_like, "optimization_args": {"disable_splitting": True}}
        solver = solve_nonhydro_global.SolveNonhydroGlobal(
            **solver_kwargs,
            allocator=model_backends.get_allocator(backend),
            backend=model_options.customize_backend(None, backend_descriptor),
        )
        solver._solve_nonhydro_global_step = (
            solver._solve_nonhydro_global_step.with_compilation_options(
                static_domains=True, static_params=_SOLVE_NONHYDRO_GLOBAL_STATIC_PARAMS
            )
        )
        next_state, next_diagnostic_state, _, _ = solver.time_step(
            diagnostic_state_nh=diagnostic_state_nh,
            prognostic_state=prognostic_state,
            intermediate_state=solver.initial_intermediate_state(),
            prep_adv=prep_adv,
            **step_kwargs,
        )
    else:
        jax = pytest.importorskip("jax")
        jnp = jax.numpy
        grid = solver_kwargs.pop("grid")
        vertical_params = solver_kwargs.pop("vertical_params")
        solver = solve_nonhydro_global.SolveNonhydroGlobal(
            grid=dataclasses.replace(
                grid, connectivities={k: _to_jax(v, jnp) for k, v in grid.connectivities.items()}
            ),
            vertical_params=v_grid.VerticalGrid(
                config=vertical_params.config,
                vct_a=_to_jax(vertical_params.vct_a, jnp),
                vct_b=_to_jax(vertical_params.vct_b, jnp),
            ),
            **{k: _to_jax(v, jnp) for k, v in solver_kwargs.items()},
            allocator=jnp,
        )
        jax_diagnostic_state = _to_jax(diagnostic_state_nh, jnp)
        jax_prep_adv = _to_jax(prep_adv, jnp)
        intermediate_state = solver.initial_intermediate_state()

        @jax.jit
        def step(fields):
            new, new_diagnostic, _, _ = solver.time_step(
                diagnostic_state_nh=jax_diagnostic_state,
                prognostic_state=prognostics.PrognosticState(**fields),
                intermediate_state=intermediate_state,
                prep_adv=jax_prep_adv,
                **step_kwargs,
            )
            return {k: getattr(new, k) for k in _PROGNOSTIC_FIELDS}, (
                new_diagnostic.exner_dynamical_increment
            )

        fields, exner_dynamical_increment = step(
            {k: _to_jax(getattr(prognostic_state, k), jnp) for k in _PROGNOSTIC_FIELDS}
        )
        next_state = prognostics.PrognosticState(**fields)
        next_diagnostic_state = types.SimpleNamespace(
            exner_dynamical_increment=exner_dynamical_increment
        )
    _check_against_savepoints(
        sp,
        prognostic_state,
        next_state,
        next_diagnostic_state,
        savepoint_nonhydro_exit,
        savepoint_nonhydro_step_final,
    )
