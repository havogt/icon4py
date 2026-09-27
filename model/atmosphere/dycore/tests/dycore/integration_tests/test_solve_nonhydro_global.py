# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import pytest

from icon4py.model.atmosphere.dycore import (
    dycore_states,
    solve_nonhydro as solve_nh,
    solve_nonhydro_global,
)
from icon4py.model.common import dimension as dims, model_backends
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import definitions as test_defs, test_utils

from .. import utils
from ..fixtures import *  # noqa: F403


@pytest.mark.embedded_only
@pytest.mark.datatest
@pytest.mark.parametrize(
    "istep_init, substep_init, istep_exit, substep_exit, at_initial_timestep", [(1, 1, 2, 1, True)]
)
@pytest.mark.parametrize(
    "experiment_description, step_date_init, step_date_exit",
    [
        (
            test_defs.Experiments.EXCLAIM_APE,
            "2000-01-01T00:00:02.000",
            "2000-01-01T00:00:02.000",
        ),
    ],
)
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
    config = experiment.config.nonhydrostatic
    sp = savepoint_nonhydro_init
    allocator = model_backends.get_allocator(backend)

    solver = solve_nonhydro_global.SolveNonhydroGlobal(
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
        allocator=allocator,
        backend=backend,
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

    next_state, next_diagnostic_state, _, _ = solver.time_step(
        diagnostic_state_nh=diagnostic_state_nh,
        prognostic_state=prognostic_state,
        intermediate_state=solver.initial_intermediate_state(),
        prep_adv=prep_adv,
        second_order_divdamp_factor=sp.divdamp_fac_o2(),
        dtime=sp.get_metadata("dtime").get("dtime"),
        ndyn_substeps_var=experiment.config.driver.ndyn_substeps,
        at_initial_timestep=at_initial_timestep,
        prepare_fluxes_for_advection=sp.get_metadata("prep_adv").get("prep_adv"),
        at_first_substep=substep_init == 1,
        at_last_substep=substep_init == experiment.config.driver.ndyn_substeps,
    )

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

    assert test_utils.dallclose(
        next_state.theta_v.asnumpy(), savepoint_nonhydro_step_final.theta_v_new().asnumpy()
    )
    assert test_utils.dallclose(
        next_state.exner.asnumpy(), savepoint_nonhydro_step_final.exner_new().asnumpy()
    )
    assert test_utils.dallclose(
        next_state.vn.asnumpy(),
        savepoint_nonhydro_exit.vn_new().asnumpy(),
        rtol=1e-12,
        atol=1e-13,
    )
    assert test_utils.dallclose(
        next_state.rho.asnumpy(), savepoint_nonhydro_exit.rho_new().asnumpy()
    )
    assert test_utils.dallclose(
        next_state.w.asnumpy(), savepoint_nonhydro_exit.w_new().asnumpy(), atol=8e-14
    )
    assert test_utils.dallclose(
        next_diagnostic_state.exner_dynamical_increment.asnumpy(),
        savepoint_nonhydro_exit.exner_dyn_incr().asnumpy(),
        atol=1e-14,
    )
