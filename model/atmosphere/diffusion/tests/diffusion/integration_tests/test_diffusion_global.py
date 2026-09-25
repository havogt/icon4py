# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import pytest

from icon4py.model.atmosphere.diffusion import diffusion, diffusion_global, diffusion_states
from icon4py.model.common import model_backends
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.testing import definitions as test_defs, test_utils

from ..fixtures import *  # noqa: F403
from .test_diffusion import (
    get_cell_geometry_for_experiment,
    get_edge_geometry_for_experiment,
    get_grid_for_experiment,
)


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment_description, step_date_init, step_date_exit",
    [
        (
            test_defs.Experiments.EXCLAIM_APE,
            "2000-01-01T00:00:02.000",
            "2000-01-01T00:00:02.000",
        ),
        (test_defs.Experiments.JW, "2008-09-01T00:05:00.000", "2008-09-01T00:05:00.000"),
        (
            test_defs.Experiments.WEISMAN_KLEMP_TORUS,
            "2008-09-01T01:59:48.000",
            "2008-09-01T01:59:48.000",
        ),
    ],
)
def test_run_diffusion_global_single_step(  # noqa: PLR0917 [too-many-positional-arguments]
    experiment,
    step_date_init,
    step_date_exit,
    savepoint_diffusion_init,
    savepoint_diffusion_exit,
    interpolation_state: diffusion_states.DiffusionInterpolationState,
    metric_state: diffusion_states.DiffusionMetricState,
    backend,
):
    grid = get_grid_for_experiment(experiment, backend)
    cell_geometry = get_cell_geometry_for_experiment(experiment, backend)
    edge_geometry = get_edge_geometry_for_experiment(experiment, backend)
    allocator = model_backends.get_allocator(backend)
    dtime = savepoint_diffusion_init.get_metadata("dtime").get("dtime")

    prognostic_state = savepoint_diffusion_init.construct_prognostics()

    vertical_config = experiment.config.vertical_grid
    vct_a, vct_b = v_grid.get_vct_a_and_vct_b(vertical_config, backend)
    vertical_grid = v_grid.VerticalGrid(config=vertical_config, vct_a=vct_a, vct_b=vct_b)
    config = experiment.config.diffusion

    diffusion_granule = diffusion_global.DiffusionGlobal(
        grid=grid,
        config=config,
        params=diffusion.DiffusionParams(config),
        vertical_grid=vertical_grid,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=edge_geometry,
        cell_params=cell_geometry,
        allocator=allocator,
        backend=backend,
        ndyn_substeps=experiment.config.driver.ndyn_substeps,
    )

    new_prognostic_state = diffusion_granule.run(prognostic_state=prognostic_state, dtime=dtime)

    for name in ("rho", "w", "vn", "exner", "theta_v"):
        assert test_utils.dallclose(
            getattr(prognostic_state, name).asnumpy(),
            getattr(savepoint_diffusion_init, name)().asnumpy(),
        )
    assert test_utils.dallclose(
        new_prognostic_state.vn.asnumpy(),
        savepoint_diffusion_exit.vn().asnumpy(),
        atol=1.0e-8,
        rtol=1.0e-9,
    )
    assert test_utils.dallclose(
        new_prognostic_state.w.asnumpy(), savepoint_diffusion_exit.w().asnumpy(), atol=1e-14
    )
    assert test_utils.dallclose(
        new_prognostic_state.theta_v.asnumpy(), savepoint_diffusion_exit.theta_v().asnumpy()
    )
    assert test_utils.dallclose(
        new_prognostic_state.exner.asnumpy(), savepoint_diffusion_exit.exner().asnumpy()
    )
