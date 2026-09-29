# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import dataclasses

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.diffusion import diffusion, diffusion_global, diffusion_states
from icon4py.model.common import model_backends
from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.testing import definitions as test_defs, test_utils

from ..fixtures import *  # noqa: F403
from .test_diffusion import (
    get_cell_geometry_for_experiment,
    get_edge_geometry_for_experiment,
    get_grid_for_experiment,
)


_STEPS = [
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
]


def _diffusion_global(
    experiment, interpolation_state, metric_state, backend
) -> diffusion_global.DiffusionGlobal:
    vertical_config = experiment.config.vertical_grid
    vct_a, vct_b = v_grid.get_vct_a_and_vct_b(vertical_config, backend)
    config = experiment.config.diffusion
    return diffusion_global.DiffusionGlobal(
        grid=get_grid_for_experiment(experiment, backend),
        config=config,
        params=diffusion.DiffusionParams(config),
        vertical_grid=v_grid.VerticalGrid(config=vertical_config, vct_a=vct_a, vct_b=vct_b),
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=get_edge_geometry_for_experiment(experiment, backend),
        cell_params=get_cell_geometry_for_experiment(experiment, backend),
        allocator=model_backends.get_allocator(backend),
        backend=backend,
        ndyn_substeps=experiment.config.driver.ndyn_substeps,
    )


@pytest.mark.datatest
@pytest.mark.parametrize("experiment_description, step_date_init, step_date_exit", _STEPS)
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
    dtime = savepoint_diffusion_init.get_metadata("dtime").get("dtime")
    prognostic_state = savepoint_diffusion_init.construct_prognostics()
    diffusion_granule = _diffusion_global(experiment, interpolation_state, metric_state, backend)

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


@pytest.mark.datatest
@pytest.mark.parametrize("experiment_description, step_date_init, step_date_exit", _STEPS)
def test_diffusion_global_matches_diffusion_without_turbulence(  # noqa: PLR0917 [too-many-positional-arguments]
    experiment,
    step_date_init,
    step_date_exit,
    savepoint_diffusion_init,
    interpolation_state: diffusion_states.DiffusionInterpolationState,
    metric_state: diffusion_states.DiffusionMetricState,
    backend,
):
    dtime = savepoint_diffusion_init.get_metadata("dtime").get("dtime")
    config = dataclasses.replace(
        experiment.config.diffusion,
        shear_type=diffusion.TurbulenceShearForcingType.VERTICAL_OF_HORIZONTAL_WIND,
        loutshs=False,
        a_hshr=0.0,
    )
    vertical_config = experiment.config.vertical_grid
    vct_a, vct_b = v_grid.get_vct_a_and_vct_b(vertical_config, backend)
    assert experiment.config.interpolation.max_nudging_coefficient is not None
    reference_granule = diffusion.Diffusion(
        grid=get_grid_for_experiment(experiment, backend),
        config=config,
        params=diffusion.DiffusionParams(config),
        vertical_grid=v_grid.VerticalGrid(config=vertical_config, vct_a=vct_a, vct_b=vct_b),
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=get_edge_geometry_for_experiment(experiment, backend),
        cell_params=get_cell_geometry_for_experiment(experiment, backend),
        backend=backend,
        exchange=decomp_defs.SingleNodeExchange(),
        ndyn_substeps=experiment.config.driver.ndyn_substeps,
        max_nudging_coefficient=experiment.config.interpolation.max_nudging_coefficient,
    )
    reference_state = savepoint_diffusion_init.construct_prognostics()
    # The APE and JW savepoints hold a placeholder for dwdx and dwdy.
    w_domain = reference_state.w.domain
    reference_granule.run(
        diagnostic_state=diffusion_states.DiffusionDiagnosticState(
            hdef_ic=savepoint_diffusion_init.hdef_ic(),
            div_ic=savepoint_diffusion_init.div_ic(),
            dwdx=gtx.as_field(w_domain, np.zeros(w_domain.shape)),
            dwdy=gtx.as_field(w_domain, np.zeros(w_domain.shape)),
        ),
        prognostic_state=reference_state,
        dtime=dtime,
    )

    new_state = _diffusion_global(experiment, interpolation_state, metric_state, backend).run(
        prognostic_state=savepoint_diffusion_init.construct_prognostics(), dtime=dtime
    )

    for name in ("vn", "w", "theta_v", "exner"):
        np.testing.assert_allclose(
            getattr(new_state, name).asnumpy(),
            getattr(reference_state, name).asnumpy(),
            rtol=1e-12,
            atol=1e-14,
            err_msg=name,
        )
