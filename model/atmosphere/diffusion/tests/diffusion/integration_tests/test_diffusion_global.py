# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import dataclasses
from typing import Any

import gt4py.next as gtx
import numpy as np
import pytest
from gt4py.next import common as gtx_common

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


def _to_torch(obj: Any, torch: Any) -> Any:
    if isinstance(obj, gtx_common.Connectivity):
        return gtx.as_connectivity(
            obj.domain,
            obj.codomain,
            torch.asarray(obj.asnumpy()),
            skip_value=obj.skip_value,
            allocator=torch,
        )
    if isinstance(obj, gtx.Field):
        return gtx.as_field(obj.domain, torch.asarray(obj.asnumpy()), allocator=torch)
    if isinstance(obj, tuple):
        return tuple(_to_torch(x, torch) for x in obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.replace(
            obj,
            **{
                f.name: _to_torch(getattr(obj, f.name), torch)
                for f in dataclasses.fields(obj)
                if f.init
            },
        )
    return obj


@pytest.mark.datatest
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("experiment_description, step_date_init, step_date_exit", _STEPS)
def test_run_diffusion_global_single_step_torch(  # noqa: PLR0917 [too-many-positional-arguments]
    experiment,
    step_date_init,
    step_date_exit,
    savepoint_diffusion_init,
    savepoint_diffusion_exit,
    interpolation_state: diffusion_states.DiffusionInterpolationState,
    metric_state: diffusion_states.DiffusionMetricState,
    backend,
    device,
):
    if backend is not None:
        pytest.skip("torch fields run on the embedded backend only.")
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA device.")

    vertical_config = experiment.config.vertical_grid
    vct_a, vct_b = v_grid.get_vct_a_and_vct_b(vertical_config, backend)
    grid = get_grid_for_experiment(experiment, backend)
    config = experiment.config.diffusion
    dtime = savepoint_diffusion_init.get_metadata("dtime").get("dtime")

    with torch.device(device):
        granule = diffusion_global.DiffusionGlobal(
            grid=dataclasses.replace(
                grid,
                connectivities={k: _to_torch(v, torch) for k, v in grid.connectivities.items()},
            ),
            config=config,
            params=diffusion.DiffusionParams(config),
            vertical_grid=v_grid.VerticalGrid(
                config=vertical_config, vct_a=_to_torch(vct_a, torch), vct_b=_to_torch(vct_b, torch)
            ),
            metric_state=_to_torch(metric_state, torch),
            interpolation_state=_to_torch(interpolation_state, torch),
            edge_params=_to_torch(get_edge_geometry_for_experiment(experiment, backend), torch),
            cell_params=_to_torch(get_cell_geometry_for_experiment(experiment, backend), torch),
            allocator=torch,
            ndyn_substeps=experiment.config.driver.ndyn_substeps,
        )
        new_state = granule.run(
            prognostic_state=_to_torch(savepoint_diffusion_init.construct_prognostics(), torch),
            dtime=dtime,
        )

    for name in ("vn", "w", "theta_v", "exner", "rho"):
        array = getattr(new_state, name).ndarray
        assert isinstance(array, torch.Tensor), name
        assert array.device.type == device, name
    for name, atol, rtol in (
        ("vn", 1.0e-8, 1.0e-9),
        ("w", 1.0e-14, 1.0e-12),
        ("theta_v", 0.0, 1.0e-12),
        ("exner", 0.0, 1.0e-12),
    ):
        actual = getattr(new_state, name).asnumpy()
        expected = getattr(savepoint_diffusion_exit, name)().asnumpy()
        print(f"{name}: max abs diff {np.max(np.abs(actual - expected)):.3e}")
        assert test_utils.dallclose(actual, expected, atol=atol, rtol=rtol), name
