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

import icon4py.model.common.utils as common_utils
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


def _to_array_ns(obj: Any, xp: Any) -> Any:
    if isinstance(obj, gtx_common.Connectivity):
        return gtx.as_connectivity(
            obj.domain,
            obj.codomain,
            xp.asarray(obj.asnumpy()),
            skip_value=obj.skip_value,
            allocator=xp,
        )
    if isinstance(obj, gtx.Field):
        return gtx.as_field(obj.domain, xp.asarray(obj.asnumpy()), allocator=xp)
    if isinstance(obj, np.ndarray):
        return xp.asarray(obj)
    if isinstance(obj, tuple):
        return tuple(_to_array_ns(x, xp) for x in obj)
    if isinstance(obj, common_utils.PredictorCorrectorPair):
        return common_utils.PredictorCorrectorPair(*(_to_array_ns(x, xp) for x in obj))
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.replace(
            obj,
            **{
                f.name: _to_array_ns(getattr(obj, f.name), xp)
                for f in dataclasses.fields(obj)
                if f.init
            },
        )
    return obj


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
@pytest.mark.parametrize("compiled", ["eager", "fullgraph", "graph_breaks"])
def test_run_solve_nonhydro_global_single_step_torch(  # noqa: PLR0917 [too-many-positional-arguments]
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
    compiled,
):
    """`SolveNonhydroGlobal` on torch tensors, eager or under torch.compile, against JAX."""
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    config = experiment.config.nonhydrostatic
    sp = savepoint_nonhydro_init
    vertical_params = utils.create_vertical_params(experiment.config.vertical_grid, grid_savepoint)
    constants = {
        "metric_state_nonhydro": utils.construct_metric_state(metrics_savepoint, grid_savepoint),
        "interpolation_state": utils.construct_interpolation_state(interpolation_savepoint),
        "edge_geometry": grid_savepoint.construct_edge_geometry(),
        "cell_geometry": grid_savepoint.construct_cell_geometry(),
    }
    prognostic_state = utils.create_prognostic_states(sp).current
    prep_adv = dycore_states.PrepAdvection(
        vn_traj=sp.vn_traj(),
        mass_flx_me=sp.mass_flx_me(),
        dynamical_vertical_mass_flux_at_cells_on_half_levels=sp.mass_flx_ic(),
        dynamical_vertical_volumetric_flux_at_cells_on_half_levels=data_alloc.zero_field(
            icon_grid, dims.CellDim, dims.KHalfDim
        ),
    )
    diagnostic_state_nh = utils.construct_diagnostics(sp, icon_grid, None)
    ndyn_substeps = experiment.config.driver.ndyn_substeps
    scalars = {
        "second_order_divdamp_factor": float(sp.divdamp_fac_o2()),
        "dtime": float(sp.get_metadata("dtime").get("dtime")),
        "ndyn_substeps_var": ndyn_substeps,
        "at_initial_timestep": at_initial_timestep,
        "prepare_fluxes_for_advection": bool(sp.get_metadata("prep_adv").get("prep_adv")),
        "at_first_substep": substep_init == 1,
        "at_last_substep": substep_init == ndyn_substeps,
    }

    def step_on(xp: Any) -> tuple[Any, dict[str, Any]]:
        solver = solve_nonhydro_global.SolveNonhydroGlobal(
            grid=dataclasses.replace(
                icon_grid,
                connectivities={
                    k: _to_array_ns(v, xp) for k, v in icon_grid.connectivities.items()
                },
            ),
            config=config,
            params=solve_nh.NonHydrostaticParams(config),
            vertical_params=type(vertical_params)(
                config=vertical_params.config,
                vct_a=_to_array_ns(vertical_params.vct_a, xp),
                vct_b=_to_array_ns(vertical_params.vct_b, xp),
            ),
            **{k: _to_array_ns(v, xp) for k, v in constants.items()},
            allocator=xp,
        )
        inputs = {
            "prognostic_state": _to_array_ns(prognostic_state, xp),
            "diagnostic_state_nh": _to_array_ns(diagnostic_state_nh, xp),
            "prep_adv": _to_array_ns(prep_adv, xp),
            "intermediate_state": solver.initial_intermediate_state(),
        }

        def step(inputs: dict[str, Any]) -> dict[str, gtx.Field]:
            next_state, next_diagnostic_state, _, _ = solver.time_step(**inputs, **scalars)
            return {
                **{k: getattr(next_state, k) for k in ("rho", "w", "vn", "exner", "theta_v")},
                "exner_dynamical_increment": next_diagnostic_state.exner_dynamical_increment,
            }

        return step, inputs

    torch_step, torch_inputs = step_on(torch)
    eager = torch_step(torch_inputs)
    if compiled == "graph_breaks":
        try:
            explanation = torch._dynamo.explain(torch_step)(torch_inputs)
        except Exception as error:
            print(f"TORCH_EXPLAIN fails: {type(error).__name__}: {str(error).splitlines()[0]}")
        else:
            print(
                f"TORCH_EXPLAIN graphs {explanation.graph_count}, "
                f"graph breaks {explanation.graph_break_count}, "
                f"reasons {[str(r.reason).splitlines()[0] for r in explanation.break_reasons]}"
            )
    if compiled != "eager":
        torch_step = torch.compile(torch_step, fullgraph=compiled == "fullgraph")
    computed = torch_step(torch_inputs)
    for name, field in computed.items():
        actual, reference = field.asnumpy(), eager[name].asnumpy()
        scale = np.max(np.abs(reference))
        print(f"{name}: max |torch - eager| {np.max(np.abs(actual - reference)):.3e}")
        test_utils.assert_dallclose(actual, reference, rtol=0.0, atol=1e-12 * scale, err_msg=name)
    jax_step, jax_inputs = step_on(jax.numpy)
    # the states are dataclasses, no pytrees: JAX runs eagerly
    expected = jax_step(jax_inputs)

    for name, field in computed.items():
        assert isinstance(field.ndarray, torch.Tensor), name
        actual, reference = field.asnumpy(), expected[name].asnumpy()
        scale = np.max(np.abs(reference))
        print(
            f"{name}: max |torch - jax| {np.max(np.abs(actual - reference)):.3e}, max abs {scale:.3e}"
        )
        test_utils.assert_dallclose(actual, reference, rtol=0.0, atol=1e-12 * scale, err_msg=name)
