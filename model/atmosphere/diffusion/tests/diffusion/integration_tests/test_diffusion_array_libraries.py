# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import dataclasses
import logging
from typing import Any

import gt4py.next as gtx
import numpy as np
import pytest
from gt4py.next import common as gtx_common

import icon4py.model.common.dimension as dims
import icon4py.model.common.grid.states as grid_states
from icon4py.model.atmosphere.diffusion import diffusion, diffusion_states
from icon4py.model.common import model_backends
from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.common.grid import (
    geometry as grid_geometry,
    geometry_attributes as geometry_meta,
    grid_manager as gm,
    vertical as v_grid,
)
from icon4py.model.common.interpolation import interpolation_attributes, interpolation_factory
from icon4py.model.common.metrics import metrics_attributes, metrics_factory
from icon4py.model.common.states import prognostic_state as prognostics
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import compiled_run
from icon4py.model.testing.fixtures.benchmark import (
    geometry_field_source,
    interpolation_field_source,
    metrics_field_source,
)
from icon4py.model.testing.fixtures.datatest import backend_like
from icon4py.model.testing.fixtures.stencil_tests import grid_manager


@pytest.fixture
def diffusion_inputs(
    geometry_field_source: grid_geometry.GridGeometry,
    grid_manager: gm.GridManager,
    interpolation_field_source: interpolation_factory.InterpolationFieldsFactory,
    metrics_field_source: metrics_factory.MetricsFieldsFactory,
    backend_like: model_backends.BackendLike,
) -> dict[str, Any]:
    if backend_like is not None:
        pytest.skip("array libraries run on the embedded backend only")
    geo = geometry_field_source.get
    mesh = grid_manager.grid
    config = diffusion.DiffusionConfig(
        diffusion_type=diffusion.DiffusionType.SMAGORINSKY_4TH_ORDER,
        apply_to_vertical_wind=True,
        apply_to_horizontal_wind=True,
        type_t_diffu=diffusion.TemperatureDiscretizationType.HETEROGENEOUS,
        type_vn_diffu=diffusion.SmagorinskyStencilType.DIAMOND_VERTICES,
        hdiff_efdt_ratio=24.0,
        hdiff_w_efdt_ratio=15.0,
        smagorinski_scaling_factor=0.025,
        apply_zdiffusion_t=False,
        velocity_boundary_diffusion_denominator=150.0,
        shear_type=diffusion.TurbulenceShearForcingType.VERTICAL_HORIZONTAL_OF_HORIZONTAL_VERTICAL_WIND,
    )
    vertical_config = v_grid.VerticalGridConfig(
        mesh.num_levels,
        lowest_layer_thickness=50,
        model_top_height=23500.0,
        stretch_factor=1.0,
        rayleigh_damping_height=1.0,
    )
    vct_a, vct_b = v_grid.get_vct_a_and_vct_b(vertical_config, allocator=None)
    interp = interpolation_field_source.get
    metric = metrics_field_source.get
    rng = np.random.default_rng(42)

    def random(*field_dims: gtx.Dimension, low: float = -1.0, high: float = 1.0) -> gtx.Field:
        return data_alloc.random_field(mesh, *field_dims, low=low, high=high, rng=rng)

    return {
        "grid": mesh,
        "config": config,
        "vertical_config": vertical_config,
        "vct_a": vct_a,
        "vct_b": vct_b,
        "cell_params": grid_states.CellParams(
            cell_center_lat=geo(geometry_meta.CELL_LAT),
            cell_center_lon=geo(geometry_meta.CELL_LON),
            area=geo(geometry_meta.CELL_AREA),
        ),
        "edge_params": grid_states.EdgeParams(
            tangent_orientation=geo(geometry_meta.TANGENT_ORIENTATION),
            inverse_primal_edge_lengths=geo(f"inverse_of_{geometry_meta.EDGE_LENGTH}"),
            inverse_dual_edge_lengths=geo(f"inverse_of_{geometry_meta.DUAL_EDGE_LENGTH}"),
            inverse_vertex_vertex_lengths=geo(f"inverse_of_{geometry_meta.VERTEX_VERTEX_LENGTH}"),
            primal_normal_vert=(
                geo(geometry_meta.EDGE_NORMAL_VERTEX_U),
                geo(geometry_meta.EDGE_NORMAL_VERTEX_V),
            ),
            dual_normal_vert=(
                geo(geometry_meta.EDGE_TANGENT_VERTEX_U),
                geo(geometry_meta.EDGE_TANGENT_VERTEX_V),
            ),
            primal_normal_cell=(
                geo(geometry_meta.EDGE_NORMAL_CELL_U),
                geo(geometry_meta.EDGE_NORMAL_CELL_V),
            ),
            dual_normal_cell=(
                geo(geometry_meta.EDGE_TANGENT_CELL_U),
                geo(geometry_meta.EDGE_TANGENT_CELL_V),
            ),
            edge_areas=geo(geometry_meta.EDGE_AREA),
            coriolis_frequency=geo(geometry_meta.CORIOLIS_PARAMETER),
            edge_center=(geo(geometry_meta.EDGE_LAT), geo(geometry_meta.EDGE_LON)),
            primal_normal=(geo(geometry_meta.EDGE_NORMAL_U), geo(geometry_meta.EDGE_NORMAL_V)),
            primal_edge_lengths=geo(geometry_meta.EDGE_LENGTH),
            dual_edge_lengths=geo(geometry_meta.DUAL_EDGE_LENGTH),
        ),
        "interpolation_state": diffusion_states.DiffusionInterpolationState(
            e_bln_c_s=interp(interpolation_attributes.E_BLN_C_S),
            rbf_coeff_1=interp(interpolation_attributes.RBF_VEC_COEFF_V1),
            rbf_coeff_2=interp(interpolation_attributes.RBF_VEC_COEFF_V2),
            geofac_div=interp(interpolation_attributes.GEOFAC_DIV),
            geofac_n2s=interp(interpolation_attributes.GEOFAC_N2S),
            geofac_grg_x=interp(interpolation_attributes.GEOFAC_GRG_X),
            geofac_grg_y=interp(interpolation_attributes.GEOFAC_GRG_Y),
            nudgecoeff_e=interp(interpolation_attributes.NUDGECOEFFS_E),
        ),
        "metric_state": diffusion_states.DiffusionMetricState(
            theta_ref_mc=metric(metrics_attributes.THETA_REF_MC),
            wgtfac_c=metric(metrics_attributes.WGTFAC_C),
            zd_intcoef=metric(metrics_attributes.ZD_INTCOEF),
            zd_vertoffset=metric(metrics_attributes.ZD_VERTOFFSET),
            zd_diffcoef=metric(metrics_attributes.ZD_DIFFCOEF),
        ),
        "diagnostic_state": diffusion_states.DiffusionDiagnosticState(
            hdef_ic=random(dims.CellDim, dims.KHalfDim),
            div_ic=random(dims.CellDim, dims.KHalfDim),
            dwdx=random(dims.CellDim, dims.KHalfDim),
            dwdy=random(dims.CellDim, dims.KHalfDim),
        ),
        "prognostic_state": prognostics.PrognosticState(
            w=random(dims.CellDim, dims.KHalfDim, low=0.0),
            vn=random(dims.EdgeDim, dims.KDim, low=-30.0, high=30.0),
            exner=random(dims.CellDim, dims.KDim, low=0.9),
            theta_v=random(dims.CellDim, dims.KDim, low=270.0, high=290.0),
            rho=random(dims.CellDim, dims.KDim),
        ),
    }


def granule_arguments(inputs: dict[str, Any], backend: Any) -> dict[str, Any]:
    xp = backend if isinstance(backend, model_backends.EmbeddedBackend) else np
    grid = inputs["grid"] if xp is np else compiled_run.with_connectivities_on(inputs["grid"], xp)
    config = inputs["config"]
    return {
        "grid": grid,
        "config": config,
        "params": diffusion.DiffusionParams(config),
        "vertical_grid": v_grid.VerticalGrid(
            config=inputs["vertical_config"],
            vct_a=compiled_run.to_array_ns(inputs["vct_a"], xp),
            vct_b=compiled_run.to_array_ns(inputs["vct_b"], xp),
        ),
        "metric_state": compiled_run.to_array_ns(inputs["metric_state"], xp),
        "interpolation_state": compiled_run.to_array_ns(inputs["interpolation_state"], xp),
        "edge_params": compiled_run.to_array_ns(inputs["edge_params"], xp),
        "cell_params": compiled_run.to_array_ns(inputs["cell_params"], xp),
        "backend": backend,
        "exchange": decomp_defs.SingleNodeExchange(),
        "ndyn_substeps": 5,
        "max_nudging_coefficient": 0.375,
    }


_OUTPUTS = ("vn", "w", "theta_v", "exner", "div_ic", "hdef_ic", "dwdx", "dwdy")


def outputs(
    diagnostic_state: diffusion_states.DiffusionDiagnosticState,
    prognostic_state: prognostics.PrognosticState,
) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(
            getattr(
                prognostic_state if hasattr(prognostic_state, name) else diagnostic_state, name
            ).asnumpy()
        )
        for name in _OUTPUTS
    }


_EXECUTIONS: dict[str, tuple[str, bool]] = {
    "numpy": ("embedded", False),
    "gtfn_cpu": ("gtfn_cpu", False),
    "torch": ("embedded_torch", False),
    "torch_compile": ("embedded_torch", True),
    "jax": ("embedded_jax", False),
    "jax_jit": ("embedded_jax", True),
}


def run(
    inputs: dict[str, Any], execution: str, *, initial_run: bool = False, steps: int = 2
) -> dict[str, np.ndarray]:
    backend_name, compiled = _EXECUTIONS[execution]
    pytest.importorskip(
        {"embedded_torch": "torch", "embedded_jax": "jax"}.get(backend_name, "numpy")
    )
    backend = model_backends.BACKENDS[backend_name]
    xp = backend if isinstance(backend, model_backends.EmbeddedBackend) else np
    diagnostic_state = compiled_run.to_array_ns(inputs["diagnostic_state"], xp)
    prognostic_state = compiled_run.to_array_ns(inputs["prognostic_state"], xp)
    granule = diffusion.Diffusion(**granule_arguments(inputs, backend))
    program = granule.apply_diffusion_to_prognostic_fields.func
    assert (program.backend is None) == (execution != "gtfn_cpu"), program.backend
    step = compiled_run.compiled_run(granule, backend) if compiled else granule.run
    for _ in range(steps):
        step(
            diagnostic_state=diagnostic_state,
            prognostic_state=prognostic_state,
            dtime=10.0,
            initial_run=initial_run,
        )
    array_type = type(xp.asarray(np.zeros(1)))
    for name in ("vn", "w", "theta_v", "exner"):
        assert isinstance(getattr(prognostic_state, name).ndarray, array_type), name
    return outputs(diagnostic_state, prognostic_state)


def compare(result: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> dict[str, float]:
    relative = {}
    for name in _OUTPUTS:
        scale = float(np.max(np.abs(reference[name])))
        relative[name] = float(np.max(np.abs(result[name] - reference[name]))) / scale
        print(f"{name}: max rel diff {relative[name]:.3e}")
    return relative


@pytest.mark.parametrize("initial_run", [False, True], ids=["regular", "initial"])
@pytest.mark.parametrize("execution", ["torch", "torch_compile", "jax", "jax_jit"])
def test_diffusion_array_library(
    diffusion_inputs: dict[str, Any],
    execution: str,
    initial_run: bool,
) -> None:
    reference = run(diffusion_inputs, "numpy", initial_run=initial_run)
    initial = outputs(diffusion_inputs["diagnostic_state"], diffusion_inputs["prognostic_state"])
    for name in ("vn", "w", "theta_v"):
        assert not np.array_equal(reference[name], initial[name]), name
    result = run(diffusion_inputs, execution, initial_run=initial_run)
    assert all(r <= 1e-12 for r in compare(result, reference).values())


@pytest.mark.parametrize("initial_run", [False, True], ids=["regular", "initial"])
def test_diffusion_numpy_against_gtfn(diffusion_inputs: dict[str, Any], initial_run: bool) -> None:
    reference = run(diffusion_inputs, "gtfn_cpu", initial_run=initial_run)
    result = run(diffusion_inputs, "numpy", initial_run=initial_run)
    assert all(r <= 1e-12 for r in compare(result, reference).values())
