# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import contextlib
import dataclasses
import time
from typing import TYPE_CHECKING, Any

import gt4py.next as gtx
import pytest
from gt4py.next import common as gtx_common

import icon4py.model.common.dimension as dims
import icon4py.model.common.grid.states as grid_states
from icon4py.model.atmosphere.diffusion import diffusion, diffusion_global, diffusion_states
from icon4py.model.common import constants, model_backends, model_options
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
from icon4py.model.common.utils import data_allocation as data_alloc, device_utils
from icon4py.model.testing import torch_steps
from icon4py.model.testing.fixtures.benchmark import (
    geometry_field_source,
    interpolation_field_source,
    metrics_field_source,
)
from icon4py.model.testing.fixtures.datatest import backend_like
from icon4py.model.testing.fixtures.stencil_tests import grid_manager


def _setup(
    geometry_field_source: grid_geometry.GridGeometry,
    grid_manager: gm.GridManager,
    interpolation_field_source: interpolation_factory.InterpolationFieldsFactory,
    metrics_field_source: metrics_factory.MetricsFieldsFactory,
    backend_like: model_backends.BackendLike,
) -> dict[str, Any]:
    allocator = model_backends.get_allocator(backend_like)
    dtime = 10.0

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

    diffusion_parameters = diffusion.DiffusionParams(config)

    mesh = grid_manager.grid

    cell_geometry = grid_states.CellParams(
        cell_center_lat=geometry_field_source.get(geometry_meta.CELL_LAT),
        cell_center_lon=geometry_field_source.get(geometry_meta.CELL_LON),
        area=geometry_field_source.get(geometry_meta.CELL_AREA),
    )
    edge_geometry = grid_states.EdgeParams(
        tangent_orientation=geometry_field_source.get(geometry_meta.TANGENT_ORIENTATION),
        inverse_primal_edge_lengths=geometry_field_source.get(
            f"inverse_of_{geometry_meta.EDGE_LENGTH}"
        ),
        inverse_dual_edge_lengths=geometry_field_source.get(
            f"inverse_of_{geometry_meta.DUAL_EDGE_LENGTH}"
        ),
        inverse_vertex_vertex_lengths=geometry_field_source.get(
            f"inverse_of_{geometry_meta.VERTEX_VERTEX_LENGTH}"
        ),
        primal_normal_vert=(
            geometry_field_source.get(geometry_meta.EDGE_NORMAL_VERTEX_U),
            geometry_field_source.get(geometry_meta.EDGE_NORMAL_VERTEX_V),
        ),
        dual_normal_vert=(
            geometry_field_source.get(geometry_meta.EDGE_TANGENT_VERTEX_U),
            geometry_field_source.get(geometry_meta.EDGE_TANGENT_VERTEX_V),
        ),
        primal_normal_cell=(
            geometry_field_source.get(geometry_meta.EDGE_NORMAL_CELL_U),
            geometry_field_source.get(geometry_meta.EDGE_NORMAL_CELL_V),
        ),
        dual_normal_cell=(
            geometry_field_source.get(geometry_meta.EDGE_TANGENT_CELL_U),
            geometry_field_source.get(geometry_meta.EDGE_TANGENT_CELL_V),
        ),
        edge_areas=geometry_field_source.get(geometry_meta.EDGE_AREA),
        coriolis_frequency=geometry_field_source.get(geometry_meta.CORIOLIS_PARAMETER),
        edge_center=(
            geometry_field_source.get(geometry_meta.EDGE_LAT),
            geometry_field_source.get(geometry_meta.EDGE_LON),
        ),
        primal_normal=(
            geometry_field_source.get(geometry_meta.EDGE_NORMAL_U),
            geometry_field_source.get(geometry_meta.EDGE_NORMAL_V),
        ),
        primal_edge_lengths=geometry_field_source.get(geometry_meta.EDGE_LENGTH),
        dual_edge_lengths=geometry_field_source.get(geometry_meta.DUAL_EDGE_LENGTH),
    )

    vertical_config = v_grid.VerticalGridConfig(
        mesh.num_levels,
        lowest_layer_thickness=50,
        model_top_height=23500.0,
        stretch_factor=1.0,
        rayleigh_damping_height=1.0,
    )
    vct_a, vct_b = v_grid.get_vct_a_and_vct_b(vertical_config, allocator=allocator)

    vertical_grid = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=vct_a,
        vct_b=vct_b,
    )

    interpolation_state = diffusion_states.DiffusionInterpolationState(
        e_bln_c_s=interpolation_field_source.get(interpolation_attributes.E_BLN_C_S),
        rbf_coeff_1=interpolation_field_source.get(interpolation_attributes.RBF_VEC_COEFF_V1),
        rbf_coeff_2=interpolation_field_source.get(interpolation_attributes.RBF_VEC_COEFF_V2),
        geofac_div=interpolation_field_source.get(interpolation_attributes.GEOFAC_DIV),
        geofac_n2s=interpolation_field_source.get(interpolation_attributes.GEOFAC_N2S),
        geofac_grg_x=interpolation_field_source.get(interpolation_attributes.GEOFAC_GRG_X),
        geofac_grg_y=interpolation_field_source.get(interpolation_attributes.GEOFAC_GRG_Y),
        nudgecoeff_e=interpolation_field_source.get(interpolation_attributes.NUDGECOEFFS_E),
    )

    metric_state = diffusion_states.DiffusionMetricState(
        theta_ref_mc=metrics_field_source.get(metrics_attributes.THETA_REF_MC),
        wgtfac_c=metrics_field_source.get(metrics_attributes.WGTFAC_C),
        zd_intcoef=metrics_field_source.get(metrics_attributes.ZD_INTCOEF),
        zd_vertoffset=metrics_field_source.get(metrics_attributes.ZD_VERTOFFSET),
        zd_diffcoef=metrics_field_source.get(metrics_attributes.ZD_DIFFCOEF),
    )
    diagnostic_state = diffusion_states.DiffusionDiagnosticState(
        hdef_ic=data_alloc.random_field(mesh, dims.CellDim, dims.KHalfDim, allocator=allocator),
        div_ic=data_alloc.random_field(mesh, dims.CellDim, dims.KHalfDim, allocator=allocator),
        dwdx=data_alloc.random_field(mesh, dims.CellDim, dims.KHalfDim, allocator=allocator),
        dwdy=data_alloc.random_field(mesh, dims.CellDim, dims.KHalfDim, allocator=allocator),
    )

    prognostic_state = prognostics.PrognosticState(
        w=data_alloc.random_field(mesh, dims.CellDim, dims.KHalfDim, low=0.0, allocator=allocator),
        vn=data_alloc.random_field(mesh, dims.EdgeDim, dims.KDim, allocator=allocator),
        exner=data_alloc.random_field(mesh, dims.CellDim, dims.KDim, allocator=allocator),
        theta_v=data_alloc.random_field(mesh, dims.CellDim, dims.KDim, allocator=allocator),
        rho=data_alloc.random_field(mesh, dims.CellDim, dims.KDim, allocator=allocator),
    )

    return dict(
        mesh=mesh,
        config=config,
        diffusion_parameters=diffusion_parameters,
        vertical_grid=vertical_grid,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_geometry=edge_geometry,
        cell_geometry=cell_geometry,
        diagnostic_state=diagnostic_state,
        prognostic_state=prognostic_state,
        dtime=dtime,
    )


@pytest.mark.embedded_remap_error
@pytest.mark.benchmark
@pytest.mark.continuous_benchmarking
@pytest.mark.benchmark_only
def test_diffusion_benchmark(  # noqa: PLR0917 [too-many-positional-arguments]
    geometry_field_source: grid_geometry.GridGeometry,
    grid_manager: gm.GridManager,
    interpolation_field_source: interpolation_factory.InterpolationFieldsFactory,
    metrics_field_source: metrics_factory.MetricsFieldsFactory,
    backend_like: model_backends.BackendLike,
    benchmark: Any,
) -> None:
    setup = _setup(
        geometry_field_source,
        grid_manager,
        interpolation_field_source,
        metrics_field_source,
        backend_like,
    )
    diffusion_granule = diffusion.Diffusion(
        grid=setup["mesh"],
        config=setup["config"],
        params=setup["diffusion_parameters"],
        vertical_grid=setup["vertical_grid"],
        metric_state=setup["metric_state"],
        interpolation_state=setup["interpolation_state"],
        edge_params=setup["edge_geometry"],
        cell_params=setup["cell_geometry"],
        backend=backend_like,
        exchange=decomp_defs.SingleNodeExchange(),
        ndyn_substeps=5,
        max_nudging_coefficient=0.375,
    )

    benchmark(
        device_utils.synchronized_function(
            diffusion_granule.run, allocator=model_backends.get_allocator(backend_like)
        ),
        setup["diagnostic_state"],
        setup["prognostic_state"],
        setup["dtime"],
    )


_PROGNOSTIC_FIELDS = ("rho", "w", "vn", "exner", "theta_v")


def _diffusion_global_step(
    setup: dict[str, Any], execution: str, backend_like: model_backends.BackendLike
) -> tuple[Any, dict[str, gtx.Field]]:
    """
    The `DiffusionGlobal` step and its prognostic input.

    `execution="jax"` jits it on JAX arrays on the default JAX device, `"torch"` and
    `"torch_compile"` run it on torch tensors on the torch device, eager and under
    `torch.compile`; the constant fields are computed with `--backend` and then converted.
    `execution="backend"` runs it on `--backend`. Every call ends synchronised.
    """
    mesh = setup["mesh"]
    if mesh.limited_area:
        pytest.skip("'DiffusionGlobal' does not support limited area grids.")
    constants = {
        k: setup[k]
        for k in ("metric_state", "interpolation_state", "edge_geometry", "cell_geometry")
    }
    vertical_grid = setup["vertical_grid"]
    prognostic_input = {k: getattr(setup["prognostic_state"], k) for k in _PROGNOSTIC_FIELDS}
    torch = pytest.importorskip("torch") if execution.startswith("torch") else None
    placement = torch_steps.device(torch) if torch is not None else contextlib.nullcontext()

    with placement:
        if execution == "backend":
            backend = model_options.customize_backend(None, backend_like)
            allocator = model_backends.get_allocator(backend_like)
        else:
            xp = torch if torch is not None else pytest.importorskip("jax").numpy
            mesh = dataclasses.replace(
                mesh,
                connectivities={
                    k: torch_steps.to_array_ns(v, xp) for k, v in mesh.connectivities.items()
                },
            )
            constants = torch_steps.to_array_ns(constants, xp)
            vertical_grid = v_grid.VerticalGrid(
                config=vertical_grid.config,
                vct_a=torch_steps.to_array_ns(vertical_grid.vct_a, xp),
                vct_b=torch_steps.to_array_ns(vertical_grid.vct_b, xp),
            )
            prognostic_input = torch_steps.to_array_ns(prognostic_input, xp)
            allocator, backend = xp, None
        granule = diffusion_global.DiffusionGlobal(
            grid=mesh,
            config=setup["config"],
            params=setup["diffusion_parameters"],
            vertical_grid=vertical_grid,
            metric_state=constants["metric_state"],
            interpolation_state=constants["interpolation_state"],
            edge_params=constants["edge_geometry"],
            cell_params=constants["cell_geometry"],
            allocator=allocator,
            ndyn_substeps=5,
            backend=backend,
        )
    dtime = setup["dtime"]

    def step(fields: dict[str, gtx.Field]) -> dict[str, gtx.Field]:
        new = granule.run(prognostic_state=prognostics.PrognosticState(**fields), dtime=dtime)
        return {k: getattr(new, k) for k in _PROGNOSTIC_FIELDS}

    if torch is not None:
        return torch_steps.step(
            step, torch, compiled=execution == "torch_compile"
        ), prognostic_input
    if execution == "jax":
        jax = pytest.importorskip("jax")
        jitted = jax.jit(step)
        return lambda fields: jax.block_until_ready(jitted(fields)), prognostic_input
    sync = device_utils.synchronized_function(
        step, allocator=model_backends.get_allocator(backend_like)
    )
    return sync, prognostic_input


@pytest.mark.benchmark
@pytest.mark.benchmark_only
@pytest.mark.parametrize("execution", ["jax", "backend", "torch", "torch_compile"])
def test_diffusion_global_benchmark(  # noqa: PLR0917 [too-many-positional-arguments]
    geometry_field_source: grid_geometry.GridGeometry,
    grid_manager: gm.GridManager,
    interpolation_field_source: interpolation_factory.InterpolationFieldsFactory,
    metrics_field_source: metrics_factory.MetricsFieldsFactory,
    backend_like: model_backends.BackendLike,
    execution: str,
    benchmark: Any,
) -> None:
    """
    Benchmark `DiffusionGlobal`, the fused single-field-operator diffusion step.

    The first call, which compiles, is timed on its own in `extra_info["first_call_s"]`.
    """
    if execution == "torch_compile" and (reason := torch_steps.compile_blocked("diffusion")):
        pytest.skip(reason)
    if execution == "torch" and not torch_steps.eager_timed():
        pytest.skip("eager torch is timed only with ICON4PY_BENCH_TORCH_EAGER=1")
    setup = _setup(
        geometry_field_source,
        grid_manager,
        interpolation_field_source,
        metrics_field_source,
        backend_like,
    )
    step, prognostic_input = _diffusion_global_step(setup, execution, backend_like)
    start = time.perf_counter()
    step(prognostic_input)
    benchmark.extra_info["first_call_s"] = time.perf_counter() - start
    benchmark(step, prognostic_input)


def test_diffusion_global_torch_matches_jax(
    geometry_field_source: grid_geometry.GridGeometry,
    grid_manager: gm.GridManager,
    interpolation_field_source: interpolation_factory.InterpolationFieldsFactory,
    metrics_field_source: metrics_factory.MetricsFieldsFactory,
    backend_like: model_backends.BackendLike,
) -> None:
    """`DiffusionGlobal` under `torch.compile` and eager on torch against `jax.jit`, same inputs."""
    setup = _setup(
        geometry_field_source,
        grid_manager,
        interpolation_field_source,
        metrics_field_source,
        backend_like,
    )
    outputs = {}
    for execution in (
        ("jax", "torch")
        if torch_steps.compile_blocked("diffusion")
        else ("jax", "torch", "torch_compile")
    ):
        step, step_input = _diffusion_global_step(setup, execution, backend_like)
        outputs[execution] = {k: v.asnumpy() for k, v in step(step_input).items()}
    failures = torch_steps.compare(outputs["torch"], outputs["jax"], "eager-vs-jax")
    if "torch_compile" in outputs:
        failures += torch_steps.compare(
            outputs["torch_compile"], outputs["torch"], "compile-vs-eager"
        )
        failures += torch_steps.compare(outputs["torch_compile"], outputs["jax"], "compile-vs-jax")
    assert not failures, failures
