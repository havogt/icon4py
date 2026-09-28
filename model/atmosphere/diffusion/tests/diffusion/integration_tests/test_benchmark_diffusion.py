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
import time
from typing import TYPE_CHECKING, Any

import gt4py.next as gtx
import numpy as np
import pytest
from gt4py.next import common as gtx_common

import icon4py.model.common.dimension as dims
import icon4py.model.common.grid.states as grid_states
from icon4py.model.atmosphere.diffusion import diffusion, diffusion_global, diffusion_states
from icon4py.model.atmosphere.diffusion.stencils import diffusion_global_step
from icon4py.model.common import constants, model_backends, model_options
from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.common.grid import (
    geometry as grid_geometry,
    geometry_attributes as geometry_meta,
    grid_manager as gm,
    icon as icon_grid,
    vertical as v_grid,
)
from icon4py.model.common.interpolation import interpolation_attributes, interpolation_factory
from icon4py.model.common.metrics import metrics_attributes, metrics_factory
from icon4py.model.common.states import prognostic_state as prognostics
from icon4py.model.common.utils import data_allocation as data_alloc, device_utils
from icon4py.model.testing import structured_torus
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
        diffusion_granule.run,
        setup["diagnostic_state"],
        setup["prognostic_state"],
        setup["dtime"],
    )


def _to_jax(obj: Any, jnp: Any) -> Any:
    if isinstance(obj, gtx_common.Connectivity):
        return gtx.as_connectivity(
            obj.domain,
            obj.codomain,
            jnp.asarray(obj.asnumpy()),
            skip_value=obj.skip_value,
            allocator=jnp,
        )
    if isinstance(obj, gtx.Field):
        return gtx.as_field(obj.domain, jnp.asarray(obj.asnumpy()), allocator=jnp)
    if isinstance(obj, tuple):
        return tuple(_to_jax(x, jnp) for x in obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.replace(
            obj,
            **{
                f.name: _to_jax(getattr(obj, f.name), jnp)
                for f in dataclasses.fields(obj)
                if f.init
            },
        )
    return obj


_PROGNOSTIC_FIELDS = ("rho", "w", "vn", "exner", "theta_v")


def _diffusion_global_granule(
    setup: dict[str, Any], execution: str, backend_like: model_backends.BackendLike
) -> tuple[diffusion_global.DiffusionGlobal, dict[str, gtx.Field]]:
    """
    The `DiffusionGlobal` granule and its prognostic input.

    With `execution="jax"` it works on JAX arrays on the default JAX device; the constant fields
    are computed with `backend_like` and then converted. With `execution="backend"` it runs on
    `backend_like`.
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

    if execution == "jax":
        jax = pytest.importorskip("jax")
        jnp = jax.numpy
        mesh = dataclasses.replace(
            mesh, connectivities={k: _to_jax(v, jnp) for k, v in mesh.connectivities.items()}
        )
        constants = {k: _to_jax(v, jnp) for k, v in constants.items()}
        vertical_grid = v_grid.VerticalGrid(
            config=vertical_grid.config,
            vct_a=_to_jax(vertical_grid.vct_a, jnp),
            vct_b=_to_jax(vertical_grid.vct_b, jnp),
        )
        prognostic_input = {k: _to_jax(v, jnp) for k, v in prognostic_input.items()}
        allocator, backend = jnp, None
    else:
        # gt4py's static domain inference takes min/max over the neighbour tables including their skip
        # values, which gives a negative edge range through V2E on the pentagons (gt4py-f103).
        backend = model_options.customize_backend(None, backend_like)
        allocator = model_backends.get_allocator(backend_like)
        if os.environ.get("ICON4PY_BENCH_REPLACE_SKIP_VALUES", "1") == "1":
            mesh = icon_grid.with_skip_values_replaced(mesh, allocator=allocator)

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
    if backend is not None and os.environ.get("ICON4PY_BENCH_STATIC_SIZES", "1") == "1":
        # The vertical sizes that reach `concat_where` and the flags are compile time, as the domains.
        granule._diffusion_global_step = granule._diffusion_global_step.with_compilation_options(
            static_params=("nrdmax", "num_levels", "apply_to_temperature", "apply_zdiffusion_t")
        )
    return granule, prognostic_input


def _diffusion_global_step(
    setup: dict[str, Any], execution: str, backend_like: model_backends.BackendLike
) -> tuple[Any, dict[str, gtx.Field]]:
    """The `DiffusionGlobal` step, jitted with `execution="jax"`, and its prognostic input."""
    granule, prognostic_input = _diffusion_global_granule(setup, execution, backend_like)
    dtime = setup["dtime"]

    def step(fields: dict[str, gtx.Field]) -> dict[str, gtx.Field]:
        new = granule.run(prognostic_state=prognostics.PrognosticState(**fields), dtime=dtime)
        return {k: getattr(new, k) for k in _PROGNOSTIC_FIELDS}

    if execution == "jax":
        jax = pytest.importorskip("jax")
        jitted = jax.jit(step)
        return lambda fields: jax.block_until_ready(jitted(fields)), prognostic_input
    allocator = model_backends.get_allocator(backend_like)

    def synced_step(fields: dict[str, gtx.Field]) -> dict[str, gtx.Field]:
        out = step(fields)
        device_utils.sync(allocator)
        return out

    return synced_step, prognostic_input


@pytest.mark.benchmark
@pytest.mark.benchmark_only
@pytest.mark.parametrize("execution", ["jax", "backend"])
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


def test_diffusion_global_jax_matches_backend(
    geometry_field_source: grid_geometry.GridGeometry,
    grid_manager: gm.GridManager,
    interpolation_field_source: interpolation_factory.InterpolationFieldsFactory,
    metrics_field_source: metrics_factory.MetricsFieldsFactory,
    backend_like: model_backends.BackendLike,
) -> None:
    """`DiffusionGlobal` under `jax.jit` against `DiffusionGlobal` and `Diffusion` on `--backend`."""
    setup = _setup(
        geometry_field_source,
        grid_manager,
        interpolation_field_source,
        metrics_field_source,
        backend_like,
    )
    jax_step, jax_input = _diffusion_global_step(setup, "jax", backend_like)
    backend_step, backend_input = _diffusion_global_step(setup, "backend", backend_like)
    computed = {k: v.asnumpy() for k, v in jax_step(jax_input).items()}
    references = {
        "global": {k: v.asnumpy() for k, v in backend_step(backend_input).items()},
    }
    granule = diffusion.Diffusion(
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
    granule.run(setup["diagnostic_state"], setup["prognostic_state"], setup["dtime"])
    references["granule"] = {
        k: getattr(setup["prognostic_state"], k).asnumpy() for k in _PROGNOSTIC_FIELDS
    }
    for reference_name, reference in references.items():
        for name in _PROGNOSTIC_FIELDS:
            diff = np.abs(computed[name] - reference[name])
            print(
                f"jax vs {reference_name} {name}: max abs {np.nanmax(diff):.3e}, "
                f"max |ref| {np.nanmax(np.abs(reference[name])):.3e}, "
                f"nan {int(np.isnan(computed[name]).sum())}/{int(np.isnan(reference[name]).sum())}"
            )
    for reference_name, reference in references.items():
        for name in _PROGNOSTIC_FIELDS:
            np.testing.assert_allclose(
                computed[name],
                reference[name],
                rtol=1e-10,
                atol=1e-12,
                equal_nan=True,
                err_msg=f"{name} against {reference_name}",
            )


_TORUS_SEED = 20260927
_DIFFUSION_TORUS_HALO = 4
_TORUS_PAIRS = (("reordered", "icon"), ("structured", "icon"), ("structured", "reordered"))


def _diffusion_torus_calls(  # noqa: PLR0917 [too-many-positional-arguments]
    geometry_field_source: grid_geometry.GridGeometry,
    grid_manager: gm.GridManager,
    interpolation_field_source: interpolation_factory.InterpolationFieldsFactory,
    metrics_field_source: metrics_factory.MetricsFieldsFactory,
    backend_like: model_backends.BackendLike,
    monkeypatch: pytest.MonkeyPatch,
    layouts: tuple[str, ...],
) -> dict[str, structured_torus.LayoutCall]:
    """
    The fused diffusion call in each torus layout, on the same random inputs drawn from `_TORUS_SEED`.

    The arguments are those `DiffusionGlobal.run` passes to the fused operator on `--backend`.
    """
    pytest.importorskip("jax")
    if not structured_torus.is_torus(grid_manager.file_path):
        pytest.skip("needs a torus grid file")
    rng = np.random.default_rng(_TORUS_SEED)
    with monkeypatch.context() as m:
        m.setattr(np.random, "default_rng", lambda *_: rng)
        setup = _setup(
            geometry_field_source,
            grid_manager,
            interpolation_field_source,
            metrics_field_source,
            backend_like,
        )
    granule, prognostic_input = _diffusion_global_granule(setup, "backend", backend_like)
    kwargs = structured_torus.record_call(
        granule,
        "_diffusion_global_step",
        lambda: granule.run(
            prognostic_state=prognostics.PrognosticState(**prognostic_input), dtime=setup["dtime"]
        ),
    )
    domain = kwargs.pop("domain")
    connectivities = kwargs.pop("offset_provider")
    layout = structured_torus.torus_layout(grid_manager.file_path)
    return {
        which: structured_torus.make_layout_call(
            diffusion_global_step._diffusion_global_step,
            kwargs,
            domain=domain,
            connectivities=connectivities,
            layout=layout,
            which=which,
            halo=_DIFFUSION_TORUS_HALO,
        )
        for which in layouts
    }


def test_diffusion_global_torus_layouts_match(  # noqa: PLR0917 [too-many-positional-arguments]
    geometry_field_source: grid_geometry.GridGeometry,
    grid_manager: gm.GridManager,
    interpolation_field_source: interpolation_factory.InterpolationFieldsFactory,
    metrics_field_source: metrics_factory.MetricsFieldsFactory,
    backend_like: model_backends.BackendLike,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fused diffusion call in the reordered and structured layouts against ICON order, after one call."""
    calls = _diffusion_torus_calls(
        geometry_field_source,
        grid_manager,
        interpolation_field_source,
        metrics_field_source,
        backend_like,
        monkeypatch,
        structured_torus.LAYOUTS,
    )
    structured_torus.check_layouts(
        calls,
        ("vn", "w", "theta_v", "exner"),
        rtol={pair: 1e-12 for pair in _TORUS_PAIRS},
    )


@pytest.mark.benchmark
@pytest.mark.benchmark_only
@pytest.mark.parametrize("layout", structured_torus.LAYOUTS)
def test_diffusion_global_torus_layout_benchmark(  # noqa: PLR0917 [too-many-positional-arguments]
    geometry_field_source: grid_geometry.GridGeometry,
    grid_manager: gm.GridManager,
    interpolation_field_source: interpolation_factory.InterpolationFieldsFactory,
    metrics_field_source: metrics_factory.MetricsFieldsFactory,
    backend_like: model_backends.BackendLike,
    monkeypatch: pytest.MonkeyPatch,
    layout: str,
    benchmark: Any,
) -> None:
    """The fused diffusion call under `jax.jit` in one torus layout; the first call is timed on its own."""
    call = _diffusion_torus_calls(
        geometry_field_source,
        grid_manager,
        interpolation_field_source,
        metrics_field_source,
        backend_like,
        monkeypatch,
        (layout,),
    )[layout]
    benchmark.extra_info["extent"] = call.extent
    start = time.perf_counter()
    call.run()
    benchmark.extra_info["first_call_s"] = time.perf_counter() - start
    benchmark(call.run)
