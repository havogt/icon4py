# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import functools
import os
import time
from typing import TYPE_CHECKING, Any

import gt4py.next as gtx
import numpy as np
import pytest
from gt4py.next import common as gtx_common


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

import icon4py.model.common.dimension as dims
import icon4py.model.common.grid.states as grid_states
from icon4py.model.atmosphere.dycore import (
    dycore_states,
    solve_nonhydro as solve_nh,
    solve_nonhydro_global,
)
from icon4py.model.atmosphere.dycore.stencils import solve_nonhydro_global_step
from icon4py.model.common import model_backends, model_options, utils as common_utils
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import (
    geometry as grid_geometry,
    geometry_attributes as geometry_meta,
    grid_manager as gm,
    icon as icon_grid,
    vertical as v_grid,
)
from icon4py.model.common.interpolation import interpolation_attributes, interpolation_factory
from icon4py.model.common.metrics import metrics_attributes, metrics_factory
from icon4py.model.common.states import factory, nonhydro_states, prognostic_state as prognostics
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
    interpolation_field_source: interpolation_factory.InterpolationFieldsFactory,
    metrics_field_source: metrics_factory.MetricsFieldsFactory,
    backend_like: model_backends.BackendLike,
) -> dict[str, Any]:
    allocator = model_backends.get_allocator(backend_like)
    mesh = geometry_field_source.grid

    config = solve_nh.NonHydrostaticConfig(
        divdamp_order=dycore_states.DivergenceDampingOrder.COMBINED,
        fourth_order_divdamp_factor=0.004,
    )

    nonhydro_params = solve_nh.NonHydrostaticParams(config)

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

    cell_geometry = grid_states.CellParams(
        cell_center_lat=geometry_field_source.get(geometry_meta.CELL_LAT),
        cell_center_lon=geometry_field_source.get(geometry_meta.CELL_LON),
        area=geometry_field_source.get(geometry_meta.CELL_AREA),
        mean_cell_area=geometry_field_source.get(geometry_meta.MEAN_CELL_AREA),  # type: ignore[arg-type]
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
    )

    interpolation_state = dycore_states.InterpolationState(
        c_lin_e=interpolation_field_source.get(interpolation_attributes.C_LIN_E),
        c_intp=interpolation_field_source.get(interpolation_attributes.CELL_AW_VERTS),
        e_flx_avg=interpolation_field_source.get(interpolation_attributes.E_FLX_AVG),
        geofac_grdiv=interpolation_field_source.get(interpolation_attributes.GEOFAC_GRDIV),
        geofac_rot=interpolation_field_source.get(interpolation_attributes.GEOFAC_ROT),
        pos_on_tplane_e_1=interpolation_field_source.get(
            interpolation_attributes.POS_ON_TPLANE_E_X
        ),
        pos_on_tplane_e_2=interpolation_field_source.get(
            interpolation_attributes.POS_ON_TPLANE_E_Y
        ),
        rbf_vec_coeff_e=interpolation_field_source.get(interpolation_attributes.RBF_VEC_COEFF_E),
        e_bln_c_s=interpolation_field_source.get(interpolation_attributes.E_BLN_C_S),
        rbf_coeff_1=interpolation_field_source.get(interpolation_attributes.RBF_VEC_COEFF_V1),
        rbf_coeff_2=interpolation_field_source.get(interpolation_attributes.RBF_VEC_COEFF_V2),
        geofac_div=interpolation_field_source.get(interpolation_attributes.GEOFAC_DIV),
        geofac_n2s=interpolation_field_source.get(interpolation_attributes.GEOFAC_N2S),
        geofac_grg_x=interpolation_field_source.get(interpolation_attributes.GEOFAC_GRG_X),
        geofac_grg_y=interpolation_field_source.get(interpolation_attributes.GEOFAC_GRG_Y),
        nudgecoeff_e=interpolation_field_source.get(interpolation_attributes.NUDGECOEFFS_E),
    )

    metric_state_nonhydro = dycore_states.MetricStateNonHydro(
        mask_prog_halo_c=metrics_field_source.get(metrics_attributes.MASK_PROG_HALO_C),
        rayleigh_w=metrics_field_source.get(metrics_attributes.RAYLEIGH_W),
        time_extrapolation_parameter_for_exner=metrics_field_source.get(
            metrics_attributes.EXNER_EXFAC
        ),
        reference_exner_at_cells_on_model_levels=metrics_field_source.get(
            metrics_attributes.EXNER_REF_MC
        ),
        wgtfac_c=metrics_field_source.get(metrics_attributes.WGTFAC_C),
        wgtfacq_c=metrics_field_source.get(metrics_attributes.WGTFACQ_C),
        inv_ddqz_z_full=metrics_field_source.get(metrics_attributes.INV_DDQZ_Z_FULL),
        reference_rho_at_cells_on_model_levels=metrics_field_source.get(
            metrics_attributes.RHO_REF_MC
        ),
        reference_theta_at_cells_on_model_levels=metrics_field_source.get(
            metrics_attributes.THETA_REF_MC
        ),
        exner_w_explicit_weight_parameter=metrics_field_source.get(
            metrics_attributes.EXNER_W_EXPLICIT_WEIGHT_PARAMETER
        ),
        ddz_of_reference_exner_at_cells_on_half_levels=metrics_field_source.get(
            metrics_attributes.D_EXNER_DZ_REF_IC
        ),
        ddqz_z_half=metrics_field_source.get(metrics_attributes.DDQZ_Z_HALF),
        reference_theta_at_cells_on_half_levels=metrics_field_source.get(
            metrics_attributes.THETA_REF_IC
        ),
        d2dexdz2_fac1_mc=metrics_field_source.get(metrics_attributes.D2DEXDZ2_FAC1_MC),
        d2dexdz2_fac2_mc=metrics_field_source.get(metrics_attributes.D2DEXDZ2_FAC1_MC),
        reference_rho_at_edges_on_model_levels=metrics_field_source.get(
            metrics_attributes.RHO_REF_ME
        ),
        reference_theta_at_edges_on_model_levels=metrics_field_source.get(
            metrics_attributes.THETA_REF_ME
        ),
        ddxn_z_full=metrics_field_source.get(metrics_attributes.DDXN_Z_FULL),
        zdiff_gradp=metrics_field_source.get(metrics_attributes.ZDIFF_GRADP),
        vertoffset_gradp=metrics_field_source.get(metrics_attributes.VERTOFFSET_GRADP),
        nflat_gradp=metrics_field_source.get_int32(metrics_attributes.NFLAT_GRADP),
        pg_exdist=metrics_field_source.get(metrics_attributes.PG_EXDIST_DSL),
        ddqz_z_full_e=metrics_field_source.get(metrics_attributes.DDQZ_Z_FULL_E),
        ddxt_z_full=metrics_field_source.get(metrics_attributes.DDXT_Z_FULL),
        wgtfac_e=metrics_field_source.get(metrics_attributes.WGTFAC_E),
        wgtfacq_e=metrics_field_source.get(metrics_attributes.WGTFACQ_E),
        exner_w_implicit_weight_parameter=metrics_field_source.get(
            metrics_attributes.EXNER_W_IMPLICIT_WEIGHT_PARAMETER
        ),
        horizontal_mask_for_3d_divdamp=metrics_field_source.get(
            metrics_attributes.HORIZONTAL_MASK_FOR_3D_DIVDAMP
        ),
        scaling_factor_for_3d_divdamp=metrics_field_source.get(
            metrics_attributes.SCALING_FACTOR_FOR_3D_DIVDAMP
        ),
        coeff1_dwdz=metrics_field_source.get(metrics_attributes.COEFF1_DWDZ),
        coeff2_dwdz=metrics_field_source.get(metrics_attributes.COEFF2_DWDZ),
        coeff_gradekin=metrics_field_source.get(metrics_attributes.COEFF_GRADEKIN),
    )

    return dict(
        mesh=mesh,
        config=config,
        params=nonhydro_params,
        metric_state_nonhydro=metric_state_nonhydro,
        interpolation_state=interpolation_state,
        vertical_params=vertical_grid,
        edge_geometry=edge_geometry,
        cell_geometry=cell_geometry,
        owner_mask=geometry_field_source.get("cell_owner_mask"),
    )


@pytest.fixture(scope="module")
def solve_nonhydro(
    geometry_field_source: grid_geometry.GridGeometry,
    interpolation_field_source: interpolation_factory.InterpolationFieldsFactory,
    metrics_field_source: metrics_factory.MetricsFieldsFactory,
    backend_like: model_backends.BackendLike,
) -> solve_nh.SolveNonhydro:
    setup = _setup(
        geometry_field_source, interpolation_field_source, metrics_field_source, backend_like
    )
    return solve_nh.SolveNonhydro(
        grid=setup.pop("mesh"),
        **setup,
        exchange=decomposition.SingleNodeExchange(),
        backend=backend_like,
        max_nudging_coefficient=0.375,
    )


def _states(
    mesh: Any, allocator: Any
) -> tuple[
    dycore_states.PrepAdvection,
    nonhydro_states.DiagnosticStateNonHydro,
    common_utils.TimeStepPair[prognostics.PrognosticState],
]:
    prep_adv = dycore_states.PrepAdvection(
        vn_traj=data_alloc.zero_field(mesh, dims.EdgeDim, dims.KDim, allocator=allocator),
        mass_flx_me=data_alloc.zero_field(mesh, dims.EdgeDim, dims.KDim, allocator=allocator),
        dynamical_vertical_mass_flux_at_cells_on_half_levels=data_alloc.zero_field(
            mesh, dims.CellDim, dims.KHalfDim, allocator=allocator
        ),
        dynamical_vertical_volumetric_flux_at_cells_on_half_levels=data_alloc.zero_field(
            mesh, dims.CellDim, dims.KHalfDim, allocator=allocator
        ),
    )

    diagnostic_state_nh = nonhydro_states.DiagnosticStateNonHydro(
        max_vertical_cfl=data_alloc.scalar_like_array(0.0, allocator),
        theta_v_at_cells_on_half_levels=data_alloc.zero_field(
            mesh, dims.CellDim, dims.KHalfDim, allocator=allocator
        ),
        perturbed_exner_at_cells_on_model_levels=data_alloc.zero_field(
            mesh, dims.CellDim, dims.KDim, allocator=allocator
        ),
        rho_at_cells_on_half_levels=data_alloc.zero_field(
            mesh, dims.CellDim, dims.KHalfDim, allocator=allocator
        ),
        exner_tendency_due_to_slow_physics=data_alloc.zero_field(
            mesh, dims.CellDim, dims.KDim, allocator=allocator
        ),
        grf_tend_rho=data_alloc.zero_field(mesh, dims.CellDim, dims.KDim, allocator=allocator),
        grf_tend_thv=data_alloc.zero_field(mesh, dims.CellDim, dims.KDim, allocator=allocator),
        grf_tend_w=data_alloc.zero_field(mesh, dims.CellDim, dims.KHalfDim, allocator=allocator),
        mass_flux_at_edges_on_model_levels=data_alloc.zero_field(
            mesh, dims.EdgeDim, dims.KDim, allocator=allocator
        ),
        normal_wind_tendency_due_to_slow_physics_process=data_alloc.zero_field(
            mesh, dims.EdgeDim, dims.KDim, allocator=allocator
        ),
        grf_tend_vn=data_alloc.zero_field(mesh, dims.EdgeDim, dims.KDim, allocator=allocator),
        normal_wind_advective_tendency=common_utils.PredictorCorrectorPair(
            data_alloc.zero_field(mesh, dims.EdgeDim, dims.KDim, allocator=allocator),
            data_alloc.zero_field(mesh, dims.EdgeDim, dims.KDim, allocator=allocator),
        ),
        vertical_wind_advective_tendency=common_utils.PredictorCorrectorPair(
            data_alloc.zero_field(mesh, dims.CellDim, dims.KHalfDim, allocator=allocator),
            data_alloc.zero_field(mesh, dims.CellDim, dims.KHalfDim, allocator=allocator),
        ),
        tangential_wind=data_alloc.zero_field(mesh, dims.EdgeDim, dims.KDim, allocator=allocator),
        vn_on_half_levels=data_alloc.zero_field(
            mesh, dims.EdgeDim, dims.KHalfDim, allocator=allocator
        ),
        contravariant_correction_at_cells_on_half_levels=data_alloc.zero_field(
            mesh, dims.CellDim, dims.KHalfDim, allocator=allocator
        ),
        rho_iau_increment=data_alloc.zero_field(mesh, dims.CellDim, dims.KDim, allocator=allocator),
        normal_wind_iau_increment=data_alloc.zero_field(
            mesh, dims.EdgeDim, dims.KDim, allocator=allocator
        ),
        exner_iau_increment=data_alloc.zero_field(
            mesh, dims.CellDim, dims.KDim, allocator=allocator
        ),
        exner_dynamical_increment=data_alloc.zero_field(
            mesh, dims.CellDim, dims.KDim, allocator=allocator
        ),
    )

    prognostic_state_nnow = prognostics.PrognosticState(
        w=data_alloc.random_field(mesh, dims.CellDim, dims.KHalfDim, allocator=allocator),
        vn=data_alloc.random_field(mesh, dims.EdgeDim, dims.KDim, allocator=allocator),
        theta_v=data_alloc.random_field(mesh, dims.CellDim, dims.KDim, allocator=allocator),
        rho=data_alloc.random_field(mesh, dims.CellDim, dims.KDim, allocator=allocator),
        exner=data_alloc.random_field(mesh, dims.CellDim, dims.KDim, allocator=allocator),
    )
    prognostic_state_nnew = prognostics.PrognosticState(
        w=data_alloc.random_field(mesh, dims.CellDim, dims.KHalfDim, allocator=allocator),
        vn=data_alloc.random_field(mesh, dims.EdgeDim, dims.KDim, allocator=allocator),
        theta_v=data_alloc.random_field(mesh, dims.CellDim, dims.KDim, allocator=allocator),
        rho=data_alloc.random_field(mesh, dims.CellDim, dims.KDim, allocator=allocator),
        exner=data_alloc.random_field(mesh, dims.CellDim, dims.KDim, allocator=allocator),
    )

    prognostic_states = common_utils.TimeStepPair(prognostic_state_nnow, prognostic_state_nnew)

    return prep_adv, diagnostic_state_nh, prognostic_states


@pytest.mark.parametrize(
    "at_first_substep, at_last_substep", [(True, False), (False, True), (False, False)]
)
@pytest.mark.benchmark
@pytest.mark.continuous_benchmarking
@pytest.mark.benchmark_only
def test_benchmark_solve_nonhydro(  # noqa: PLR0917 [too-many-positional-arguments]
    grid_manager: gm.GridManager,
    solve_nonhydro: solve_nh.SolveNonhydro,
    at_first_substep: bool,
    at_last_substep: bool,
    backend_like: model_backends.BackendLike,
    benchmark: Any,
) -> None:
    allocator = model_backends.get_allocator(backend_like)
    mesh = grid_manager.grid

    dtime = 10.0 if mesh.limited_area else 90.0

    prepare_fluxes_for_advection = True
    ndyn_substeps = 5
    at_initial_timestep = False
    second_order_divdamp_factor = 0.02

    prep_adv, diagnostic_state_nh, prognostic_states = _states(mesh, allocator)

    solve_nonhydro_timestep_variants = functools.partial(
        solve_nonhydro.time_step,
        diagnostic_state_nh=diagnostic_state_nh,
        prognostic_states=prognostic_states,
        prep_adv=prep_adv,
        second_order_divdamp_factor=second_order_divdamp_factor,
        dtime=dtime,
        ndyn_substeps_var=ndyn_substeps,
        at_initial_timestep=at_initial_timestep,
        prepare_fluxes_for_advection=prepare_fluxes_for_advection,
    )

    benchmark(
        solve_nonhydro_timestep_variants,
        at_first_substep=at_first_substep,
        at_last_substep=at_last_substep,
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
    if data_alloc.is_ndarray(obj):
        # 0-d arrays like `max_vertical_cfl` are allocated by the backend, e.g. as cupy arrays
        return jnp.asarray(data_alloc.as_numpy(obj))
    if isinstance(obj, tuple):
        return tuple(_to_jax(x, jnp) for x in obj)
    if isinstance(obj, common_utils.PredictorCorrectorPair):
        return common_utils.PredictorCorrectorPair(*(_to_jax(x, jnp) for x in obj))
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
_SUBSTEP_ARGS: dict[str, Any] = dict(
    second_order_divdamp_factor=0.02,
    ndyn_substeps_var=5,
    at_initial_timestep=False,
    prepare_fluxes_for_advection=True,
)


def _move_factory_fields_to_host(*sources: Any) -> None:
    """Replace the device fields cached by the factories with host copies and free the device pool."""
    for source in sources:
        for provider in {id(p): p for p in source._providers.values()}.values():
            fields = getattr(provider, "_fields", None)
            if not isinstance(fields, dict):
                continue
            for name, field in fields.items():
                if isinstance(field, gtx.Field) and not isinstance(field.ndarray, np.ndarray):
                    fields[name] = gtx.as_field(field.domain, field.asnumpy())
    try:
        import cupy  # noqa: PLC0415 [import-outside-top-level]
    except ImportError:
        return
    cupy.get_default_memory_pool().free_all_blocks()


def _solve_nonhydro_global_jax_step(
    geometry_field_source: grid_geometry.GridGeometry,
    interpolation_field_source: interpolation_factory.InterpolationFieldsFactory,
    metrics_field_source: metrics_factory.MetricsFieldsFactory,
    backend_like: model_backends.BackendLike,
    states: tuple[Any, Any, Any],
    *,
    at_first_substep: bool,
    at_last_substep: bool,
    free_backend_constants: bool = False,
) -> tuple[Any, dict[str, gtx.Field]]:
    """
    The jitted `SolveNonhydroGlobal` substep on JAX arrays on the default JAX device, and its
    prognostic input; the constant fields are computed with `backend_like` and then converted.
    """
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    setup = _setup(
        geometry_field_source, interpolation_field_source, metrics_field_source, backend_like
    )
    mesh = setup.pop("mesh")
    if mesh.limited_area:
        pytest.skip("'SolveNonhydroGlobal' does not support limited area grids.")
    vertical_params = setup.pop("vertical_params")
    setup.pop("owner_mask")
    jax_mesh = dataclasses.replace(
        mesh, connectivities={k: _to_jax(v, jnp) for k, v in mesh.connectivities.items()}
    )
    solver = solve_nonhydro_global.SolveNonhydroGlobal(
        grid=jax_mesh,
        vertical_params=v_grid.VerticalGrid(
            config=vertical_params.config,
            vct_a=_to_jax(vertical_params.vct_a, jnp),
            vct_b=_to_jax(vertical_params.vct_b, jnp),
        ),
        **{k: _to_jax(v, jnp) for k, v in setup.items()},
        allocator=jnp,
    )

    if free_backend_constants:
        # The factories keep the backend's copy of every constant alive for the session; on the
        # largest grids that copy and the JAX one do not fit on one GPU together.
        del setup
        _move_factory_fields_to_host(
            geometry_field_source, interpolation_field_source, metrics_field_source
        )
    prep_adv, diagnostic_state_nh, prognostic_states = states
    prep_adv, diagnostic_state_nh = _to_jax(prep_adv, jnp), _to_jax(diagnostic_state_nh, jnp)
    intermediate_state = solver.initial_intermediate_state()
    prognostic_input = {
        k: _to_jax(getattr(prognostic_states.current, k), jnp) for k in _PROGNOSTIC_FIELDS
    }
    dtime = 10.0 if mesh.limited_area else 90.0

    def step(fields: dict[str, gtx.Field]) -> dict[str, gtx.Field]:
        new, *_ = solver.time_step(
            diagnostic_state_nh=diagnostic_state_nh,
            prognostic_state=prognostics.PrognosticState(**fields),
            intermediate_state=intermediate_state,
            prep_adv=prep_adv,
            dtime=dtime,
            at_first_substep=at_first_substep,
            at_last_substep=at_last_substep,
            **_SUBSTEP_ARGS,
        )
        return {k: getattr(new, k) for k in _PROGNOSTIC_FIELDS}

    jitted = jax.jit(step)
    return lambda fields: jax.block_until_ready(jitted(fields)), prognostic_input


@pytest.mark.parametrize(
    "at_first_substep, at_last_substep", [(True, False), (False, True), (False, False)]
)
@pytest.mark.benchmark
@pytest.mark.benchmark_only
def test_benchmark_solve_nonhydro_global_jax(  # noqa: PLR0917 [too-many-positional-arguments]
    geometry_field_source: grid_geometry.GridGeometry,
    grid_manager: gm.GridManager,
    interpolation_field_source: interpolation_factory.InterpolationFieldsFactory,
    metrics_field_source: metrics_factory.MetricsFieldsFactory,
    at_first_substep: bool,
    at_last_substep: bool,
    backend_like: model_backends.BackendLike,
    benchmark: Any,
) -> None:
    """
    Benchmark `SolveNonhydroGlobal`, the fused single-field-operator substep, under `jax.jit`.

    The first call, which compiles, is timed on its own in `extra_info["first_call_s"]`.
    """
    step, prognostic_input = _solve_nonhydro_global_jax_step(
        geometry_field_source,
        interpolation_field_source,
        metrics_field_source,
        backend_like,
        _states(grid_manager.grid, model_backends.get_allocator(backend_like)),
        at_first_substep=at_first_substep,
        at_last_substep=at_last_substep,
        free_backend_constants=True,
    )
    start = time.perf_counter()
    step(prognostic_input)
    benchmark.extra_info["first_call_s"] = time.perf_counter() - start
    benchmark(step, prognostic_input)


#: The integer and boolean arguments of the fused step: level indices, options and flags.
_SOLVE_NONHYDRO_GLOBAL_STATIC_PARAMS = (
    "at_first_substep",
    "at_last_substep",
    "skip_compute_predictor_vertical_advection",
    "prepare_fluxes_for_advection",
    "apply_2nd_order_divergence_damping",
    "apply_4th_order_divergence_damping",
    "igradp_method",
    "rayleigh_type",
    "divdamp_type",
    "divdamp_order",
    "nlev",
    "nflatlev",
    "nflat_gradp",
    "start_of_ddz_of_exner_extrapolation",
    "start_of_d2dz2_of_exner_extrapolation",
    "end_index_of_damping_layer",
    "kstart_moist",
)


def _seeded_states(mesh: Any, allocator: Any, seed: int) -> Any:
    """`_states` with its random fields drawn from a generator seeded with `seed`."""
    rng = np.random.default_rng(seed)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(np.random, "default_rng", lambda *_, **__: rng)
        return _states(mesh, allocator)


def _solve_nonhydro_global_backend_solver(
    geometry_field_source: grid_geometry.GridGeometry,
    interpolation_field_source: interpolation_factory.InterpolationFieldsFactory,
    metrics_field_source: metrics_factory.MetricsFieldsFactory,
    backend_like: model_backends.BackendLike,
) -> tuple[solve_nonhydro_global.SolveNonhydroGlobal, icon_grid.IconGrid, Any]:
    """`SolveNonhydroGlobal` on `--backend`, with static domains and static integer and boolean arguments."""
    setup = _setup(
        geometry_field_source, interpolation_field_source, metrics_field_source, backend_like
    )
    mesh = setup.pop("mesh")
    if mesh.limited_area:
        pytest.skip("'SolveNonhydroGlobal' does not support limited area grids.")
    setup.pop("owner_mask")
    allocator = model_backends.get_allocator(backend_like)
    if os.environ.get("ICON4PY_BENCH_REPLACE_SKIP_VALUES", "1") == "1":
        # gt4py's static domain inference takes min/max over the neighbour tables including their
        # skip values (gt4py-f103).
        mesh = icon_grid.with_skip_values_replaced(mesh, allocator=allocator)
    backend = backend_like
    if os.environ.get("ICON4PY_BENCH_DACE_DISABLE_SPLITTING", "0") == "1":
        assert isinstance(backend_like, dict)
        backend = {**backend_like, "optimization_args": {"disable_splitting": True}}
    solver = solve_nonhydro_global.SolveNonhydroGlobal(
        grid=mesh,
        **setup,
        allocator=allocator,
        backend=model_options.customize_backend(None, backend),
    )
    options: dict[str, Any] = {}
    if os.environ.get("ICON4PY_BENCH_STATIC_DOMAINS", "1") == "1":
        options["static_domains"] = True
    if os.environ.get("ICON4PY_BENCH_STATIC_SIZES", "1") == "1":
        options["static_params"] = _SOLVE_NONHYDRO_GLOBAL_STATIC_PARAMS
    solver._solve_nonhydro_global_step = (
        solver._solve_nonhydro_global_step.with_compilation_options(**options)
    )
    return solver, mesh, allocator


@pytest.mark.parametrize(
    "at_first_substep, at_last_substep", [(True, False), (False, True), (False, False)]
)
@pytest.mark.benchmark
@pytest.mark.benchmark_only
def test_benchmark_solve_nonhydro_global_backend(  # noqa: PLR0917 [too-many-positional-arguments]
    geometry_field_source: grid_geometry.GridGeometry,
    grid_manager: gm.GridManager,
    interpolation_field_source: interpolation_factory.InterpolationFieldsFactory,
    metrics_field_source: metrics_factory.MetricsFieldsFactory,
    at_first_substep: bool,
    at_last_substep: bool,
    backend_like: model_backends.BackendLike,
    benchmark: Any,
) -> None:
    """
    Benchmark `SolveNonhydroGlobal`, the fused single-field-operator substep, on `--backend`, with
    static domains and static integer and boolean arguments.

    The first call, which compiles, is timed on its own in `extra_info["first_call_s"]`.
    """
    solver, mesh, allocator = _solve_nonhydro_global_backend_solver(
        geometry_field_source, interpolation_field_source, metrics_field_source, backend_like
    )
    prep_adv, diagnostic_state_nh, prognostic_states = _states(mesh, allocator)
    intermediate_state = solver.initial_intermediate_state()

    def step() -> None:
        solver.time_step(
            diagnostic_state_nh=diagnostic_state_nh,
            prognostic_state=prognostic_states.current,
            intermediate_state=intermediate_state,
            prep_adv=prep_adv,
            dtime=90.0,
            at_first_substep=at_first_substep,
            at_last_substep=at_last_substep,
            **_SUBSTEP_ARGS,
        )
        device_utils.sync(allocator)

    start = time.perf_counter()
    step()
    benchmark.extra_info["first_call_s"] = time.perf_counter() - start
    benchmark(step)


@pytest.mark.parametrize(
    "at_first_substep, at_last_substep", [(True, False), (False, True), (False, False)]
)
def test_solve_nonhydro_global_backend_matches_granule(  # noqa: PLR0917 [too-many-positional-arguments]
    geometry_field_source: grid_geometry.GridGeometry,
    grid_manager: gm.GridManager,
    interpolation_field_source: interpolation_factory.InterpolationFieldsFactory,
    metrics_field_source: metrics_factory.MetricsFieldsFactory,
    at_first_substep: bool,
    at_last_substep: bool,
    backend_like: model_backends.BackendLike,
) -> None:
    """`SolveNonhydroGlobal` on `--backend` against a fresh `SolveNonhydro` on `--backend`."""
    solver, mesh, allocator = _solve_nonhydro_global_backend_solver(
        geometry_field_source, interpolation_field_source, metrics_field_source, backend_like
    )
    # the states are random: both solvers must start from the same ones, in the allocator's layout
    prep_adv, diagnostic_state_nh, prognostic_states = _seeded_states(mesh, allocator, seed=0)
    reference_states = _seeded_states(mesh, allocator, seed=0)
    for name in _PROGNOSTIC_FIELDS:
        assert np.array_equal(
            getattr(prognostic_states.current, name).asnumpy(),
            getattr(reference_states[2].current, name).asnumpy(),
        ), name
    start = time.perf_counter()
    new, *_ = solver.time_step(
        diagnostic_state_nh=diagnostic_state_nh,
        prognostic_state=prognostic_states.current,
        intermediate_state=solver.initial_intermediate_state(),
        prep_adv=prep_adv,
        dtime=90.0,
        at_first_substep=at_first_substep,
        at_last_substep=at_last_substep,
        **_SUBSTEP_ARGS,
    )
    computed = {k: getattr(new, k).asnumpy() for k in _PROGNOSTIC_FIELDS}
    print(f"first call (compile and run) {time.perf_counter() - start:.1f} s")

    setup = _setup(
        geometry_field_source, interpolation_field_source, metrics_field_source, backend_like
    )
    granule = solve_nh.SolveNonhydro(
        grid=setup.pop("mesh"),
        **setup,
        exchange=decomposition.SingleNodeExchange(),
        backend=backend_like,
        max_nudging_coefficient=0.375,
    )
    prep_adv, diagnostic_state_nh, prognostic_states = reference_states
    granule.time_step(
        diagnostic_state_nh=diagnostic_state_nh,
        prognostic_states=prognostic_states,
        prep_adv=prep_adv,
        dtime=90.0,
        at_first_substep=at_first_substep,
        at_last_substep=at_last_substep,
        **_SUBSTEP_ARGS,
    )
    reference = {k: getattr(prognostic_states.next, k).asnumpy() for k in _PROGNOSTIC_FIELDS}
    for name in _PROGNOSTIC_FIELDS:
        diff = np.abs(computed[name] - reference[name])
        print(
            f"backend vs granule {name}: max abs {np.nanmax(diff):.3e}, "
            f"max |ref| {np.nanmax(np.abs(reference[name])):.3e}, "
            f"nan {int(np.isnan(computed[name]).sum())}/{int(np.isnan(reference[name]).sum())}"
        )
    for name in _PROGNOSTIC_FIELDS:
        np.testing.assert_allclose(
            computed[name], reference[name], rtol=1e-10, atol=1e-12, equal_nan=True, err_msg=name
        )


@pytest.mark.parametrize(
    "at_first_substep, at_last_substep", [(True, False), (False, True), (False, False)]
)
def test_solve_nonhydro_global_jax_matches_granule(  # noqa: PLR0917 [too-many-positional-arguments]
    geometry_field_source: grid_geometry.GridGeometry,
    grid_manager: gm.GridManager,
    interpolation_field_source: interpolation_factory.InterpolationFieldsFactory,
    metrics_field_source: metrics_factory.MetricsFieldsFactory,
    at_first_substep: bool,
    at_last_substep: bool,
    backend_like: model_backends.BackendLike,
) -> None:
    """`SolveNonhydroGlobal` under `jax.jit` against a fresh `SolveNonhydro` on `--backend`."""
    mesh = grid_manager.grid
    states = _states(mesh, model_backends.get_allocator(backend_like))
    step, prognostic_input = _solve_nonhydro_global_jax_step(
        geometry_field_source,
        interpolation_field_source,
        metrics_field_source,
        backend_like,
        states,
        at_first_substep=at_first_substep,
        at_last_substep=at_last_substep,
    )
    computed = {k: v.asnumpy() for k, v in step(prognostic_input).items()}

    setup = _setup(
        geometry_field_source, interpolation_field_source, metrics_field_source, backend_like
    )
    granule = solve_nh.SolveNonhydro(
        grid=setup.pop("mesh"),
        **setup,
        exchange=decomposition.SingleNodeExchange(),
        backend=backend_like,
        max_nudging_coefficient=0.375,
    )
    prep_adv, diagnostic_state_nh, prognostic_states = states
    granule.time_step(
        diagnostic_state_nh=diagnostic_state_nh,
        prognostic_states=prognostic_states,
        prep_adv=prep_adv,
        dtime=90.0,
        at_first_substep=at_first_substep,
        at_last_substep=at_last_substep,
        **_SUBSTEP_ARGS,
    )
    reference = {k: getattr(prognostic_states.next, k).asnumpy() for k in _PROGNOSTIC_FIELDS}
    for name in _PROGNOSTIC_FIELDS:
        diff = np.abs(computed[name] - reference[name])
        print(
            f"jax vs granule {name}: max abs {np.nanmax(diff):.3e}, "
            f"max |ref| {np.nanmax(np.abs(reference[name])):.3e}, "
            f"nan {int(np.isnan(computed[name]).sum())}/{int(np.isnan(reference[name]).sum())}"
        )
    for name in _PROGNOSTIC_FIELDS:
        np.testing.assert_allclose(
            computed[name], reference[name], rtol=1e-10, atol=1e-12, equal_nan=True, err_msg=name
        )


_TORUS_SEED = 20260927
_SOLVER_TORUS_HALO = 9


def _solve_nonhydro_torus_calls(  # noqa: PLR0917 [too-many-positional-arguments]
    geometry_field_source: grid_geometry.GridGeometry,
    grid_manager: gm.GridManager,
    interpolation_field_source: interpolation_factory.InterpolationFieldsFactory,
    metrics_field_source: metrics_factory.MetricsFieldsFactory,
    backend_like: model_backends.BackendLike,
    monkeypatch: pytest.MonkeyPatch,
    layouts: tuple[str, ...],
    *,
    at_first_substep: bool,
    at_last_substep: bool,
) -> dict[str, structured_torus.LayoutCall]:
    """
    The fused solve_nonhydro substep in each torus layout, on the same random states drawn from
    `_TORUS_SEED`.

    The arguments are those `SolveNonhydroGlobal.time_step` passes to the fused operator on
    `--backend`.
    """
    pytest.importorskip("jax")
    if not structured_torus.is_torus(grid_manager.file_path):
        pytest.skip("needs a torus grid file")
    mesh = grid_manager.grid
    allocator = model_backends.get_allocator(backend_like)
    setup = _setup(
        geometry_field_source, interpolation_field_source, metrics_field_source, backend_like
    )
    setup.pop("mesh")
    setup.pop("owner_mask")
    solver = solve_nonhydro_global.SolveNonhydroGlobal(
        grid=mesh,
        **setup,
        allocator=allocator,
        backend=model_options.customize_backend(None, backend_like),
    )
    rng = np.random.default_rng(_TORUS_SEED)
    with monkeypatch.context() as m:
        m.setattr(np.random, "default_rng", lambda *_: rng)
        prep_adv, diagnostic_state_nh, prognostic_states = _states(mesh, allocator)
    kwargs = structured_torus.record_call(
        solver,
        "_solve_nonhydro_global_step",
        lambda: solver.time_step(
            diagnostic_state_nh=diagnostic_state_nh,
            prognostic_state=prognostic_states.current,
            intermediate_state=solver.initial_intermediate_state(),
            prep_adv=prep_adv,
            dtime=90.0,
            at_first_substep=at_first_substep,
            at_last_substep=at_last_substep,
            **_SUBSTEP_ARGS,
        ),
    )
    domain = kwargs.pop("domain")
    connectivities = kwargs.pop("offset_provider")
    layout = structured_torus.torus_layout(grid_manager.file_path)
    return {
        which: structured_torus.make_layout_call(
            solve_nonhydro_global_step._solve_nonhydro_global_step,
            kwargs,
            domain=domain,
            connectivities=connectivities,
            layout=layout,
            which=which,
            halo=_SOLVER_TORUS_HALO,
        )
        for which in layouts
    }


@pytest.mark.parametrize(
    "at_first_substep, at_last_substep", [(True, False), (False, True), (False, False)]
)
def test_solve_nonhydro_global_torus_layouts_match(  # noqa: PLR0917 [too-many-positional-arguments]
    geometry_field_source: grid_geometry.GridGeometry,
    grid_manager: gm.GridManager,
    interpolation_field_source: interpolation_factory.InterpolationFieldsFactory,
    metrics_field_source: metrics_factory.MetricsFieldsFactory,
    backend_like: model_backends.BackendLike,
    monkeypatch: pytest.MonkeyPatch,
    at_first_substep: bool,
    at_last_substep: bool,
) -> None:
    """The fused substep in the reordered and structured layouts against ICON order, after one call."""
    calls = _solve_nonhydro_torus_calls(
        geometry_field_source,
        grid_manager,
        interpolation_field_source,
        metrics_field_source,
        backend_like,
        monkeypatch,
        structured_torus.LAYOUTS,
        at_first_substep=at_first_substep,
        at_last_substep=at_last_substep,
    )
    # The random states grow to 1e23 in one substep, which amplifies the rounding of the
    # canonicalised V2E/V2C sums, so only the two layouts sharing that slot order are held to rounding.
    structured_torus.check_layouts(
        calls,
        ("vn", "w", "rho", "exner", "theta_v"),
        rtol={("structured", "reordered"): 1e-12},
    )


@pytest.mark.parametrize(
    "at_first_substep, at_last_substep", [(True, False), (False, True), (False, False)]
)
@pytest.mark.parametrize("layout", structured_torus.LAYOUTS)
@pytest.mark.benchmark
@pytest.mark.benchmark_only
def test_solve_nonhydro_global_torus_layout_benchmark(  # noqa: PLR0917 [too-many-positional-arguments]
    geometry_field_source: grid_geometry.GridGeometry,
    grid_manager: gm.GridManager,
    interpolation_field_source: interpolation_factory.InterpolationFieldsFactory,
    metrics_field_source: metrics_factory.MetricsFieldsFactory,
    backend_like: model_backends.BackendLike,
    monkeypatch: pytest.MonkeyPatch,
    layout: str,
    at_first_substep: bool,
    at_last_substep: bool,
    benchmark: Any,
) -> None:
    """The fused substep under `jax.jit` in one torus layout; the first call is timed on its own."""
    call = _solve_nonhydro_torus_calls(
        geometry_field_source,
        grid_manager,
        interpolation_field_source,
        metrics_field_source,
        backend_like,
        monkeypatch,
        (layout,),
        at_first_substep=at_first_substep,
        at_last_substep=at_last_substep,
    )[layout]
    benchmark.extra_info["extent"] = call.extent
    start = time.perf_counter()
    call.run()
    benchmark.extra_info["first_call_s"] = time.perf_counter() - start
    benchmark(call.run)
