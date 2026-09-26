# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import abs, astype, maximum, reduce  # noqa: A004
from gt4py.next.experimental import concat_where

from icon4py.model.atmosphere.dycore.dycore_utils import _compute_rayleigh_damping_factor
from icon4py.model.atmosphere.dycore.stencils.add_temporal_tendencies_to_vn import (
    _add_temporal_tendencies_to_vn,
)
from icon4py.model.atmosphere.dycore.stencils.compute_cell_diagnostics_for_dycore import (
    _compute_interpolation_and_nonhydro_buoy,
    _compute_perturbed_quantities_and_interpolation,
)
from icon4py.model.atmosphere.dycore.stencils.compute_edge_diagnostics_for_dycore_and_update_vn import (
    _apply_divergence_damping_and_update_vn,
    _compute_horizontal_pressure_gradient,
)
from icon4py.model.atmosphere.dycore.stencils.compute_horizontal_advection_of_rho_and_theta import (
    _compute_horizontal_advection_of_rho_and_theta,
)
from icon4py.model.atmosphere.dycore.stencils.compute_horizontal_velocity_quantities import (
    _compute_averaged_vn_and_fluxes,
    _compute_horizontal_velocity_quantities_and_fluxes,
)
from icon4py.model.atmosphere.dycore.stencils.compute_hydrostatic_correction_term import (
    _compute_hydrostatic_correction_term,
)
from icon4py.model.atmosphere.dycore.stencils.extrapolate_at_top import _extrapolate_at_top
from icon4py.model.atmosphere.dycore.stencils.velocity_advection_corrector import (
    _compute_velocity_advection_in_corrector_step,
)
from icon4py.model.atmosphere.dycore.stencils.velocity_advection_predictor import (
    _compute_velocity_advection_in_predictor_step,
)
from icon4py.model.atmosphere.dycore.stencils.vertically_implicit_dycore_solver import (
    _compute_thermodynamic_variables_at_corrector_step,
    _compute_thermodynamic_variables_at_predictor_step,
    _interpolate_contravariant_correction_from_edges_on_model_levels_to_cells_on_half_levels,
    _solve_w_and_update_vertical_fluxes_at_corrector_step,
    _solve_w_at_predictor_step,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.math.value_of_size import value_of_size_on_cells_on_half_levels_wp
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _plus(a: vpfloat, b: vpfloat) -> vpfloat:
    return a + b  # type: ignore[operator]


@gtx.field_operator
def _max(a: vpfloat, b: vpfloat) -> vpfloat:
    return maximum(a, b)


@gtx.field_operator
def _w_with_boundary_conditions(
    w: fa.CellKHalfField[wpfloat],
    contravariant_correction_at_cells_on_half_levels: fa.CellKHalfField[vpfloat],
    nlev: gtx.int32,
) -> fa.CellKHalfField[wpfloat]:
    """`w` with zero at the model top and the surface boundary condition at the bottom."""
    return concat_where(
        dims.KHalfDim < 1,
        value_of_size_on_cells_on_half_levels_wp(wpfloat("0.0"), w),
        concat_where(
            dims.KHalfDim < nlev,
            w,
            astype(contravariant_correction_at_cells_on_half_levels, wpfloat),
        ),
    )


@gtx.field_operator
def _solve_nonhydro_global_step(
    # prognostic state
    current_vn: fa.EdgeKField[wpfloat],
    current_w: fa.CellKHalfField[wpfloat],
    current_rho: fa.CellKField[wpfloat],
    current_exner: fa.CellKField[wpfloat],
    current_theta_v: fa.CellKField[wpfloat],
    # diagnostic state
    tangential_wind: fa.EdgeKField[vpfloat],
    vn_on_half_levels: fa.EdgeKHalfField[vpfloat],
    contravariant_correction_at_cells_on_half_levels: fa.CellKHalfField[vpfloat],
    theta_v_at_cells_on_half_levels: fa.CellKHalfField[wpfloat],
    perturbed_exner_at_cells_on_model_levels: fa.CellKField[wpfloat],
    rho_at_cells_on_half_levels: fa.CellKHalfField[wpfloat],
    exner_tendency_due_to_slow_physics: fa.CellKField[vpfloat],
    normal_wind_tendency_due_to_slow_physics_process: fa.EdgeKField[vpfloat],
    predictor_normal_wind_advective_tendency: fa.EdgeKField[vpfloat],
    predictor_vertical_wind_advective_tendency: fa.CellKHalfField[vpfloat],
    corrector_vertical_wind_advective_tendency: fa.CellKHalfField[vpfloat],
    exner_dynamical_increment: fa.CellKField[vpfloat],
    rho_iau_increment: fa.CellKField[vpfloat],
    normal_wind_iau_increment: fa.EdgeKField[vpfloat],
    exner_iau_increment: fa.CellKField[vpfloat],
    # intermediate state
    tangential_wind_on_half_levels: fa.EdgeKHalfField[vpfloat],
    contravariant_correction_at_edges_on_model_levels: fa.EdgeKField[vpfloat],
    # advection state
    vn_traj: fa.EdgeKField[wpfloat],
    mass_flx_me: fa.EdgeKField[wpfloat],
    dynamical_vertical_mass_flux_at_cells_on_half_levels: fa.CellKHalfField[wpfloat],
    dynamical_vertical_volumetric_flux_at_cells_on_half_levels: fa.CellKHalfField[wpfloat],
    # zeros where the program-based solver keeps zero-allocated locals
    zeros_cells_on_half_levels: fa.CellKHalfField[vpfloat],
    zeros_cells_on_model_levels: fa.CellKField[vpfloat],
    zeros_cells_on_model_levels_extended: fa.CellKField[vpfloat],
    # metric state
    rayleigh_w: fa.KHalfField[wpfloat],
    wgtfac_c: fa.CellKHalfField[vpfloat],
    wgtfacq_c: fa.CellKField[vpfloat],
    wgtfac_e: fa.EdgeKHalfField[vpfloat],
    wgtfacq_e: fa.EdgeKField[vpfloat],
    time_extrapolation_parameter_for_exner: fa.CellKField[vpfloat],
    reference_exner_at_cells_on_model_levels: fa.CellKField[vpfloat],
    reference_rho_at_cells_on_model_levels: fa.CellKField[vpfloat],
    reference_theta_at_cells_on_model_levels: fa.CellKField[vpfloat],
    reference_rho_at_edges_on_model_levels: fa.EdgeKField[vpfloat],
    reference_theta_at_edges_on_model_levels: fa.EdgeKField[vpfloat],
    reference_theta_at_cells_on_half_levels: fa.CellKHalfField[vpfloat],
    ddz_of_reference_exner_at_cells_on_half_levels: fa.CellKHalfField[vpfloat],
    ddqz_z_half: fa.CellKHalfField[vpfloat],
    d2dexdz2_fac1_mc: fa.CellKField[vpfloat],
    d2dexdz2_fac2_mc: fa.CellKField[vpfloat],
    ddxn_z_full: fa.EdgeKField[vpfloat],
    ddqz_z_full_e: fa.EdgeKField[vpfloat],
    ddxt_z_full: fa.EdgeKField[vpfloat],
    inv_ddqz_z_full: fa.CellKField[vpfloat],
    vertoffset_gradp: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim, dims.KDim], gtx.int32],
    zdiff_gradp: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim, dims.KDim], vpfloat],
    pg_exdist: fa.EdgeKField[vpfloat],
    exner_w_explicit_weight_parameter: fa.CellField[wpfloat],
    exner_w_implicit_weight_parameter: fa.CellField[wpfloat],
    horizontal_mask_for_3d_divdamp: fa.EdgeField[wpfloat],
    scaling_factor_for_3d_divdamp: fa.KField[wpfloat],
    coeff1_dwdz: fa.CellKField[vpfloat],
    coeff2_dwdz: fa.CellKField[vpfloat],
    coeff_gradekin: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], vpfloat],
    # interpolation state
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], wpfloat],
    geofac_grg_x: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], wpfloat],
    geofac_grg_y: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], wpfloat],
    nudgecoeff_e: fa.EdgeField[wpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    geofac_grdiv: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EODim], wpfloat],
    rbf_vec_coeff_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], wpfloat],
    c_intp: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], wpfloat],
    geofac_rot: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], wpfloat],
    pos_on_tplane_e_1: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    pos_on_tplane_e_2: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    e_flx_avg: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EODim], wpfloat],
    # geometry
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
    coriolis_frequency: fa.EdgeField[wpfloat],
    edge_area: fa.EdgeField[wpfloat],
    primal_normal_cell_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    primal_normal_cell_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    dual_normal_cell_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    dual_normal_cell_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    cell_area: fa.CellField[wpfloat],
    owner_mask: fa.CellField[bool],
    interpolated_fourth_order_divdamp_factor: fa.KField[wpfloat],
    # scalars
    dtime: wpfloat,
    second_order_divdamp_factor: wpfloat,
    second_order_divdamp_scaling_coeff: wpfloat,
    r_nsubsteps: wpfloat,
    ndyn_substeps_var: wpfloat,
    mean_cell_area: wpfloat,
    advection_explicit_weight_parameter: wpfloat,
    advection_implicit_weight_parameter: wpfloat,
    rhotheta_explicit_weight_parameter: wpfloat,
    rhotheta_implicit_weight_parameter: wpfloat,
    grav_o_cpd: wpfloat,
    dbl_eps: wpfloat,
    at_first_substep: bool,
    at_last_substep: bool,
    skip_compute_predictor_vertical_advection: bool,
    prepare_fluxes_for_advection: bool,
    apply_2nd_order_divergence_damping: bool,
    apply_4th_order_divergence_damping: bool,
    igradp_method: gtx.int32,
    rayleigh_type: gtx.int32,
    divdamp_type: gtx.int32,
    divdamp_order: gtx.int32,
    nlev: gtx.int32,
    nflatlev: gtx.int32,
    nflat_gradp: gtx.int32,
    start_of_ddz_of_exner_extrapolation: gtx.int32,
    start_of_d2dz2_of_exner_extrapolation: gtx.int32,
    end_index_of_damping_layer: gtx.int32,
    kstart_moist: gtx.int32,
) -> tuple[
    fa.EdgeKField[wpfloat],
    fa.CellKHalfField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.EdgeKField[vpfloat],
    fa.EdgeKHalfField[vpfloat],
    fa.CellKHalfField[vpfloat],
    fa.CellKHalfField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKHalfField[wpfloat],
    fa.EdgeKField[wpfloat],
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
    fa.CellKHalfField[vpfloat],
    fa.CellKHalfField[vpfloat],
    fa.CellKField[vpfloat],
    fa.EdgeKHalfField[vpfloat],
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[wpfloat],
    fa.EdgeKField[wpfloat],
    fa.CellKHalfField[wpfloat],
    fa.CellKHalfField[wpfloat],
    fa.CellField[vpfloat],
]:
    rayleigh_damping_factor = _compute_rayleigh_damping_factor(rayleigh_w, dtime)
    max_vertical_cfl = abs(astype(cell_area, vpfloat)) * vpfloat("0.0")

    # --- predictor ---
    if at_first_substep:
        (
            tangential_wind,
            new_tangential_wind_on_half_levels,
            vn_on_half_levels,
            _horizontal_kinetic_energy,
            new_contravariant_correction_at_edges,
            new_contravariant_correction_at_cells,
            new_predictor_w_advective_tendency,
            vertical_cfl,
            predictor_normal_wind_advective_tendency,
        ) = _compute_velocity_advection_in_predictor_step(
            tangential_wind_on_half_levels=tangential_wind_on_half_levels,
            vertical_wind_advective_tendency=predictor_vertical_wind_advective_tendency,
            vn=current_vn,
            w=current_w,
            rbf_vec_coeff_e=rbf_vec_coeff_e,
            wgtfac_e=wgtfac_e,
            wgtfacq_e=wgtfacq_e,
            ddxn_z_full=ddxn_z_full,
            ddxt_z_full=ddxt_z_full,
            coeff1_dwdz=coeff1_dwdz,
            coeff2_dwdz=coeff2_dwdz,
            c_intp=c_intp,
            inv_dual_edge_length=inv_dual_edge_length,
            inv_primal_edge_length=inv_primal_edge_length,
            tangent_orientation=tangent_orientation,
            e_bln_c_s=e_bln_c_s,
            wgtfac_c=wgtfac_c,
            ddqz_z_half=ddqz_z_half,
            area=cell_area,
            geofac_n2s=geofac_n2s,
            owner_mask=owner_mask,
            coriolis_frequency=coriolis_frequency,
            geofac_rot=geofac_rot,
            coeff_gradekin=coeff_gradekin,
            c_lin_e=c_lin_e,
            ddqz_z_full_e=ddqz_z_full_e,
            area_edge=edge_area,
            geofac_grdiv=geofac_grdiv,
            dtime=dtime,
            skip_compute_predictor_vertical_advection=skip_compute_predictor_vertical_advection,
            apply_extra_diffusion_on_vn=True,
            nflatlev=nflatlev,
            nlev=nlev,
            end_index_of_damping_layer=end_index_of_damping_layer,
        )
        tangential_wind_on_half_levels = concat_where(
            dims.KHalfDim < nlev,
            new_tangential_wind_on_half_levels,
            tangential_wind_on_half_levels,
        )
        contravariant_correction_at_edges_on_model_levels = concat_where(
            nflatlev <= dims.KDim,
            new_contravariant_correction_at_edges,
            contravariant_correction_at_edges_on_model_levels,
        )
        contravariant_correction_at_cells_on_half_levels = concat_where(
            dims.KHalfDim < nlev,
            new_contravariant_correction_at_cells,
            contravariant_correction_at_cells_on_half_levels,
        )
        predictor_vertical_wind_advective_tendency = concat_where(
            (1 <= dims.KHalfDim) & (dims.KHalfDim < nlev),
            new_predictor_w_advective_tendency,
            predictor_vertical_wind_advective_tendency,
        )
        max_vertical_cfl = reduce(_max, range=(dims.KHalfDim, 0, nlev))(abs(vertical_cfl))

    (
        perturbed_rho,
        perturbed_theta_v,
        perturbed_exner_at_cells_on_model_levels,
        new_rho_ic,
        new_theta_v_ic,
        new_nonhydro_buoy,
        new_exner_extrapolation,
        new_ddz_exner_extrapolation,
        new_d2dz2_exner_extrapolation,
    ) = _compute_perturbed_quantities_and_interpolation(
        time_extrapolation_parameter_for_exner=time_extrapolation_parameter_for_exner,
        current_exner=current_exner,
        reference_exner_at_cells_on_model_levels=reference_exner_at_cells_on_model_levels,
        current_rho=current_rho,
        reference_rho_at_cells_on_model_levels=reference_rho_at_cells_on_model_levels,
        current_theta_v=current_theta_v,
        reference_theta_at_cells_on_model_levels=reference_theta_at_cells_on_model_levels,
        wgtfac_c=wgtfac_c,
        exner_w_explicit_weight_parameter=exner_w_explicit_weight_parameter,
        perturbed_exner_at_cells_on_model_levels=perturbed_exner_at_cells_on_model_levels,
        ddz_of_reference_exner_at_cells_on_half_levels=ddz_of_reference_exner_at_cells_on_half_levels,
        ddqz_z_half=ddqz_z_half,
        wgtfacq_c=wgtfacq_c,
        reference_theta_at_cells_on_half_levels=reference_theta_at_cells_on_half_levels,
        inv_ddqz_z_full=inv_ddqz_z_full,
        d2dexdz2_fac1_mc=d2dexdz2_fac1_mc,
        d2dexdz2_fac2_mc=d2dexdz2_fac2_mc,
        ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels=zeros_cells_on_model_levels,
        d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels=zeros_cells_on_model_levels,
        igradp_method=igradp_method,
        surface_level=nlev + 1,
    )
    rho_ic = concat_where(
        (1 <= dims.KHalfDim) & (dims.KHalfDim < nlev), new_rho_ic, rho_at_cells_on_half_levels
    )
    theta_v_ic = concat_where(1 <= dims.KHalfDim, new_theta_v_ic, theta_v_at_cells_on_half_levels)
    nonhydro_buoy = concat_where(
        (1 <= dims.KHalfDim) & (dims.KHalfDim < nlev),
        new_nonhydro_buoy,
        zeros_cells_on_half_levels,
    )
    exner_extrapolation = concat_where(
        dims.KDim < nlev, new_exner_extrapolation, zeros_cells_on_model_levels_extended
    )
    ddz_exner_extrapolation = concat_where(
        start_of_ddz_of_exner_extrapolation <= dims.KDim,
        new_ddz_exner_extrapolation,
        zeros_cells_on_model_levels,
    )
    d2dz2_exner_extrapolation = concat_where(
        start_of_d2dz2_of_exner_extrapolation <= dims.KDim,
        new_d2dz2_exner_extrapolation,
        zeros_cells_on_model_levels,
    )

    hydrostatic_correction_on_lowest_level = reduce(_plus, range=(dims.KDim, nlev - 1, nlev))(
        _compute_hydrostatic_correction_term(
            theta_v=current_theta_v,
            ikoffset=vertoffset_gradp,
            zdiff_gradp=zdiff_gradp,
            theta_v_ic=theta_v_ic,
            inv_ddqz_z_full=inv_ddqz_z_full,
            inv_dual_edge_length=inv_dual_edge_length,
            grav_o_cpd=grav_o_cpd,
        )
    )

    rho_at_edges, theta_v_at_edges = _compute_horizontal_advection_of_rho_and_theta(
        p_vn=current_vn,
        p_vt=tangential_wind,
        pos_on_tplane_e_1=pos_on_tplane_e_1,
        pos_on_tplane_e_2=pos_on_tplane_e_2,
        primal_normal_cell_1=primal_normal_cell_x,
        dual_normal_cell_1=dual_normal_cell_x,
        primal_normal_cell_2=primal_normal_cell_y,
        dual_normal_cell_2=dual_normal_cell_y,
        p_dthalf=wpfloat("0.5") * dtime,
        rho_ref_me=reference_rho_at_edges_on_model_levels,
        theta_ref_me=reference_theta_at_edges_on_model_levels,
        perturbed_rho_at_cells_on_model_levels=perturbed_rho,
        perturbed_theta_v_at_cells_on_model_levels=perturbed_theta_v,
        geofac_grg_x=geofac_grg_x,
        geofac_grg_y=geofac_grg_y,
    )
    horizontal_pressure_gradient = _compute_horizontal_pressure_gradient(
        temporal_extrapolation_of_perturbed_exner=exner_extrapolation,
        ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels=ddz_exner_extrapolation,
        d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels=d2dz2_exner_extrapolation,
        hydrostatic_correction_on_lowest_level=hydrostatic_correction_on_lowest_level,
        ddxn_z_full=ddxn_z_full,
        c_lin_e=c_lin_e,
        ikoffset=vertoffset_gradp,
        zdiff_gradp=zdiff_gradp,
        pg_exdist=pg_exdist,
        inv_dual_edge_length=inv_dual_edge_length,
        nflatlev=nflatlev,
        nflat_gradp=nflat_gradp,
    )
    next_vn = _add_temporal_tendencies_to_vn(
        vn_nnow=current_vn,
        ddt_vn_apc_ntl1=predictor_normal_wind_advective_tendency,
        ddt_vn_phy=normal_wind_tendency_due_to_slow_physics_process,
        z_theta_v_e=theta_v_at_edges,
        z_gradh_exner=horizontal_pressure_gradient,
        dtime=dtime,
    )

    (
        _spatially_averaged_vn,
        horizontal_gradient_of_normal_wind_divergence,
        tangential_wind,
        mass_flux_at_edges,
        theta_v_flux_at_edges,
        new_vn_on_half_levels,
        new_tangential_wind_on_half_levels,
        horizontal_kinetic_energy,
        contravariant_correction_at_edges_on_model_levels,
    ) = _compute_horizontal_velocity_quantities_and_fluxes(
        contravariant_correction_at_edges_on_model_levels=contravariant_correction_at_edges_on_model_levels,
        vn=next_vn,
        e_flx_avg=e_flx_avg,
        geofac_grdiv=geofac_grdiv,
        rbf_vec_coeff_e=rbf_vec_coeff_e,
        rho_at_edges_on_model_levels=rho_at_edges,
        theta_v_at_edges_on_model_levels=theta_v_at_edges,
        ddqz_z_full_e=ddqz_z_full_e,
        ddxn_z_full=ddxn_z_full,
        ddxt_z_full=ddxt_z_full,
        wgtfac_e=wgtfac_e,
        nflatlev=nflatlev,
    )
    vn_on_half_levels = concat_where(
        dims.KHalfDim < nlev, new_vn_on_half_levels, _extrapolate_at_top(next_vn, wgtfacq_e)
    )
    tangential_wind_on_half_levels = concat_where(
        dims.KHalfDim < nlev, new_tangential_wind_on_half_levels, tangential_wind_on_half_levels
    )

    contravariant_correction_at_cells_on_half_levels = concat_where(
        nflatlev + 1 <= dims.KHalfDim,
        _interpolate_contravariant_correction_from_edges_on_model_levels_to_cells_on_half_levels(
            contravariant_correction_at_edges_on_model_levels=contravariant_correction_at_edges_on_model_levels,
            e_bln_c_s=e_bln_c_s,
            wgtfac_c=wgtfac_c,
            wgtfacq_c=wgtfacq_c,
            nlev=nlev,
        ),
        contravariant_correction_at_cells_on_half_levels,
    )
    w_boundary = _w_with_boundary_conditions(
        current_w, contravariant_correction_at_cells_on_half_levels, nlev
    )
    next_w = concat_where(
        (1 <= dims.KHalfDim) & (dims.KHalfDim < nlev),
        _solve_w_at_predictor_step(
            next_w=w_boundary,
            geofac_div=geofac_div,
            mass_flux_at_edges_on_model_levels=mass_flux_at_edges,
            theta_v_flux_at_edges_on_model_levels=theta_v_flux_at_edges,
            predictor_vertical_wind_advective_tendency=predictor_vertical_wind_advective_tendency,
            nonhydro_buoy_at_cells_on_half_levels=nonhydro_buoy,
            rho_at_cells_on_half_levels=rho_ic,
            contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
            exner_w_explicit_weight_parameter=exner_w_explicit_weight_parameter,
            current_exner=current_exner,
            current_rho=current_rho,
            current_theta_v=current_theta_v,
            current_w=current_w,
            inv_ddqz_z_full=inv_ddqz_z_full,
            exner_w_implicit_weight_parameter=exner_w_implicit_weight_parameter,
            theta_v_at_cells_on_half_levels=theta_v_ic,
            perturbed_exner_at_cells_on_model_levels=perturbed_exner_at_cells_on_model_levels,
            exner_tendency_due_to_slow_physics=exner_tendency_due_to_slow_physics,
            exner_iau_increment=exner_iau_increment,
            ddqz_z_half=ddqz_z_half,
            rayleigh_damping_factor=rayleigh_damping_factor,
            iau_wgt_dyn=wpfloat("0.0"),
            dtime=dtime,
            rayleigh_type=rayleigh_type,
            is_iau_active=False,
            end_index_of_damping_layer=end_index_of_damping_layer,
            n_lev=nlev,
        ),
        w_boundary,
    )
    (
        next_rho,
        next_exner,
        next_theta_v,
        dwdz_at_cells,
        exner_dynamical_increment,
    ) = _compute_thermodynamic_variables_at_predictor_step(
        next_w=next_w,
        geofac_div=geofac_div,
        mass_flux_at_edges_on_model_levels=mass_flux_at_edges,
        theta_v_flux_at_edges_on_model_levels=theta_v_flux_at_edges,
        rho_at_cells_on_half_levels=rho_ic,
        contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
        exner_w_explicit_weight_parameter=exner_w_explicit_weight_parameter,
        current_exner=current_exner,
        current_rho=current_rho,
        current_theta_v=current_theta_v,
        current_w=current_w,
        inv_ddqz_z_full=inv_ddqz_z_full,
        exner_w_implicit_weight_parameter=exner_w_implicit_weight_parameter,
        theta_v_at_cells_on_half_levels=theta_v_ic,
        perturbed_exner_at_cells_on_model_levels=perturbed_exner_at_cells_on_model_levels,
        exner_tendency_due_to_slow_physics=exner_tendency_due_to_slow_physics,
        rho_iau_increment=rho_iau_increment,
        exner_iau_increment=exner_iau_increment,
        exner_dynamical_increment=exner_dynamical_increment,
        dwdz_at_cells_on_model_levels=zeros_cells_on_model_levels,
        reference_exner_at_cells_on_model_levels=reference_exner_at_cells_on_model_levels,
        iau_wgt_dyn=wpfloat("0.0"),
        dtime=dtime,
        divdamp_type=divdamp_type,
        is_iau_active=False,
        at_first_substep=at_first_substep,
        kstart_moist=kstart_moist,
        n_lev=nlev,
    )

    # --- corrector ---
    (
        new_corrector_w_advective_tendency,
        vertical_cfl,
        corrector_normal_wind_advective_tendency,
    ) = _compute_velocity_advection_in_corrector_step(
        vn=next_vn,
        w=next_w,
        tangential_wind=tangential_wind,
        tangential_wind_on_half_levels=tangential_wind_on_half_levels,
        vn_on_half_levels=vn_on_half_levels,
        horizontal_kinetic_energy_at_edges_on_model_levels=horizontal_kinetic_energy,
        contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
        coeff1_dwdz=coeff1_dwdz,
        coeff2_dwdz=coeff2_dwdz,
        c_intp=c_intp,
        inv_dual_edge_length=inv_dual_edge_length,
        inv_primal_edge_length=inv_primal_edge_length,
        tangent_orientation=tangent_orientation,
        e_bln_c_s=e_bln_c_s,
        ddqz_z_half=ddqz_z_half,
        area=cell_area,
        geofac_n2s=geofac_n2s,
        owner_mask=owner_mask,
        coriolis_frequency=coriolis_frequency,
        geofac_rot=geofac_rot,
        coeff_gradekin=coeff_gradekin,
        c_lin_e=c_lin_e,
        ddqz_z_full_e=ddqz_z_full_e,
        area_edge=edge_area,
        geofac_grdiv=geofac_grdiv,
        dtime=dtime,
        apply_extra_diffusion_on_vn=True,
        nlev=nlev,
        end_index_of_damping_layer=end_index_of_damping_layer,
    )
    corrector_vertical_wind_advective_tendency = concat_where(
        (1 <= dims.KHalfDim) & (dims.KHalfDim < nlev),
        new_corrector_w_advective_tendency,
        corrector_vertical_wind_advective_tendency,
    )
    max_vertical_cfl = maximum(
        max_vertical_cfl, reduce(_max, range=(dims.KHalfDim, 0, nlev))(abs(vertical_cfl))
    )

    new_rho_ic, new_theta_v_ic, new_nonhydro_buoy = _compute_interpolation_and_nonhydro_buoy(
        w=next_w,
        contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
        current_rho=current_rho,
        next_rho=next_rho,
        current_theta_v=current_theta_v,
        next_theta_v=next_theta_v,
        perturbed_exner_at_cells_on_model_levels=perturbed_exner_at_cells_on_model_levels,
        reference_theta_at_cells_on_model_levels=reference_theta_at_cells_on_model_levels,
        ddz_of_reference_exner_at_cells_on_half_levels=ddz_of_reference_exner_at_cells_on_half_levels,
        ddqz_z_half=ddqz_z_half,
        wgtfac_c=wgtfac_c,
        exner_w_explicit_weight_parameter=exner_w_explicit_weight_parameter,
        dtime=dtime,
        rhotheta_explicit_weight_parameter=rhotheta_explicit_weight_parameter,
        rhotheta_implicit_weight_parameter=rhotheta_implicit_weight_parameter,
    )
    inner_half_levels = (1 <= dims.KHalfDim) & (dims.KHalfDim < nlev)
    rho_ic = concat_where(inner_half_levels, new_rho_ic, rho_ic)
    theta_v_ic = concat_where(inner_half_levels, new_theta_v_ic, theta_v_ic)
    nonhydro_buoy = concat_where(inner_half_levels, new_nonhydro_buoy, nonhydro_buoy)

    next_vn = _apply_divergence_damping_and_update_vn(
        horizontal_gradient_of_normal_wind_divergence=horizontal_gradient_of_normal_wind_divergence,
        next_vn=next_vn,
        current_vn=current_vn,
        dwdz_at_cells_on_model_levels=dwdz_at_cells,
        predictor_normal_wind_advective_tendency=predictor_normal_wind_advective_tendency,
        corrector_normal_wind_advective_tendency=corrector_normal_wind_advective_tendency,
        normal_wind_tendency_due_to_slow_physics_process=normal_wind_tendency_due_to_slow_physics_process,
        normal_wind_iau_increment=normal_wind_iau_increment,
        second_order_divdamp_scaling_coeff=second_order_divdamp_scaling_coeff,
        theta_v_at_edges_on_model_levels=theta_v_at_edges,
        horizontal_pressure_gradient=horizontal_pressure_gradient,
        horizontal_mask_for_3d_divdamp=horizontal_mask_for_3d_divdamp,
        scaling_factor_for_3d_divdamp=scaling_factor_for_3d_divdamp,
        inv_dual_edge_length=inv_dual_edge_length,
        nudgecoeff_e=nudgecoeff_e,
        geofac_grdiv=geofac_grdiv,
        interpolated_fourth_order_divdamp_factor=interpolated_fourth_order_divdamp_factor,
        advection_explicit_weight_parameter=advection_explicit_weight_parameter,
        advection_implicit_weight_parameter=advection_implicit_weight_parameter,
        dtime=dtime,
        is_iau_active=False,
        iau_wgt_dyn=wpfloat("0.0"),
        limited_area=False,
        apply_2nd_order_divergence_damping=apply_2nd_order_divergence_damping,
        apply_4th_order_divergence_damping=apply_4th_order_divergence_damping,
        divdamp_order=divdamp_order,
        mean_cell_area=mean_cell_area,
        second_order_divdamp_factor=second_order_divdamp_factor,
        max_nudging_coefficient=wpfloat("0.0"),
        dbl_eps=dbl_eps,
    )

    (
        _spatially_averaged_vn,
        mass_flux_at_edges,
        theta_v_flux_at_edges,
        vn_traj,
        mass_flx_me,
    ) = _compute_averaged_vn_and_fluxes(
        substep_and_spatially_averaged_vn=vn_traj,
        substep_averaged_mass_flux=mass_flx_me,
        e_flx_avg=e_flx_avg,
        vn=next_vn,
        rho_at_edges_on_model_levels=rho_at_edges,
        ddqz_z_full_e=ddqz_z_full_e,
        theta_v_at_edges_on_model_levels=theta_v_at_edges,
        prepare_fluxes_for_advection=prepare_fluxes_for_advection,
        at_first_substep=at_first_substep,
        r_nsubsteps=r_nsubsteps,
    )

    w_boundary = _w_with_boundary_conditions(
        next_w, contravariant_correction_at_cells_on_half_levels, nlev
    )
    (
        new_next_w,
        new_dynamical_mass_flux,
        new_dynamical_volumetric_flux,
    ) = _solve_w_and_update_vertical_fluxes_at_corrector_step(
        next_w=w_boundary,
        dynamical_vertical_mass_flux_at_cells_on_half_levels=dynamical_vertical_mass_flux_at_cells_on_half_levels,
        dynamical_vertical_volumetric_flux_at_cells_on_half_levels=dynamical_vertical_volumetric_flux_at_cells_on_half_levels,
        geofac_div=geofac_div,
        mass_flux_at_edges_on_model_levels=mass_flux_at_edges,
        theta_v_flux_at_edges_on_model_levels=theta_v_flux_at_edges,
        predictor_vertical_wind_advective_tendency=predictor_vertical_wind_advective_tendency,
        corrector_vertical_wind_advective_tendency=corrector_vertical_wind_advective_tendency,
        nonhydro_buoy_at_cells_on_half_levels=nonhydro_buoy,
        rho_at_cells_on_half_levels=rho_ic,
        contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
        exner_w_explicit_weight_parameter=exner_w_explicit_weight_parameter,
        current_exner=current_exner,
        current_rho=current_rho,
        current_theta_v=current_theta_v,
        current_w=current_w,
        inv_ddqz_z_full=inv_ddqz_z_full,
        exner_w_implicit_weight_parameter=exner_w_implicit_weight_parameter,
        theta_v_at_cells_on_half_levels=theta_v_ic,
        perturbed_exner_at_cells_on_model_levels=perturbed_exner_at_cells_on_model_levels,
        exner_tendency_due_to_slow_physics=exner_tendency_due_to_slow_physics,
        exner_iau_increment=exner_iau_increment,
        ddqz_z_half=ddqz_z_half,
        rayleigh_damping_factor=rayleigh_damping_factor,
        advection_explicit_weight_parameter=advection_explicit_weight_parameter,
        advection_implicit_weight_parameter=advection_implicit_weight_parameter,
        prepare_fluxes_for_advection=prepare_fluxes_for_advection,
        r_nsubsteps=r_nsubsteps,
        iau_wgt_dyn=wpfloat("0.0"),
        dtime=dtime,
        is_iau_active=False,
        rayleigh_type=rayleigh_type,
        at_first_substep=at_first_substep,
        end_index_of_damping_layer=end_index_of_damping_layer,
        n_lev=nlev,
    )
    next_w = concat_where(inner_half_levels, new_next_w, w_boundary)
    dynamical_vertical_mass_flux_at_cells_on_half_levels = concat_where(
        inner_half_levels,
        new_dynamical_mass_flux,
        dynamical_vertical_mass_flux_at_cells_on_half_levels,
    )
    dynamical_vertical_volumetric_flux_at_cells_on_half_levels = concat_where(
        inner_half_levels,
        new_dynamical_volumetric_flux,
        dynamical_vertical_volumetric_flux_at_cells_on_half_levels,
    )
    (
        next_rho,
        next_exner,
        next_theta_v,
        exner_dynamical_increment,
    ) = _compute_thermodynamic_variables_at_corrector_step(
        next_w=next_w,
        exner_dynamical_increment=exner_dynamical_increment,
        geofac_div=geofac_div,
        mass_flux_at_edges_on_model_levels=mass_flux_at_edges,
        theta_v_flux_at_edges_on_model_levels=theta_v_flux_at_edges,
        rho_at_cells_on_half_levels=rho_ic,
        contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
        exner_w_explicit_weight_parameter=exner_w_explicit_weight_parameter,
        current_exner=current_exner,
        current_rho=current_rho,
        current_theta_v=current_theta_v,
        current_w=current_w,
        inv_ddqz_z_full=inv_ddqz_z_full,
        exner_w_implicit_weight_parameter=exner_w_implicit_weight_parameter,
        theta_v_at_cells_on_half_levels=theta_v_ic,
        perturbed_exner_at_cells_on_model_levels=perturbed_exner_at_cells_on_model_levels,
        exner_tendency_due_to_slow_physics=exner_tendency_due_to_slow_physics,
        rho_iau_increment=rho_iau_increment,
        exner_iau_increment=exner_iau_increment,
        reference_exner_at_cells_on_model_levels=reference_exner_at_cells_on_model_levels,
        ndyn_substeps_var=ndyn_substeps_var,
        iau_wgt_dyn=wpfloat("0.0"),
        dtime=dtime,
        is_iau_active=False,
        at_last_substep=at_last_substep,
        kstart_moist=kstart_moist,
        n_lev=nlev,
    )

    return (
        next_vn,
        next_w,
        next_rho,
        next_exner,
        next_theta_v,
        tangential_wind,
        vn_on_half_levels,
        contravariant_correction_at_cells_on_half_levels,
        theta_v_ic,
        perturbed_exner_at_cells_on_model_levels,
        rho_ic,
        mass_flux_at_edges,
        predictor_normal_wind_advective_tendency,
        corrector_normal_wind_advective_tendency,
        predictor_vertical_wind_advective_tendency,
        corrector_vertical_wind_advective_tendency,
        exner_dynamical_increment,
        tangential_wind_on_half_levels,
        contravariant_correction_at_edges_on_model_levels,
        vn_traj,
        mass_flx_me,
        dynamical_vertical_mass_flux_at_cells_on_half_levels,
        dynamical_vertical_volumetric_flux_at_cells_on_half_levels,
        max_vertical_cfl,
    )
