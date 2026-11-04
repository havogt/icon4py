# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""
One tmx step on a grid without lateral boundaries, as one field operator.

Embedded execution evaluates every expression on the largest domain its inputs allow, so a vertical
shift narrows the column. The two results that stay on a narrowed column, the w tendency and the
lowest level of `km`, `kh`, are completed with `concat_where` against a zero field spanning the full
column.
"""

import gt4py.next as gtx
from gt4py.next.experimental import concat_where

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.diagnostics_stencils import (
    _compute_shear_and_viscosity_diagnostics,
    _compute_thermodynamic_diagnostics,
    _interpolate_km,
    _interpolate_wind_to_vertices,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.energy_update_stencils import (
    _compute_end_of_step_diagnostics,
    _update_temperature,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.scalar_diffusion_stencils import (
    _assemble_scalar_diffusion_matrix,
    _compute_internal_energy_from_temperature,
    _diffuse_energy_and_update_temperature,
    _diffuse_tracer,
)
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.wind_diffusion_stencils import (
    _compute_vn_diffusion_tendency,
    _compute_w_diffusion_tendency_and_update_w,
    _compute_w_horizontal_stress_tendency,
    _interpolate_vn_tendency_to_cells_and_update_uv,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.interpolation.stencils.interpolate_cell_vector_to_edge_normal import (
    _interpolate_cell_vector_to_edge_normal,
)
from icon4py.model.common.physics.thermodynamics.compute_energy import (
    _compute_moist_air_isochoric_heat_capacity_per_area,
)
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _tmx_global_step(
    temperature: fa.CellKField[wpfloat],
    virtual_temperature: fa.CellKField[wpfloat],
    pressure: fa.CellKField[wpfloat],
    u: fa.CellKField[wpfloat],
    v: fa.CellKField[wpfloat],
    w: fa.CellKHalfField[wpfloat],
    qv: fa.CellKField[wpfloat],
    qc: fa.CellKField[wpfloat],
    qi: fa.CellKField[wpfloat],
    qr: fa.CellKField[wpfloat],
    qs: fa.CellKField[wpfloat],
    qg: fa.CellKField[wpfloat],
    rho: fa.CellKField[wpfloat],
    air_mass: fa.CellKField[wpfloat],
    evapotranspiration: fa.CellField[wpfloat],
    u_stress: fa.CellField[wpfloat],
    v_stress: fa.CellField[wpfloat],
    q_snocpymlt: fa.CellField[wpfloat],
    thermal_energy_flux: fa.CellField[wpfloat],
    vapor_energy_flux: fa.CellField[wpfloat],
    ddqz_z_full: fa.CellKField[wpfloat],
    inv_ddqz_z_full: fa.CellKField[wpfloat],
    inv_ddqz_z_half: fa.CellKHalfField[wpfloat],
    inv_ddqz_z_full_e: fa.EdgeKField[wpfloat],
    inv_ddqz_z_half_e: fa.EdgeKHalfField[wpfloat],
    inv_ddqz_z_half_v: fa.VertexKHalfField[wpfloat],
    wgtfac_c: fa.CellKHalfField[wpfloat],
    wgtfac_e: fa.EdgeKHalfField[wpfloat],
    wgtfacq_c: fa.CellKField[wpfloat],
    wgtfacq1_c: fa.CellKField[wpfloat],
    wgtfacq_e: fa.EdgeKField[wpfloat],
    wgtfacq1_e: fa.EdgeKField[wpfloat],
    height_above_ground: fa.CellKField[wpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    cells_aw_verts: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], wpfloat],
    rbf_coeff_v1: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], wpfloat],
    rbf_coeff_v2: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], wpfloat],
    rbf_coeff_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], wpfloat],
    rbf_coeff_c1: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2EDim], wpfloat],
    rbf_coeff_c2: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2EDim], wpfloat],
    primal_normal_cell_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    primal_normal_cell_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    primal_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    primal_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    dual_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    dual_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    edge_cell_length: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    inv_vert_vert_length: fa.EdgeField[wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    mixing_length_sq: fa.CellKHalfField[wpfloat],
    scaling_factor_louis: fa.CellField[wpfloat],
    fract_land: fa.CellField[wpfloat],
    fract_ice: fa.CellField[wpfloat],
    zero_full: fa.CellKField[wpfloat],
    zero_half: fa.CellKHalfField[wpfloat],
    rturb_prandtl: wpfloat,
    louis_constant_b: wpfloat,
    km_const: wpfloat,
    km_min: wpfloat,
    km_surface: wpfloat,
    kh_surface: wpfloat,
    energy_flux_factor: wpfloat,
    dissipation_factor: wpfloat,
    grav: wpfloat,
    dtime: wpfloat,
    use_km_const: bool,
    use_louis: bool,
    use_louis_land: bool,
    use_louis_ice: bool,
    num_levels: gtx.int32,
) -> tuple[
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKHalfField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKHalfField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
]:
    """
    Return the new temperature, qv, qc, qi, u, v and w, their tendencies, `km`, `kh`, the
    dissipated kinetic energy and the heating, and the running sums from the top of the dry static
    energy, the dissipated kinetic energy, the internal energy and its tendency, whose lowest level
    is the column integral.

    Below the lowest half level `km` and `kh` are `km_surface` and `kh_surface`.
    """
    nlev = num_levels

    # Compute_diagnostics
    cv_air = _compute_moist_air_isochoric_heat_capacity_per_area(qv, qc, qi, qr, qs, qg, air_mass)
    _, _, rho_ic, bruvais = _compute_thermodynamic_diagnostics(
        temperature=temperature,
        virtual_temperature=virtual_temperature,
        pressure=pressure,
        rho=rho,
        height_above_ground=height_above_ground,
        wgtfac_c=wgtfac_c,
        inv_ddqz_z_half=inv_ddqz_z_half,
        wgtfacq1_c=wgtfacq1_c,
        wgtfacq_c=wgtfacq_c,
        grav=grav,
        nlev=nlev,
    )
    vn = _interpolate_cell_vector_to_edge_normal(
        u, v, primal_normal_cell_x, primal_normal_cell_y, c_lin_e
    )
    w_vert, u_vert, v_vert = _interpolate_wind_to_vertices(
        w=w,
        vn=vn,
        cells_aw_verts=cells_aw_verts,
        rbf_coeff_v1=rbf_coeff_v1,
        rbf_coeff_v2=rbf_coeff_v2,
    )
    w_ie, _, _, _, _, div_c, _, km_ic, kh_ic = _compute_shear_and_viscosity_diagnostics(
        w=w,
        vn=vn,
        u_vert=u_vert,
        v_vert=v_vert,
        w_vert=w_vert,
        bruvais=bruvais,
        rho_ic=rho_ic,
        c_lin_e=c_lin_e,
        wgtfac_e=wgtfac_e,
        wgtfacq1_e=wgtfacq1_e,
        wgtfacq_e=wgtfacq_e,
        rbf_vec_coeff_e=rbf_coeff_e,
        primal_normal_vert_x=primal_normal_vert_x,
        primal_normal_vert_y=primal_normal_vert_y,
        dual_normal_vert_x=dual_normal_vert_x,
        dual_normal_vert_y=dual_normal_vert_y,
        tangent_orientation=tangent_orientation,
        inv_primal_edge_length=inv_primal_edge_length,
        inv_vert_vert_length=inv_vert_vert_length,
        inv_dual_edge_length=inv_dual_edge_length,
        inv_ddqz_z_full_e=inv_ddqz_z_full_e,
        e_bln_c_s=e_bln_c_s,
        wgtfac_c=wgtfac_c,
        mixing_length_sq=mixing_length_sq,
        scaling_factor_louis=scaling_factor_louis,
        fract_land=fract_land,
        fract_ice=fract_ice,
        rturb_prandtl=rturb_prandtl,
        louis_constant_b=louis_constant_b,
        km_const=km_const,
        use_km_const=use_km_const,
        use_louis=use_louis,
        use_louis_land=use_louis_land,
        use_louis_ice=use_louis_ice,
        nlev=nlev,
    )
    km_c, km_iv, km_ie = _interpolate_km(km_ic, cells_aw_verts, c_lin_e, km_min)

    # Compute_diffusion_hydrometeors
    a, b, c = _assemble_scalar_diffusion_matrix(
        kh_ic, inv_ddqz_z_half, air_mass, wpfloat("1.0"), 0, nlev - 1
    )
    no_surface_flux = wpfloat("0.0") * evapotranspiration
    new_qv, tend_qv = _diffuse_tracer(
        qv,
        a,
        b,
        c,
        evapotranspiration,
        air_mass,
        rho,
        km_ie,
        inv_dual_edge_length,
        geofac_div,
        rturb_prandtl,
        wpfloat("1.0"),
        dtime,
        0,
        nlev - 1,
    )
    new_qc, tend_qc = _diffuse_tracer(
        qc,
        a,
        b,
        c,
        no_surface_flux,
        air_mass,
        rho,
        km_ie,
        inv_dual_edge_length,
        geofac_div,
        rturb_prandtl,
        wpfloat("1.0"),
        dtime,
        0,
        nlev - 1,
    )
    new_qi, tend_qi = _diffuse_tracer(
        qi,
        a,
        b,
        c,
        no_surface_flux,
        air_mass,
        rho,
        km_ie,
        inv_dual_edge_length,
        geofac_div,
        rturb_prandtl,
        wpfloat("1.0"),
        dtime,
        0,
        nlev - 1,
    )

    # Compute_diffusion_temperature
    energy = _compute_internal_energy_from_temperature(
        temperature, qv, qc, qi, qr, qs, qg, height_above_ground, grav
    )
    _, heat_diffusion_tendency = _diffuse_energy_and_update_temperature(
        energy,
        kh_ic,
        inv_ddqz_z_half,
        thermal_energy_flux,
        vapor_energy_flux,
        temperature,
        new_qv,
        new_qc,
        new_qi,
        qr,
        qs,
        qg,
        air_mass,
        rho,
        km_ie,
        height_above_ground,
        inv_dual_edge_length,
        geofac_div,
        rturb_prandtl,
        energy_flux_factor,
        grav,
        dtime,
        0,
        nlev - 1,
    )

    # wind diffusion
    vn_tendency = _compute_vn_diffusion_tendency(
        rho=rho,
        w=w,
        vn=vn,
        u_vert=u_vert,
        v_vert=v_vert,
        km_c=km_c,
        div_c=div_c,
        km_iv=km_iv,
        km_ie=km_ie,
        u_stress=u_stress,
        v_stress=v_stress,
        inv_ddqz_z_full_e=inv_ddqz_z_full_e,
        inv_ddqz_z_half_e=inv_ddqz_z_half_e,
        c_lin_e=c_lin_e,
        primal_normal_cell_x=primal_normal_cell_x,
        primal_normal_cell_y=primal_normal_cell_y,
        primal_normal_vert_x=primal_normal_vert_x,
        primal_normal_vert_y=primal_normal_vert_y,
        dual_normal_vert_x=dual_normal_vert_x,
        dual_normal_vert_y=dual_normal_vert_y,
        tangent_orientation=tangent_orientation,
        inv_primal_edge_length=inv_primal_edge_length,
        inv_vert_vert_length=inv_vert_vert_length,
        inv_dual_edge_length=inv_dual_edge_length,
        dtime=dtime,
        vertical_start=0,
        vertical_end=nlev,
    )
    tend_u, tend_v, new_u, new_v = _interpolate_vn_tendency_to_cells_and_update_uv(
        vn_tendency, u, v, rbf_coeff_c1, rbf_coeff_c2, dtime
    )
    w_horizontal_stress_tendency = _compute_w_horizontal_stress_tendency(
        u=u,
        v=v,
        vn=vn,
        u_vert=u_vert,
        v_vert=v_vert,
        w_vert=w_vert,
        w_ie=w_ie,
        km_ic=km_ic,
        km_iv=km_iv,
        inv_ddqz_z_half=inv_ddqz_z_half,
        inv_ddqz_z_half_v=inv_ddqz_z_half_v,
        rbf_coeff_e=rbf_coeff_e,
        primal_normal_cell_x=primal_normal_cell_x,
        primal_normal_cell_y=primal_normal_cell_y,
        dual_normal_vert_x=dual_normal_vert_x,
        dual_normal_vert_y=dual_normal_vert_y,
        edge_cell_length=edge_cell_length,
        tangent_orientation=tangent_orientation,
        inv_primal_edge_length=inv_primal_edge_length,
        inv_vert_vert_length=inv_vert_vert_length,
        inv_dual_edge_length=inv_dual_edge_length,
    )
    interior_tend_w, _ = _compute_w_diffusion_tendency_and_update_w(
        w=w,
        rho_ic=rho_ic,
        km_c=km_c,
        div_c=div_c,
        horizontal_stress_tendency=w_horizontal_stress_tendency,
        inv_ddqz_z_half=inv_ddqz_z_half,
        inv_ddqz_z_full=inv_ddqz_z_full,
        e_bln_c_s=e_bln_c_s,
        dtime=dtime,
        vertical_start=1,
        vertical_end=nlev,
    )
    # w is not diffused on the top and the surface half level
    tend_w = concat_where((dims.KHalfDim >= 1) & (dims.KHalfDim < nlev), interior_tend_w, zero_half)
    new_w = w + tend_w * dtime

    # energy update and end-of-step diagnostics
    dissip_ke, heating, tend_temperature, new_temperature = _update_temperature(
        u=u,
        v=v,
        new_u=new_u,
        new_v=new_v,
        air_mass=air_mass,
        cv_air=cv_air,
        temperature=temperature,
        tend_temperature=heat_diffusion_tendency,
        q_snocpymlt=q_snocpymlt,
        dissipation_factor=dissipation_factor,
        dtime=dtime,
        nlev=nlev,
    )
    _, cptgz_vi, dissip_ke_vi, int_energy_vi, tend_int_energy_vi = _compute_end_of_step_diagnostics(
        temperature=temperature,
        new_temperature=new_temperature,
        dissip_ke=dissip_ke,
        qv=qv,
        qc=qc,
        qi=qi,
        new_qv=new_qv,
        new_qc=new_qc,
        new_qi=new_qi,
        qr=qr,
        qs=qs,
        qg=qg,
        rho=rho,
        ddqz_z_full=ddqz_z_full,
        height_above_ground=height_above_ground,
        grav=grav,
        dtime=dtime,
        vertical_start=0,
        vertical_end=nlev,
    )
    km = concat_where(dims.KDim < nlev - 1, km_ic(dims.KDim + 0.5), zero_full + km_surface)
    kh = concat_where(dims.KDim < nlev - 1, kh_ic(dims.KDim + 0.5), zero_full + kh_surface)

    return (
        new_temperature,
        new_qv,
        new_qc,
        new_qi,
        new_u,
        new_v,
        new_w,
        tend_temperature,
        tend_qv,
        tend_qc,
        tend_qi,
        tend_u,
        tend_v,
        tend_w,
        km,
        kh,
        dissip_ke,
        heating,
        cptgz_vi,
        dissip_ke_vi,
        int_energy_vi,
        tend_int_energy_vi,
    )
