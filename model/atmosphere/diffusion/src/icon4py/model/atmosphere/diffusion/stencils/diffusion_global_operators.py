# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""
The diffusion field operators that `diffusion_global_step` is composed of, as they were before the
diffusion granule was refactored into `apply_diffusion_to_prognostic_fields`.

Keeping them apart from the granule's makes the granule an independent witness of the global step.
"""

import gt4py.next as gtx
from gt4py.next import astype, max_over, maximum, minimum, neighbor_sum, sqrt, where
from gt4py.next.experimental import as_offset

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import C2E, C2E2C, C2E2CO, E2C, E2C2V, Koff
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _calculate_nabla2_and_smag_coefficients_for_vn(
    diff_multfac_smag: gtx.Field[gtx.Dims[dims.KDim], vpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    inv_vert_vert_length: fa.EdgeField[wpfloat],
    u_vert: fa.VertexKField[vpfloat],
    v_vert: fa.VertexKField[vpfloat],
    primal_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    primal_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    dual_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    dual_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    vn: fa.EdgeKField[wpfloat],
    smag_limit: gtx.Field[gtx.Dims[dims.KDim], vpfloat],
    smag_offset: vpfloat,
) -> tuple[
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[wpfloat],
]:
    diff_multfac_smag_wp, u_vert_wp, v_vert_wp, smag_offset_wp = astype(
        (diff_multfac_smag, u_vert, v_vert, smag_offset), wpfloat
    )

    v_n = u_vert_wp(E2C2V) * primal_normal_vert_x + v_vert_wp(E2C2V) * primal_normal_vert_y

    nabla2_of_vn = (v_n[dims.E2C2VDim(0)] + v_n[dims.E2C2VDim(1)] - wpfloat("2.0") * vn) * (
        inv_primal_edge_length * inv_primal_edge_length
    ) + (v_n[dims.E2C2VDim(2)] + v_n[dims.E2C2VDim(3)] - wpfloat("2.0") * vn) * (
        inv_vert_vert_length * inv_vert_vert_length
    )
    # The factor of 4 comes from the lengths in the denominator being twice those needed
    # for the diffusion stencil (https://doi.org/10.1002%2Fqj.2378).
    nabla2_of_vn = wpfloat("4.0") * nabla2_of_vn

    v_t = u_vert_wp(E2C2V) * dual_normal_vert_x + v_vert_wp(E2C2V) * dual_normal_vert_y
    l_p = tangent_orientation * inv_primal_edge_length
    l_vv = inv_vert_vert_length

    kh_smag_part1 = (v_n[dims.E2C2VDim(3)] - v_n[dims.E2C2VDim(2)]) * l_vv - (
        v_t[dims.E2C2VDim(1)] - v_t[dims.E2C2VDim(0)]
    ) * l_p
    kh_smag_part2 = (v_n[dims.E2C2VDim(1)] - v_n[dims.E2C2VDim(0)]) * l_p + (
        v_t[dims.E2C2VDim(3)] - v_t[dims.E2C2VDim(2)]
    ) * l_vv

    kh_smag_wp = diff_multfac_smag_wp * sqrt(
        kh_smag_part1 * kh_smag_part1 + kh_smag_part2 * kh_smag_part2
    )

    kh_smag_vp_for_diffusion = maximum(vpfloat("0.0"), astype(kh_smag_wp - smag_offset_wp, vpfloat))
    kh_smag_vp_for_diffusion = minimum(kh_smag_vp_for_diffusion, smag_limit)

    return kh_smag_vp_for_diffusion, astype(kh_smag_wp, vpfloat), nabla2_of_vn


@gtx.field_operator
def _temporary_field_for_grid_point_cold_pools_enhancement(
    theta_v: fa.CellKField[wpfloat],
    theta_ref_mc: fa.CellKField[vpfloat],
    thresh_tdiff: wpfloat,
    smallest_vpfloat: vpfloat,
) -> fa.CellKField[vpfloat]:
    theta_ref_mc_wp = astype(theta_ref_mc, wpfloat)

    tdiff = theta_v - neighbor_sum(theta_v(C2E2C), axis=dims.C2E2CDim) / wpfloat("3.0")
    trefdiff = theta_ref_mc_wp - astype(
        neighbor_sum(theta_ref_mc(C2E2C), axis=dims.C2E2CDim), wpfloat
    ) / wpfloat("3.0")
    enh_diffu_3d_vp = where(
        ((tdiff - trefdiff) < thresh_tdiff) & (trefdiff < wpfloat("0.0"))
        | (tdiff - trefdiff < wpfloat("1.5") * thresh_tdiff),
        astype((thresh_tdiff - tdiff + trefdiff) * wpfloat("5.0e-4"), vpfloat),
        smallest_vpfloat,
    )
    return enh_diffu_3d_vp


@gtx.field_operator
def _enhance_diffusion_coefficient_for_grid_point_cold_pools(
    kh_smag_e: fa.EdgeKField[vpfloat],
    enh_diffu_3d: fa.CellKField[vpfloat],
) -> fa.EdgeKField[vpfloat]:
    kh_smag_e_vp = maximum(kh_smag_e, max_over(enh_diffu_3d(E2C), axis=dims.E2CDim))
    return kh_smag_e_vp


@gtx.field_operator
def _calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools(
    theta_v: fa.CellKField[wpfloat],
    theta_ref_mc: fa.CellKField[vpfloat],
    thresh_tdiff: wpfloat,
    smallest_vpfloat: vpfloat,
    kh_smag_e: fa.EdgeKField[vpfloat],
) -> fa.EdgeKField[vpfloat]:
    enh_diffu_3d = _temporary_field_for_grid_point_cold_pools_enhancement(
        theta_v,
        theta_ref_mc,
        thresh_tdiff,
        smallest_vpfloat,
    )
    kh_smag_e_vp = _enhance_diffusion_coefficient_for_grid_point_cold_pools(kh_smag_e, enh_diffu_3d)
    return kh_smag_e_vp


@gtx.field_operator
def _calculate_nabla4(
    u_vert: fa.VertexKField[vpfloat],
    v_vert: fa.VertexKField[vpfloat],
    primal_normal_vert_v1: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    primal_normal_vert_v2: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    z_nabla2_e: fa.EdgeKField[wpfloat],
    inv_vert_vert_length: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
) -> fa.EdgeKField[vpfloat]:
    u_vert_wp, v_vert_wp = astype((u_vert, v_vert), wpfloat)

    nabv_tang_vp = astype(
        (
            u_vert_wp(E2C2V[0]) * primal_normal_vert_v1[dims.E2C2VDim(0)]
            + v_vert_wp(E2C2V[0]) * primal_normal_vert_v2[dims.E2C2VDim(0)]
            + u_vert_wp(E2C2V[1]) * primal_normal_vert_v1[dims.E2C2VDim(1)]
            + v_vert_wp(E2C2V[1]) * primal_normal_vert_v2[dims.E2C2VDim(1)]
        ),
        vpfloat,
    )

    nabv_norm_vp = astype(
        (
            u_vert_wp(E2C2V[2]) * primal_normal_vert_v1[dims.E2C2VDim(2)]
            + v_vert_wp(E2C2V[2]) * primal_normal_vert_v2[dims.E2C2VDim(2)]
            + u_vert_wp(E2C2V[3]) * primal_normal_vert_v1[dims.E2C2VDim(3)]
            + v_vert_wp(E2C2V[3]) * primal_normal_vert_v2[dims.E2C2VDim(3)]
        ),
        vpfloat,
    )
    nabv_tang_wp, nabv_norm_wp = astype((nabv_tang_vp, nabv_norm_vp), wpfloat)
    z_nabla4_e2_wp = wpfloat("4.0") * (
        (nabv_norm_wp - wpfloat("2.0") * z_nabla2_e) * (inv_vert_vert_length * inv_vert_vert_length)
        + (nabv_tang_wp - wpfloat("2.0") * z_nabla2_e)
        * (inv_primal_edge_length * inv_primal_edge_length)
    )
    return astype(z_nabla4_e2_wp, vpfloat)


@gtx.field_operator
def _apply_nabla2_and_nabla4_global_to_vn(
    area_edge: fa.EdgeField[wpfloat],
    kh_smag_e: fa.EdgeKField[vpfloat],
    z_nabla2_e: fa.EdgeKField[wpfloat],
    z_nabla4_e2: fa.EdgeKField[vpfloat],
    diff_multfac_vn: fa.KField[wpfloat],
    vn: fa.EdgeKField[wpfloat],
) -> fa.EdgeKField[wpfloat]:
    kh_smag_e_wp, z_nabla4_e2_wp = astype((kh_smag_e, z_nabla4_e2), wpfloat)

    vn_wp = vn + area_edge * (
        kh_smag_e_wp * z_nabla2_e - diff_multfac_vn * z_nabla4_e2_wp * area_edge
    )
    return vn_wp


@gtx.field_operator
def _calculate_nabla2_for_w(
    w: fa.CellKHalfField[wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], wpfloat],
) -> fa.CellKHalfField[vpfloat]:
    z_nabla2_c_wp = neighbor_sum(w(C2E2CO) * geofac_n2s, axis=dims.C2E2CODim)
    return astype(z_nabla2_c_wp, vpfloat)


@gtx.field_operator
def _apply_nabla2_to_w(
    area: fa.CellField[wpfloat],
    z_nabla2_c: fa.CellKHalfField[vpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], wpfloat],
    w: fa.CellKHalfField[wpfloat],
    diff_multfac_w: wpfloat,
) -> fa.CellKHalfField[wpfloat]:
    z_nabla2_c_wp = astype(z_nabla2_c, wpfloat)

    w_wp = w - diff_multfac_w * (area * area) * neighbor_sum(
        z_nabla2_c_wp(C2E2CO) * geofac_n2s, axis=dims.C2E2CODim
    )
    return w_wp


@gtx.field_operator
def _apply_nabla2_to_w_in_upper_damping_layer(
    w: fa.CellKHalfField[wpfloat],
    diff_multfac_n2w: fa.KHalfField[wpfloat],
    cell_area: fa.CellField[wpfloat],
    z_nabla2_c: fa.CellKHalfField[vpfloat],
) -> fa.CellKHalfField[wpfloat]:
    z_nabla2_c_wp = astype(z_nabla2_c, wpfloat)
    w_wp = w + diff_multfac_n2w * cell_area * z_nabla2_c_wp
    return w_wp


@gtx.field_operator
def _calculate_nabla2_for_z(
    kh_smag_e: fa.EdgeKField[vpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    theta_v: fa.CellKField[wpfloat],
) -> fa.EdgeKField[wpfloat]:
    kh_smag_e_wp = astype(kh_smag_e, wpfloat)

    z_nabla2_e_wp = kh_smag_e_wp * inv_dual_edge_length * (theta_v(E2C[1]) - theta_v(E2C[0]))
    return z_nabla2_e_wp


@gtx.field_operator
def _calculate_nabla2_of_theta(
    z_nabla2_e: fa.EdgeKField[wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
) -> fa.CellKField[vpfloat]:
    z_temp_wp = neighbor_sum(z_nabla2_e(C2E) * geofac_div, axis=dims.C2EDim)
    return astype(z_temp_wp, vpfloat)


@gtx.field_operator
def _truly_horizontal_diffusion_nabla_of_theta_over_steep_points(
    zd_vertoffset: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim, dims.KDim], gtx.int32],
    zd_diffcoef: fa.CellKField[wpfloat],
    geofac_n2s_c: fa.CellField[wpfloat],
    geofac_n2s_nbh: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], wpfloat],
    vcoef: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim, dims.KDim], wpfloat],
    theta_v: fa.CellKField[wpfloat],
    z_temp: fa.CellKField[vpfloat],
) -> fa.CellKField[vpfloat]:
    z_temp_wp = astype(z_temp, wpfloat)

    theta_v_0 = theta_v(C2E2C[0])(as_offset(Koff, zd_vertoffset[dims.C2E2CDim(0)]))
    theta_v_1 = theta_v(C2E2C[1])(as_offset(Koff, zd_vertoffset[dims.C2E2CDim(1)]))
    theta_v_2 = theta_v(C2E2C[2])(as_offset(Koff, zd_vertoffset[dims.C2E2CDim(2)]))

    theta_v_0_m1 = theta_v(C2E2C[0])(as_offset(Koff, zd_vertoffset[dims.C2E2CDim(0)] + 1))
    theta_v_1_m1 = theta_v(C2E2C[1])(as_offset(Koff, zd_vertoffset[dims.C2E2CDim(1)] + 1))
    theta_v_2_m1 = theta_v(C2E2C[2])(as_offset(Koff, zd_vertoffset[dims.C2E2CDim(2)] + 1))

    sum_tmp = (
        theta_v * geofac_n2s_c
        + geofac_n2s_nbh[dims.C2E2CDim(0)]
        * (
            vcoef[dims.C2E2CDim(0)] * theta_v_0
            + (wpfloat("1.0") - vcoef[dims.C2E2CDim(0)]) * theta_v_0_m1
        )
        + geofac_n2s_nbh[dims.C2E2CDim(1)]
        * (
            vcoef[dims.C2E2CDim(1)] * theta_v_1
            + (wpfloat("1.0") - vcoef[dims.C2E2CDim(1)]) * theta_v_1_m1
        )
        + geofac_n2s_nbh[dims.C2E2CDim(2)]
        * (
            vcoef[dims.C2E2CDim(2)] * theta_v_2
            + (wpfloat("1.0") - vcoef[dims.C2E2CDim(2)]) * theta_v_2_m1
        )
    )

    # Note: In the original Fortran code `zd_diffcoef` is implemented as a list,
    # in ICON4Py it's a full field intialized with zeros for points that are not in the list.
    z_temp_wp = z_temp_wp + zd_diffcoef * sum_tmp

    return astype(z_temp_wp, vpfloat)


@gtx.field_operator
def _update_theta_and_exner(
    z_temp: fa.CellKField[vpfloat],
    area: fa.CellField[wpfloat],
    theta_v: fa.CellKField[wpfloat],
    exner: fa.CellKField[wpfloat],
    rd_o_cvd: vpfloat,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    rd_o_cvd_wp, z_temp_wp = astype((rd_o_cvd, z_temp), wpfloat)

    z_theta = theta_v
    theta_v = theta_v + (area * z_temp_wp)
    exner = exner * (wpfloat("1.0") + rd_o_cvd_wp * (theta_v / z_theta - wpfloat("1.0")))
    return theta_v, exner


@gtx.field_operator
def _apply_diffusion_to_theta_and_exner(
    kh_smag_e: fa.EdgeKField[vpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    theta_v: fa.CellKField[wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    zd_vertoffset: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim, dims.KDim], gtx.int32],
    zd_diffcoef: fa.CellKField[wpfloat],
    geofac_n2s_c: fa.CellField[wpfloat],
    geofac_n2s_nbh: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], wpfloat],
    vcoef: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim, dims.KDim], wpfloat],
    area: fa.CellField[wpfloat],
    exner: fa.CellKField[wpfloat],
    rd_o_cvd: vpfloat,
    apply_zdiffusion_t: bool,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    z_nabla2_e = _calculate_nabla2_for_z(kh_smag_e, inv_dual_edge_length, theta_v)
    z_temp = _calculate_nabla2_of_theta(z_nabla2_e, geofac_div)

    if apply_zdiffusion_t:
        # Only where `zd_diffcoef` is not 0 to avoid loading the other fields.
        # Eventually the toolchain could consider extracting a mask `zd_diffcoef != 0` if we tell gt4py that `zd_diffcoef` is static.
        z_temp = where(
            zd_diffcoef != 0.0,
            _truly_horizontal_diffusion_nabla_of_theta_over_steep_points(
                zd_vertoffset,
                zd_diffcoef,
                geofac_n2s_c,
                geofac_n2s_nbh,
                vcoef,
                theta_v,
                z_temp,
            ),
            z_temp,
        )

    theta_v, exner = _update_theta_and_exner(z_temp, area, theta_v, exner, rd_o_cvd)

    return theta_v, exner
