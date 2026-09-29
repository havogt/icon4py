# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""
One horizontal diffusion step, following Zängl et al. (2015) §2.5, https://doi.org/10.1002/qj.2378.

The coefficients are dimensionless and contain the time step: `kh` is K_h Δt / a_e (Eq 37),
`diff_multfac_vn` is k_4 Δt (Eq 39) and `diff_multfac_w` is k_w Δt (Eq 40).
"""

import gt4py.next as gtx
from gt4py.next import max_over, maximum, minimum, neighbor_sum, where
from gt4py.next.experimental import as_offset, concat_where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import C2E2C, E2C, Koff
from icon4py.model.common.math.differential_operators import (
    components_at_diamond_vertices,
    div,
    grad_n,
    horizontal_deformation,
    nabla2_diamond,
    nabla2_khalf,
    nabla2_n,
    rbf_vector_at_vertices,
)
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def cold_pool_diffusion_coefficient(
    theta_v: fa.CellKField[wpfloat],
    theta_ref_mc: fa.CellKField[wpfloat],
    thresh_tdiff: wpfloat,
    smallest_coefficient: wpfloat,
) -> fa.CellKField[wpfloat]:
    """Extra diffusion coefficient where a cell is much colder than its neighbours."""
    tdiff = theta_v - neighbor_sum(theta_v(C2E2C), axis=dims.C2E2CDim) / 3.0
    trefdiff = theta_ref_mc - neighbor_sum(theta_ref_mc(C2E2C), axis=dims.C2E2CDim) / 3.0
    return where(
        ((tdiff - trefdiff < thresh_tdiff) & (trefdiff < 0.0))
        | (tdiff - trefdiff < 1.5 * thresh_tdiff),
        5.0e-4 * (thresh_tdiff - tdiff + trefdiff),
        smallest_coefficient,
    )


@gtx.field_operator
def nabla2_at_constant_height(
    psi: fa.CellKField[wpfloat],
    zd_vertoffset: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim, dims.KDim], gtx.int32],
    zd_intcoef: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim, dims.KDim], wpfloat],
    geofac_n2s_c: fa.CellField[wpfloat],
    geofac_n2s_nbh: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], wpfloat],
) -> fa.CellKField[wpfloat]:
    """Laplacian with the neighbours interpolated vertically to the cell's height."""
    psi_1 = zd_intcoef[dims.C2E2CDim(0)] * psi(C2E2C[0])(
        as_offset(Koff, zd_vertoffset[dims.C2E2CDim(0)])
    ) + (1.0 - zd_intcoef[dims.C2E2CDim(0)]) * psi(C2E2C[0])(
        as_offset(Koff, zd_vertoffset[dims.C2E2CDim(0)] + 1)
    )
    psi_2 = zd_intcoef[dims.C2E2CDim(1)] * psi(C2E2C[1])(
        as_offset(Koff, zd_vertoffset[dims.C2E2CDim(1)])
    ) + (1.0 - zd_intcoef[dims.C2E2CDim(1)]) * psi(C2E2C[1])(
        as_offset(Koff, zd_vertoffset[dims.C2E2CDim(1)] + 1)
    )
    psi_3 = zd_intcoef[dims.C2E2CDim(2)] * psi(C2E2C[2])(
        as_offset(Koff, zd_vertoffset[dims.C2E2CDim(2)])
    ) + (1.0 - zd_intcoef[dims.C2E2CDim(2)]) * psi(C2E2C[2])(
        as_offset(Koff, zd_vertoffset[dims.C2E2CDim(2)] + 1)
    )
    return (
        geofac_n2s_c * psi
        + geofac_n2s_nbh[dims.C2E2CDim(0)] * psi_1
        + geofac_n2s_nbh[dims.C2E2CDim(1)] * psi_2
        + geofac_n2s_nbh[dims.C2E2CDim(2)] * psi_3
    )


@gtx.field_operator
def exner_at_constant_density(
    exner: fa.CellKField[wpfloat],
    theta_v_new: fa.CellKField[wpfloat],
    theta_v: fa.CellKField[wpfloat],
    rd_o_cvd: wpfloat,
) -> fa.CellKField[wpfloat]:
    """Exner pressure after a change of theta_v at fixed density, linearised."""
    return exner * (1.0 + rd_o_cvd * (theta_v_new / theta_v - 1.0))


@gtx.field_operator
def _diffusion_global_step(
    vn: fa.EdgeKField[wpfloat],
    w: fa.CellKHalfField[wpfloat],
    theta_v: fa.CellKField[wpfloat],
    exner: fa.CellKField[wpfloat],
    diff_multfac_vn: fa.KField[wpfloat],
    smag_limit: fa.KField[wpfloat],
    enh_smag_fac: fa.KField[wpfloat],
    diff_multfac_n2w: fa.KHalfField[wpfloat],
    rbf_coeff_1: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], wpfloat],
    rbf_coeff_2: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], wpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    inv_vert_vert_length: fa.EdgeField[wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    primal_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    primal_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    dual_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    dual_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    edge_area: fa.EdgeField[wpfloat],
    cell_area: fa.CellField[wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    geofac_n2s_c: fa.CellField[wpfloat],
    geofac_n2s_nbh: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], wpfloat],
    theta_ref_mc: fa.CellKField[wpfloat],
    zd_vertoffset: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim, dims.KDim], gtx.int32],
    zd_diffcoef: fa.CellKField[wpfloat],
    zd_intcoef: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim, dims.KDim], wpfloat],
    dtime: wpfloat,
    smag_offset: wpfloat,
    diff_multfac_w: wpfloat,
    thresh_tdiff: wpfloat,
    smallest_coefficient: wpfloat,
    rd_o_cvd: wpfloat,
    nrdmax: gtx.int32,
    num_levels: gtx.int32,
    apply_to_temperature: bool,
    apply_zdiffusion_t: bool,
) -> tuple[
    fa.EdgeKField[wpfloat],
    fa.CellKHalfField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
]:
    # Eq 37: Smagorinsky coefficient, less the offset and bounded by the stability limit
    u, v = rbf_vector_at_vertices(vn, rbf_coeff_1, rbf_coeff_2)
    vn_1, vn_2, vn_3, vn_4 = components_at_diamond_vertices(
        u, v, primal_normal_vert_x, primal_normal_vert_y
    )
    vt_1, vt_2, vt_3, vt_4 = components_at_diamond_vertices(
        u, v, dual_normal_vert_x, dual_normal_vert_y
    )
    deformation = horizontal_deformation(
        vn_1,
        vn_2,
        vn_3,
        vn_4,
        vt_1,
        vt_2,
        vt_3,
        vt_4,
        tangent_orientation,
        inv_primal_edge_length,
        inv_vert_vert_length,
    )
    kh = minimum(maximum(0.0, enh_smag_fac * dtime * deformation - smag_offset), smag_limit)

    # Eqs 35, 36 and 39
    nabla2_vn = nabla2_diamond(
        vn, vn_1, vn_2, vn_3, vn_4, inv_primal_edge_length, inv_vert_vert_length
    )
    nabla4_vn = nabla2_n(
        nabla2_vn,
        rbf_coeff_1,
        rbf_coeff_2,
        primal_normal_vert_x,
        primal_normal_vert_y,
        inv_primal_edge_length,
        inv_vert_vert_length,
    )
    # The factor 4 of Eq 35 is applied inside both Laplacians of the fourth-order term, as in ICON.
    vn_new = (
        vn + 4.0 * edge_area * kh * nabla2_vn - 16.0 * diff_multfac_vn * edge_area**2 * nabla4_vn
    )

    # Eqs 40 and 41, plus a second-order term in the upper damping layer; the surface level is kept
    nabla2_w = nabla2_khalf(w, inv_dual_edge_length, geofac_div)
    w_new = w - diff_multfac_w * cell_area**2 * nabla2_khalf(
        nabla2_w, inv_dual_edge_length, geofac_div
    )
    w_new = concat_where(
        (dims.KHalfDim >= 1) & (dims.KHalfDim < nrdmax),
        w_new + diff_multfac_n2w * cell_area * nabla2_w,
        w_new,
    )
    w_new = concat_where(dims.KHalfDim < num_levels, w_new, w)

    # Eq 38, with the coefficient raised in cold pools on the two lowest levels
    theta_v_new, exner_new = theta_v, exner
    if apply_to_temperature:
        cold_pool = cold_pool_diffusion_coefficient(
            theta_v, theta_ref_mc, thresh_tdiff, smallest_coefficient
        )
        kh_theta = concat_where(
            num_levels - 2 <= dims.KDim,
            maximum(kh, max_over(cold_pool(E2C), axis=dims.E2CDim)),
            kh,
        )
        tendency = div(kh_theta * grad_n(theta_v, inv_dual_edge_length), geofac_div)
        if apply_zdiffusion_t:
            tendency = where(
                zd_diffcoef != 0.0,
                tendency
                + zd_diffcoef
                * nabla2_at_constant_height(
                    theta_v, zd_vertoffset, zd_intcoef, geofac_n2s_c, geofac_n2s_nbh
                ),
                tendency,
            )
        theta_v_new = theta_v + cell_area * tendency
        exner_new = exner_at_constant_density(exner, theta_v_new, theta_v, rd_o_cvd)
    return vn_new, w_new, theta_v_new, exner_new
