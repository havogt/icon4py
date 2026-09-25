# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next.experimental import concat_where

from icon4py.model.atmosphere.diffusion.diffusion_utils import _scale_k
from icon4py.model.atmosphere.diffusion.stencils.apply_diffusion_to_theta_and_exner import (
    _apply_diffusion_to_theta_and_exner,
)
from icon4py.model.atmosphere.diffusion.stencils.apply_nabla2_and_nabla4_global_to_vn import (
    _apply_nabla2_and_nabla4_global_to_vn,
)
from icon4py.model.atmosphere.diffusion.stencils.apply_nabla2_to_w import _apply_nabla2_to_w
from icon4py.model.atmosphere.diffusion.stencils.apply_nabla2_to_w_in_upper_damping_layer import (
    _apply_nabla2_to_w_in_upper_damping_layer,
)
from icon4py.model.atmosphere.diffusion.stencils.calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools import (
    _calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools,
)
from icon4py.model.atmosphere.diffusion.stencils.calculate_nabla2_and_smag_coefficients_for_vn import (
    _calculate_nabla2_and_smag_coefficients_for_vn,
)
from icon4py.model.atmosphere.diffusion.stencils.calculate_nabla2_for_w import (
    _calculate_nabla2_for_w,
)
from icon4py.model.atmosphere.diffusion.stencils.calculate_nabla4 import _calculate_nabla4
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.interpolation.stencils.mo_intp_rbf_rbf_vec_interpol_vertex import (
    _mo_intp_rbf_rbf_vec_interpol_vertex,
)
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _diffusion_global_step(
    vn: fa.EdgeKField[wpfloat],
    w: fa.CellKHalfField[wpfloat],
    theta_v: fa.CellKField[wpfloat],
    exner: fa.CellKField[wpfloat],
    diff_multfac_vn: fa.KField[wpfloat],
    smag_limit: fa.KField[vpfloat],
    enh_smag_fac: fa.KField[float],
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
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    geofac_n2s_c: fa.CellField[wpfloat],
    geofac_n2s_nbh: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], wpfloat],
    theta_ref_mc: fa.CellKField[vpfloat],
    zd_vertoffset: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim, dims.KDim], gtx.int32],
    zd_diffcoef: fa.CellKField[wpfloat],
    zd_intcoef: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim, dims.KDim], wpfloat],
    dtime: wpfloat,
    smag_offset: vpfloat,
    diff_multfac_w: wpfloat,
    thresh_tdiff: wpfloat,
    smallest_vpfloat: vpfloat,
    rd_o_cvd: vpfloat,
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
    diff_multfac_smag = _scale_k(enh_smag_fac, dtime)
    u_vert, v_vert = _mo_intp_rbf_rbf_vec_interpol_vertex(vn, rbf_coeff_1, rbf_coeff_2)
    kh_smag_e, _, z_nabla2_e = _calculate_nabla2_and_smag_coefficients_for_vn(
        diff_multfac_smag,
        tangent_orientation,
        inv_primal_edge_length,
        inv_vert_vert_length,
        u_vert,
        v_vert,
        primal_normal_vert_x,
        primal_normal_vert_y,
        dual_normal_vert_x,
        dual_normal_vert_y,
        vn,
        smag_limit,
        smag_offset,
    )
    u_vert, v_vert = _mo_intp_rbf_rbf_vec_interpol_vertex(z_nabla2_e, rbf_coeff_1, rbf_coeff_2)
    z_nabla4_e2 = _calculate_nabla4(
        u_vert,
        v_vert,
        primal_normal_vert_x,
        primal_normal_vert_y,
        z_nabla2_e,
        inv_vert_vert_length,
        inv_primal_edge_length,
    )
    vn_new = _apply_nabla2_and_nabla4_global_to_vn(
        edge_area, kh_smag_e, z_nabla2_e, z_nabla4_e2, diff_multfac_vn, vn
    )

    z_nabla2_c = _calculate_nabla2_for_w(w, geofac_n2s)
    w_new = _apply_nabla2_to_w(cell_area, z_nabla2_c, geofac_n2s, w, diff_multfac_w)
    w_new = concat_where(
        (dims.KHalfDim >= 1) & (dims.KHalfDim < nrdmax),
        _apply_nabla2_to_w_in_upper_damping_layer(w_new, diff_multfac_n2w, cell_area, z_nabla2_c),
        w_new,
    )
    w_new = concat_where(dims.KHalfDim < num_levels, w_new, w)

    kh_smag_e = concat_where(
        num_levels - 2 <= dims.KDim,
        _calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools(
            theta_v, theta_ref_mc, thresh_tdiff, smallest_vpfloat, kh_smag_e
        ),
        kh_smag_e,
    )
    theta_v_new, exner_new = (
        _apply_diffusion_to_theta_and_exner(
            kh_smag_e,
            inv_dual_edge_length,
            theta_v,
            geofac_div,
            zd_vertoffset,
            zd_diffcoef,
            geofac_n2s_c,
            geofac_n2s_nbh,
            zd_intcoef,
            cell_area,
            exner,
            rd_o_cvd,
            apply_zdiffusion_t,
        )
        if apply_to_temperature
        else (theta_v, exner)
    )
    return vn_new, w_new, theta_v_new, exner_new
