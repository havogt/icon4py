# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import where

from icon4py.model.atmosphere.dycore.stencils.compute_dwdz_for_divergence_damping import (
    _compute_dwdz_for_divergence_damping,
)
from icon4py.model.atmosphere.dycore.stencils.compute_vn_on_lateral_boundary import (
    _compute_vn_on_lateral_boundary,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _compute_vn_on_masked_lateral_boundary(
    mask: fa.EdgeField[bool],
    grf_tend_vn: fa.EdgeKField[wpfloat],
    vn_now: fa.EdgeKField[wpfloat],
    vn_new: fa.EdgeKField[wpfloat],
    dtime: wpfloat,
) -> fa.EdgeKField[wpfloat]:
    return where(
        mask,
        _compute_vn_on_lateral_boundary(grf_tend_vn=grf_tend_vn, vn_now=vn_now, dtime=dtime),
        vn_new,
    )


@gtx.field_operator
def _update_density_exner_wind_on_masked_lateral_boundary(
    mask: fa.CellField[bool],
    rho_now: fa.CellKField[wpfloat],
    grf_tend_rho: fa.CellKField[wpfloat],
    theta_v_now: fa.CellKField[wpfloat],
    grf_tend_thv: fa.CellKField[wpfloat],
    w_now: fa.CellKHalfField[wpfloat],
    grf_tend_w: fa.CellKHalfField[wpfloat],
    rho_new: fa.CellKField[wpfloat],
    exner_new: fa.CellKField[wpfloat],
    w_new: fa.CellKHalfField[wpfloat],
    dtime: wpfloat,
) -> tuple[
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKHalfField[wpfloat],
]:
    rho_new = where(mask, rho_now + dtime * grf_tend_rho, rho_new)
    exner_new = where(mask, theta_v_now + dtime * grf_tend_thv, exner_new)
    w_new = where(mask, w_now + dtime * grf_tend_w, w_new)
    return rho_new, exner_new, w_new


@gtx.field_operator
def _compute_dwdz_for_divergence_damping_on_masked_cells(
    mask: fa.CellField[bool],
    inv_ddqz_z_full: fa.CellKField[vpfloat],
    w: fa.CellKHalfField[wpfloat],
    w_concorr_c: fa.CellKHalfField[vpfloat],
    z_dwdz_dd: fa.CellKField[vpfloat],
) -> fa.CellKField[vpfloat]:
    return where(
        mask,
        _compute_dwdz_for_divergence_damping(
            inv_ddqz_z_full=inv_ddqz_z_full, w=w, w_concorr_c=w_concorr_c
        ),
        z_dwdz_dd,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_vn_on_masked_lateral_boundary(
    mask: fa.EdgeField[bool],
    grf_tend_vn: fa.EdgeKField[wpfloat],
    vn_now: fa.EdgeKField[wpfloat],
    vn_new: fa.EdgeKField[wpfloat],
    dtime: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Set `vn_new` to `vn_now + dtime * grf_tend_vn` on the edges selected by `mask`."""
    _compute_vn_on_masked_lateral_boundary(
        mask=mask,
        grf_tend_vn=grf_tend_vn,
        vn_now=vn_now,
        vn_new=vn_new,
        dtime=dtime,
        out=vn_new,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def update_density_exner_wind_on_masked_lateral_boundary(
    mask: fa.CellField[bool],
    rho_now: fa.CellKField[wpfloat],
    grf_tend_rho: fa.CellKField[wpfloat],
    theta_v_now: fa.CellKField[wpfloat],
    grf_tend_thv: fa.CellKField[wpfloat],
    w_now: fa.CellKHalfField[wpfloat],
    grf_tend_w: fa.CellKHalfField[wpfloat],
    rho_new: fa.CellKField[wpfloat],
    exner_new: fa.CellKField[wpfloat],
    w_new: fa.CellKHalfField[wpfloat],
    dtime: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """
    Apply the lateral boundary tendencies on the cells selected by `mask`.

    As in `update_density_exner_wind`, `exner_new` receives the updated `theta_v`.
    `vertical_end` is the number of half levels.
    """
    _update_density_exner_wind_on_masked_lateral_boundary(
        mask=mask,
        rho_now=rho_now,
        grf_tend_rho=grf_tend_rho,
        theta_v_now=theta_v_now,
        grf_tend_thv=grf_tend_thv,
        w_now=w_now,
        grf_tend_w=grf_tend_w,
        rho_new=rho_new,
        exner_new=exner_new,
        w_new=w_new,
        dtime=dtime,
        out=(rho_new, exner_new, w_new),
        domain=(
            {
                dims.CellDim: (horizontal_start, horizontal_end),
                dims.KDim: (vertical_start, vertical_end - 1),
            },
            {
                dims.CellDim: (horizontal_start, horizontal_end),
                dims.KDim: (vertical_start, vertical_end - 1),
            },
            {
                dims.CellDim: (horizontal_start, horizontal_end),
                dims.KHalfDim: (vertical_start, vertical_end),
            },
        ),
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_dwdz_for_divergence_damping_on_masked_cells(
    mask: fa.CellField[bool],
    inv_ddqz_z_full: fa.CellKField[vpfloat],
    w: fa.CellKHalfField[wpfloat],
    w_concorr_c: fa.CellKHalfField[vpfloat],
    z_dwdz_dd: fa.CellKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_dwdz_for_divergence_damping_on_masked_cells(
        mask=mask,
        inv_ddqz_z_full=inv_ddqz_z_full,
        w=w,
        w_concorr_c=w_concorr_c,
        z_dwdz_dd=z_dwdz_dd,
        out=z_dwdz_dd,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
