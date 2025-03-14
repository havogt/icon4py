# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, int32

from icon4py.model.common.dimension import E2C, E2V, CellDim, EdgeDim, KDim, VertexDim


@field_operator
def _mo_velocity_advection_stencil_07(
    vn_ie: Field[[EdgeDim, KDim], float],
    inv_dual_edge_length: Field[[EdgeDim], float],
    w: Field[[CellDim, KDim], float],
    z_vt_ie: Field[[EdgeDim, KDim], float],
    inv_primal_edge_length: Field[[EdgeDim], float],
    tangent_orientation: Field[[EdgeDim], float],
    z_w_v: Field[[VertexDim, KDim], float],
) -> Field[[EdgeDim, KDim], float]:
    return vn_ie * inv_dual_edge_length * (
        w(E2C[0]) - w(E2C[1])
    ) + z_vt_ie * inv_primal_edge_length * tangent_orientation * (z_w_v(E2V[0]) - z_w_v(E2V[1]))


@program(grid_type=GridType.UNSTRUCTURED)
def mo_velocity_advection_stencil_07(
    vn_ie: Field[[EdgeDim, KDim], float],
    inv_dual_edge_length: Field[[EdgeDim], float],
    w: Field[[CellDim, KDim], float],
    z_vt_ie: Field[[EdgeDim, KDim], float],
    inv_primal_edge_length: Field[[EdgeDim], float],
    tangent_orientation: Field[[EdgeDim], float],
    z_w_v: Field[[VertexDim, KDim], float],
    z_v_grad_w: Field[[EdgeDim, KDim], float],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _mo_velocity_advection_stencil_07(
        vn_ie,
        inv_dual_edge_length,
        w,
        z_vt_ie,
        inv_primal_edge_length,
        tangent_orientation,
        z_w_v,
        out=z_v_grad_w,
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
