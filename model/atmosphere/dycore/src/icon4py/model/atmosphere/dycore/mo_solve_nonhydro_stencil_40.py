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
from gt4py.next.ffront.fbuiltins import Field, neighbor_sum

from icon4py.model.common.dimension import C2CE, C2E, C2EDim, CEDim, CellDim, EdgeDim, KDim, Koff


@field_operator
def _mo_solve_nonhydro_stencil_40(
    e_bln_c_s: Field[[CEDim], float],
    z_w_concorr_me: Field[[EdgeDim, KDim], float],
    wgtfacq_c: Field[[CellDim, KDim], float],
) -> Field[[CellDim, KDim], float]:
    z_w_concorr_me_offset_1 = z_w_concorr_me(Koff[-1])
    z_w_concorr_me_offset_2 = z_w_concorr_me(Koff[-2])
    z_w_concorr_me_offset_3 = z_w_concorr_me(Koff[-3])

    z_w_concorr_mc_m1 = neighbor_sum(e_bln_c_s(C2CE) * z_w_concorr_me_offset_1(C2E), axis=C2EDim)
    z_w_concorr_mc_m2 = neighbor_sum(e_bln_c_s(C2CE) * z_w_concorr_me_offset_2(C2E), axis=C2EDim)
    z_w_concorr_mc_m3 = neighbor_sum(e_bln_c_s(C2CE) * z_w_concorr_me_offset_3(C2E), axis=C2EDim)

    return (
        wgtfacq_c(Koff[-1]) * z_w_concorr_mc_m1
        + wgtfacq_c(Koff[-2]) * z_w_concorr_mc_m2
        + wgtfacq_c(Koff[-3]) * z_w_concorr_mc_m3
    )


@program(grid_type=GridType.UNSTRUCTURED)
def mo_solve_nonhydro_stencil_40(
    e_bln_c_s: Field[[CEDim], float],
    z_w_concorr_me: Field[[EdgeDim, KDim], float],
    wgtfacq_c: Field[[CellDim, KDim], float],
    w_concorr_c: Field[[CellDim, KDim], float],
):
    _mo_solve_nonhydro_stencil_40(e_bln_c_s, z_w_concorr_me, wgtfacq_c, out=w_concorr_c[:, -1:])
