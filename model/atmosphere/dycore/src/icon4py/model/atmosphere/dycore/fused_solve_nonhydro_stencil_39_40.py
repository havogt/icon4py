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

from gt4py.next.ffront.decorator import GridType, field_operator, program
from gt4py.next.ffront.fbuiltins import Field, int32, where

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_39 import (
    _mo_solve_nonhydro_stencil_39,
)
from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_40 import (
    _mo_solve_nonhydro_stencil_40,
)
from icon4py.model.common.dimension import CEDim, CellDim, EdgeDim, KDim


@field_operator
def _fused_solve_nonhydro_stencil_39_40(
    e_bln_c_s: Field[[CEDim], float],
    z_w_concorr_me: Field[[EdgeDim, KDim], float],
    wgtfac_c: Field[[CellDim, KDim], float],
    wgtfacq_c: Field[[CellDim, KDim], float],
    vert_idx: Field[[KDim], int32],
    nlev: int32,
    nflatlev: int32,
) -> Field[[CellDim, KDim], float]:
    w_concorr_c = where(
        nflatlev + 1 <= vert_idx < nlev,
        _mo_solve_nonhydro_stencil_39(e_bln_c_s, z_w_concorr_me, wgtfac_c),
        _mo_solve_nonhydro_stencil_40(e_bln_c_s, z_w_concorr_me, wgtfacq_c),
    )
    return w_concorr_c


@program(grid_type=GridType.UNSTRUCTURED)
def fused_solve_nonhydro_stencil_39_40(
    e_bln_c_s: Field[[CEDim], float],
    z_w_concorr_me: Field[[EdgeDim, KDim], float],
    wgtfac_c: Field[[CellDim, KDim], float],
    wgtfacq_c: Field[[CellDim, KDim], float],
    vert_idx: Field[[KDim], int32],
    nlev: int32,
    nflatlev: int32,
    w_concorr_c: Field[[CellDim, KDim], float],
):
    _fused_solve_nonhydro_stencil_39_40(
        e_bln_c_s,
        z_w_concorr_me,
        wgtfac_c,
        wgtfacq_c,
        vert_idx,
        nlev,
        nflatlev,
        out=w_concorr_c[:, -1:],
    )
