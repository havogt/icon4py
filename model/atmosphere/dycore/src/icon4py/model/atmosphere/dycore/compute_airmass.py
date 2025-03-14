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
from gt4py.next import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field

from icon4py.model.common.dimension import CellDim, KDim


@field_operator
def _compute_airmass(
    rho_in: Field[[CellDim, KDim], float],
    ddqz_z_full_in: Field[[CellDim, KDim], float],
    deepatmo_t1mc_in: Field[[KDim], float],
) -> Field[[CellDim, KDim], float]:
    return rho_in * ddqz_z_full_in * deepatmo_t1mc_in


@program(grid_type=GridType.UNSTRUCTURED)
def compute_airmass(
    rho_in: Field[[CellDim, KDim], float],
    ddqz_z_full_in: Field[[CellDim, KDim], float],
    deepatmo_t1mc_in: Field[[KDim], float],
    airmass_out: Field[[CellDim, KDim], float],
):
    _compute_airmass(rho_in, ddqz_z_full_in, deepatmo_t1mc_in, out=airmass_out)
