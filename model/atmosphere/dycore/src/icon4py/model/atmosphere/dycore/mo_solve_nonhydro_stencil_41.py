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
from gt4py.next.ffront.fbuiltins import Field, int32, neighbor_sum

from icon4py.model.common.dimension import C2CE, C2E, C2EDim, CEDim, CellDim, EdgeDim, KDim


@field_operator
def _mo_solve_nonhydro_stencil_41(
    geofac_div: Field[[CEDim], float],
    mass_fl_e: Field[[EdgeDim, KDim], float],
    z_theta_v_fl_e: Field[[EdgeDim, KDim], float],
) -> tuple[Field[[CellDim, KDim], float], Field[[CellDim, KDim], float]]:
    z_flxdiv_mass = neighbor_sum(geofac_div(C2CE) * mass_fl_e(C2E), axis=C2EDim)
    z_flxdiv_theta = neighbor_sum(geofac_div(C2CE) * z_theta_v_fl_e(C2E), axis=C2EDim)
    return z_flxdiv_mass, z_flxdiv_theta


@program(grid_type=GridType.UNSTRUCTURED)
def mo_solve_nonhydro_stencil_41(
    geofac_div: Field[[CEDim], float],
    mass_fl_e: Field[[EdgeDim, KDim], float],
    z_theta_v_fl_e: Field[[EdgeDim, KDim], float],
    z_flxdiv_mass: Field[[CellDim, KDim], float],
    z_flxdiv_theta: Field[[CellDim, KDim], float],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _mo_solve_nonhydro_stencil_41(
        geofac_div,
        mass_fl_e,
        z_theta_v_fl_e,
        out=(z_flxdiv_mass, z_flxdiv_theta),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
