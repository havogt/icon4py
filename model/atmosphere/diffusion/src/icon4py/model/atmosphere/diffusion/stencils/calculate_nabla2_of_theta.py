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

from icon4py.model.common.dimension import C2CE, C2E, C2EDim, CEDim, CellDim, EdgeDim, KDim


@field_operator
def _calculate_nabla2_of_theta(
    z_nabla2_e: Field[[EdgeDim, KDim], float],
    geofac_div: Field[[CEDim], float],
) -> Field[[CellDim, KDim], float]:
    z_temp = neighbor_sum(z_nabla2_e(C2E) * geofac_div(C2CE), axis=C2EDim)
    return z_temp


@program(grid_type=GridType.UNSTRUCTURED)
def calculate_nabla2_of_theta(
    z_nabla2_e: Field[[EdgeDim, KDim], float],
    geofac_div: Field[[CEDim], float],
    z_temp: Field[[CellDim, KDim], float],
):
    _calculate_nabla2_of_theta(z_nabla2_e, geofac_div, out=z_temp)
