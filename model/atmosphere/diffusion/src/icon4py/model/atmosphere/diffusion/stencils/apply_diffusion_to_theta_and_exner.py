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

from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, int32

from icon4py.model.atmosphere.diffusion.stencils.calculate_nabla2_for_z import (
    _calculate_nabla2_for_z,
)
from icon4py.model.atmosphere.diffusion.stencils.calculate_nabla2_of_theta import (
    _calculate_nabla2_of_theta,
)
from icon4py.model.atmosphere.diffusion.stencils.truly_horizontal_diffusion_nabla_of_theta_over_steep_points import (
    _truly_horizontal_diffusion_nabla_of_theta_over_steep_points,
)
from icon4py.model.atmosphere.diffusion.stencils.update_theta_and_exner import (
    _update_theta_and_exner,
)
from icon4py.model.common.dimension import CECDim, CEDim, CellDim, EdgeDim, KDim


@field_operator
def _apply_diffusion_to_theta_and_exner(
    kh_smag_e: Field[[EdgeDim, KDim], float],
    inv_dual_edge_length: Field[[EdgeDim], float],
    theta_v_in: Field[[CellDim, KDim], float],
    geofac_div: Field[[CEDim], float],
    mask: Field[[CellDim, KDim], bool],
    zd_vertoffset: Field[[CECDim, KDim], int32],
    zd_diffcoef: Field[[CellDim, KDim], float],
    geofac_n2s_c: Field[[CellDim], float],
    geofac_n2s_nbh: Field[[CECDim], float],
    vcoef: Field[[CECDim, KDim], float],
    area: Field[[CellDim], float],
    exner: Field[[CellDim, KDim], float],
    rd_o_cvd: float,
) -> tuple[Field[[CellDim, KDim], float], Field[[CellDim, KDim], float]]:
    z_nabla2_e = _calculate_nabla2_for_z(kh_smag_e, inv_dual_edge_length, theta_v_in)
    z_temp = _calculate_nabla2_of_theta(z_nabla2_e, geofac_div)
    z_temp = _truly_horizontal_diffusion_nabla_of_theta_over_steep_points(
        mask,
        zd_vertoffset,
        zd_diffcoef,
        geofac_n2s_c,
        geofac_n2s_nbh,
        vcoef,
        theta_v_in,
        z_temp,
    )
    theta_v, exner = _update_theta_and_exner(z_temp, area, theta_v_in, exner, rd_o_cvd)
    return theta_v, exner


@program
def apply_diffusion_to_theta_and_exner(
    kh_smag_e: Field[[EdgeDim, KDim], float],
    inv_dual_edge_length: Field[[EdgeDim], float],
    theta_v_in: Field[[CellDim, KDim], float],
    geofac_div: Field[[CEDim], float],
    mask: Field[[CellDim, KDim], bool],
    zd_vertoffset: Field[[CECDim, KDim], int32],
    zd_diffcoef: Field[[CellDim, KDim], float],
    geofac_n2s_c: Field[[CellDim], float],
    geofac_n2s_nbh: Field[[CECDim], float],
    vcoef: Field[[CECDim, KDim], float],
    area: Field[[CellDim], float],
    theta_v: Field[[CellDim, KDim], float],
    exner: Field[[CellDim, KDim], float],
    rd_o_cvd: float,
):
    _apply_diffusion_to_theta_and_exner(
        kh_smag_e,
        inv_dual_edge_length,
        theta_v_in,
        geofac_div,
        mask,
        zd_vertoffset,
        zd_diffcoef,
        geofac_n2s_c,
        geofac_n2s_nbh,
        vcoef,
        area,
        exner,
        rd_o_cvd,
        out=(theta_v, exner),
    )
