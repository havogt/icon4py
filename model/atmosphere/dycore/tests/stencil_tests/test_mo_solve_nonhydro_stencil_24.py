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

import numpy as np
import pytest
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_24 import (
    mo_solve_nonhydro_stencil_24,
)
from icon4py.model.common.dimension import EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field


class TestMoSolveNonhydroStencil24(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_24
    OUTPUTS = ("vn_nnew",)

    @staticmethod
    def reference(
        grid,
        vn_nnow: np.array,
        ddt_vn_apc_ntl1: np.array,
        ddt_vn_phy: np.array,
        z_theta_v_e: np.array,
        z_gradh_exner: np.array,
        dtime: float,
        cpd: float,
        **kwargs,
    ) -> np.array:
        vn_nnew = vn_nnow + dtime * (
            ddt_vn_apc_ntl1 + ddt_vn_phy - cpd * z_theta_v_e * z_gradh_exner
        )
        return dict(vn_nnew=vn_nnew)

    @pytest.fixture
    def input_data(self, grid):
        dtime, cpd = 10.0, 10.0
        vn_nnow = random_field(grid, EdgeDim, KDim)
        ddt_vn_apc_ntl1 = random_field(grid, EdgeDim, KDim)
        ddt_vn_phy = random_field(grid, EdgeDim, KDim)
        z_theta_v_e = random_field(grid, EdgeDim, KDim)
        z_gradh_exner = random_field(grid, EdgeDim, KDim)
        vn_nnew = zero_field(grid, EdgeDim, KDim)

        return dict(
            vn_nnow=vn_nnow,
            ddt_vn_apc_ntl1=ddt_vn_apc_ntl1,
            ddt_vn_phy=ddt_vn_phy,
            z_theta_v_e=z_theta_v_e,
            z_gradh_exner=z_gradh_exner,
            vn_nnew=vn_nnew,
            dtime=dtime,
            cpd=cpd,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_edges),
            vertical_start=int32(0),
            vertical_end=int32(grid.num_levels),
        )
