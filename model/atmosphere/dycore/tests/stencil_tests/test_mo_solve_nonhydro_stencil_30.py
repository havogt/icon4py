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

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_30 import (
    mo_solve_nonhydro_stencil_30,
)
from icon4py.model.common.dimension import E2C2EDim, E2C2EODim, EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field


class TestMoSolveNonhydroStencil30(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_30
    OUTPUTS = ("z_vn_avg", "z_graddiv_vn", "vt")

    @staticmethod
    def reference(
        grid,
        e_flx_avg: np.array,
        vn: np.array,
        geofac_grdiv: np.array,
        rbf_vec_coeff_e: np.array,
        **kwargs,
    ) -> dict:
        e2c2eO = grid.connectivities[E2C2EODim]
        e2c2e = grid.connectivities[E2C2EDim]
        e_flx_avg = np.expand_dims(e_flx_avg, axis=-1)
        z_vn_avg = np.sum(
            np.where((e2c2eO != -1)[:, :, np.newaxis], vn[e2c2eO] * e_flx_avg, 0), axis=1
        )
        geofac_grdiv = np.expand_dims(geofac_grdiv, axis=-1)
        z_graddiv_vn = np.sum(
            np.where((e2c2eO != -1)[:, :, np.newaxis], vn[e2c2eO] * geofac_grdiv, 0), axis=1
        )
        rbf_vec_coeff_e = np.expand_dims(rbf_vec_coeff_e, axis=-1)
        vt = np.sum(
            np.where((e2c2e != -1)[:, :, np.newaxis], vn[e2c2e] * rbf_vec_coeff_e, 0), axis=1
        )
        return dict(z_vn_avg=z_vn_avg, z_graddiv_vn=z_graddiv_vn, vt=vt)

    @pytest.fixture
    def input_data(self, grid):
        e_flx_avg = random_field(grid, EdgeDim, E2C2EODim)
        geofac_grdiv = random_field(grid, EdgeDim, E2C2EODim)
        rbf_vec_coeff_e = random_field(grid, EdgeDim, E2C2EDim)
        vn = random_field(grid, EdgeDim, KDim)
        z_vn_avg = zero_field(grid, EdgeDim, KDim)
        z_graddiv_vn = zero_field(grid, EdgeDim, KDim)
        vt = zero_field(grid, EdgeDim, KDim)

        return dict(
            e_flx_avg=e_flx_avg,
            vn=vn,
            geofac_grdiv=geofac_grdiv,
            rbf_vec_coeff_e=rbf_vec_coeff_e,
            z_vn_avg=z_vn_avg,
            z_graddiv_vn=z_graddiv_vn,
            vt=vt,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_edges),
            vertical_start=int32(0),
            vertical_end=int32(grid.num_levels),
        )
