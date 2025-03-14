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

from icon4py.model.atmosphere.dycore.mo_velocity_advection_stencil_18 import (
    mo_velocity_advection_stencil_18,
)
from icon4py.model.common.dimension import C2E2CODim, CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, random_mask


class TestMoVelocityAdvectionStencil18(StencilTest):
    PROGRAM = mo_velocity_advection_stencil_18
    OUTPUTS = ("ddt_w_adv",)

    @staticmethod
    def reference(
        grid,
        levmask: np.array,
        cfl_clipping: np.array,
        owner_mask: np.array,
        z_w_con_c: np.array,
        ddqz_z_half: np.array,
        area: np.array,
        geofac_n2s: np.array,
        w: np.array,
        ddt_w_adv: np.array,
        scalfac_exdiff: float,
        cfl_w_limit: float,
        dtime: float,
        **kwargs,
    ):
        levmask = np.expand_dims(levmask, axis=0)
        owner_mask = np.expand_dims(owner_mask, axis=-1)
        area = np.expand_dims(area, axis=-1)
        geofac_n2s = np.expand_dims(geofac_n2s, axis=-1)

        difcoef = np.where(
            (levmask == 1) & (cfl_clipping == 1) & (owner_mask == 1),
            scalfac_exdiff
            * np.minimum(
                0.85 - cfl_w_limit * dtime,
                np.abs(z_w_con_c) * dtime / ddqz_z_half - cfl_w_limit * dtime,
            ),
            0,
        )

        ddt_w_adv = np.where(
            (levmask == 1) & (cfl_clipping == 1) & (owner_mask == 1),
            ddt_w_adv
            + difcoef
            * area
            * np.sum(
                np.where(
                    (grid.connectivities[C2E2CODim] != -1)[:, :, np.newaxis],
                    w[grid.connectivities[C2E2CODim]] * geofac_n2s,
                    0,
                ),
                axis=1,
            ),
            ddt_w_adv,
        )

        return dict(ddt_w_adv=ddt_w_adv)

    @pytest.fixture
    def input_data(self, grid):
        levmask = random_mask(grid, KDim)
        cfl_clipping = random_mask(grid, CellDim, KDim)
        owner_mask = random_mask(grid, CellDim)
        z_w_con_c = random_field(grid, CellDim, KDim)
        ddqz_z_half = random_field(grid, CellDim, KDim)
        area = random_field(grid, CellDim)
        geofac_n2s = random_field(grid, CellDim, C2E2CODim)
        w = random_field(grid, CellDim, KDim)
        ddt_w_adv = random_field(grid, CellDim, KDim)
        scalfac_exdiff = 10.0
        cfl_w_limit = 3.0
        dtime = 2.0

        return dict(
            levmask=levmask,
            cfl_clipping=cfl_clipping,
            owner_mask=owner_mask,
            z_w_con_c=z_w_con_c,
            ddqz_z_half=ddqz_z_half,
            area=area,
            geofac_n2s=geofac_n2s,
            w=w,
            ddt_w_adv=ddt_w_adv,
            scalfac_exdiff=scalfac_exdiff,
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_cells),
            vertical_start=int32(0),
            vertical_end=int32(grid.num_levels),
        )
