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

from icon4py.model.atmosphere.dycore.mo_solve_nonhydro_stencil_28 import (
    mo_solve_nonhydro_stencil_28,
)
from icon4py.model.common.dimension import EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field


class TestMoSolveNonhydroStencil28(StencilTest):
    PROGRAM = mo_solve_nonhydro_stencil_28
    OUTPUTS = ("vn",)

    @staticmethod
    def reference(grid, vn_incr: np.array, vn: np.array, iau_wgt_dyn, **kwargs) -> np.array:
        vn = vn + (iau_wgt_dyn * vn_incr)
        return dict(vn=vn)

    @pytest.fixture
    def input_data(self, grid):
        vn_incr = random_field(grid, EdgeDim, KDim)
        vn = random_field(grid, EdgeDim, KDim)
        iau_wgt_dyn = 5.0

        return dict(
            vn_incr=vn_incr,
            vn=vn,
            iau_wgt_dyn=iau_wgt_dyn,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_edges),
            vertical_start=int32(0),
            vertical_end=int32(grid.num_levels),
        )
