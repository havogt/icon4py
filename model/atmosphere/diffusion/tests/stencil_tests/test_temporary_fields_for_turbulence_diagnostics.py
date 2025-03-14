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

from icon4py.model.atmosphere.diffusion.stencils.temporary_fields_for_turbulence_diagnostics import (
    temporary_fields_for_turbulence_diagnostics,
)
from icon4py.model.common.dimension import C2EDim, CEDim, CellDim, EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import (
    StencilTest,
    as_1D_sparse_field,
    random_field,
    zero_field,
)


class TestTemporaryFieldsForTurbulenceDiagnostics(StencilTest):
    PROGRAM = temporary_fields_for_turbulence_diagnostics
    OUTPUTS = ("div", "kh_c")

    @staticmethod
    def reference(
        grid,
        kh_smag_ec: np.array,
        vn: np.array,
        e_bln_c_s: np.array,
        geofac_div: np.array,
        diff_multfac_smag: np.array,
        **kwargs,
    ) -> dict:
        c2e = grid.connectivities[C2EDim]
        geofac_div = np.expand_dims(geofac_div, axis=-1)
        vn_geofac = vn[c2e] * geofac_div[grid.get_offset_provider("C2CE").table]
        div = np.sum(vn_geofac, axis=1)
        e_bln_c_s = np.expand_dims(e_bln_c_s, axis=-1)
        diff_multfac_smag = np.expand_dims(diff_multfac_smag, axis=0)
        mul = kh_smag_ec[c2e] * e_bln_c_s[grid.get_offset_provider("C2CE").table]
        summed = np.sum(mul, axis=1)
        kh_c = summed / diff_multfac_smag

        return dict(div=div, kh_c=kh_c)

    @pytest.fixture
    def input_data(self, grid):
        vn = random_field(grid, EdgeDim, KDim)
        geofac_div = as_1D_sparse_field(random_field(grid, CellDim, C2EDim), CEDim)
        kh_smag_ec = random_field(grid, EdgeDim, KDim)
        e_bln_c_s = as_1D_sparse_field(random_field(grid, CellDim, C2EDim), CEDim)
        diff_multfac_smag = random_field(grid, KDim)

        kh_c = zero_field(grid, CellDim, KDim)
        div = zero_field(grid, CellDim, KDim)

        return dict(
            kh_smag_ec=kh_smag_ec,
            vn=vn,
            e_bln_c_s=e_bln_c_s,
            geofac_div=geofac_div,
            diff_multfac_smag=diff_multfac_smag,
            kh_c=kh_c,
            div=div,
        )
