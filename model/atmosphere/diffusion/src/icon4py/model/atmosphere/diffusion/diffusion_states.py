# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import dataclasses
import functools

import gt4py.next as gtx
from gt4py.next import common as gtx_common

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta


@dataclasses.dataclass(frozen=True)
class DiffusionMetricState:
    """Represents the metric state fields needed in diffusion."""

    theta_ref_mc: fa.CellKField[ta.vpfloat]
    wgtfac_c: fa.CellKHalfField[
        ta.vpfloat
    ]  # weighting factor for interpolation from full to half levels (nproma,nlevp1,nblks_c)
    zd_vertoffset: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim, dims.KDim], gtx.int32]
    zd_diffcoef: fa.CellKField[ta.wpfloat]
    zd_intcoef: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim, dims.KDim], ta.wpfloat]


@dataclasses.dataclass(frozen=True)
class DiffusionInterpolationState:
    """Represents the ICON interpolation state needed in diffusion."""

    e_bln_c_s: gtx.Field[
        gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat
    ]  # coefficent for bilinear interpolation from edge to cell ()
    rbf_coeff_1: gtx.Field[
        gtx.Dims[dims.VertexDim, dims.V2EDim], ta.wpfloat
    ]  # rbf_vec_coeff_v_1(nproma, rbf_vec_dim_v, nblks_v)
    rbf_coeff_2: gtx.Field[
        gtx.Dims[dims.VertexDim, dims.V2EDim], ta.wpfloat
    ]  # rbf_vec_coeff_v_2(nproma, rbf_vec_dim_v, nblks_v)

    geofac_div: gtx.Field[
        gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat
    ]  # factor for divergence (nproma,cell_type,nblks_c)

    geofac_n2s: gtx.Field[
        gtx.Dims[dims.CellDim, dims.C2E2CODim], ta.wpfloat
    ]  # factor for nabla2-scalar (nproma,cell_type+1,nblks_c)
    geofac_grg_x: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], ta.wpfloat]
    geofac_grg_y: gtx.Field[
        gtx.Dims[dims.CellDim, dims.C2E2CODim], ta.wpfloat
    ]  # factors for green gauss gradient (nproma,4,nblks_c,2)
    nudgecoeff_e: fa.EdgeField[ta.wpfloat]  # Nudging coefficients for edges

    # Views of `geofac_n2s` through GT4Py's internal API: `gtx.as_field` without an allocator
    # copies the buffer to NumPy, whatever array library it is of.
    @functools.cached_property
    def geofac_n2s_c(self) -> fa.CellField[ta.wpfloat]:
        return gtx_common._field(
            self.geofac_n2s.ndarray[:, 0],
            domain={dims.CellDim: self.geofac_n2s.domain[dims.CellDim].unit_range},
        )

    @functools.cached_property
    def geofac_n2s_nbh(self) -> gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat]:
        return gtx_common._field(
            self.geofac_n2s.ndarray[:, 1:],
            domain={
                dims.CellDim: self.geofac_n2s.domain[dims.CellDim].unit_range,
                dims.C2E2CDim: (0, self.geofac_n2s.ndarray.shape[1] - 1),
            },
        )
