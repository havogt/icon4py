# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import Any

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.common import constants as phy_const, dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.driver.physics_global import _surface_and_hydrostatic_pressure
from icon4py.model.testing import stencil_tests


def _surface_pressure(
    exner: np.ndarray, virtual_temperature: np.ndarray, ddqz_z_full: np.ndarray
) -> np.ndarray:
    return phy_const.P0REF * np.exp(
        phy_const.CPD_O_RD * np.log(exner[:, -3])
        + phy_const.GRAV_O_RD
        * (
            ddqz_z_full[:, -1] / virtual_temperature[:, -1]
            + ddqz_z_full[:, -2] / virtual_temperature[:, -2]
            + 0.5 * ddqz_z_full[:, -3] / virtual_temperature[:, -3]
        )
    )


@pytest.mark.embedded_only
class TestSurfaceAndHydrostaticPressure(stencil_tests.StencilTest):
    PROGRAM = _surface_and_hydrostatic_pressure
    OUTPUTS = ("out",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        exner: np.ndarray,
        virtual_temperature: np.ndarray,
        ddqz_z_full: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        surface_pressure = _surface_pressure(exner, virtual_temperature, ddqz_z_full)
        num_levels = virtual_temperature.shape[1]
        pressure_ifc = np.zeros((virtual_temperature.shape[0], num_levels + 1))
        pressure = np.zeros_like(virtual_temperature)
        pressure_ifc[:, num_levels] = surface_pressure
        for k in range(num_levels - 1, -1, -1):
            pressure_ifc[:, k] = pressure_ifc[:, k + 1] * np.exp(
                -phy_const.GRAV_O_RD * ddqz_z_full[:, k] / virtual_temperature[:, k]
            )
            pressure[:, k] = np.sqrt(pressure_ifc[:, k] * pressure_ifc[:, k + 1])
        return dict(out=(pressure, pressure_ifc))

    @stencil_tests.input_data_fixture
    def input_data(data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid) -> dict:
        exner = data_alloc.random_field(dims.CellDim, dims.KDim, low=0.7, high=1.0)
        virtual_temperature = data_alloc.random_field(
            dims.CellDim, dims.KDim, low=200.0, high=300.0
        )
        ddqz_z_full = data_alloc.random_field(dims.CellDim, dims.KDim, low=50.0, high=500.0)
        surface_pressure = gtx.as_field(
            (dims.CellDim,),
            _surface_pressure(  # type: ignore[arg-type]
                exner.asnumpy(), virtual_temperature.asnumpy(), ddqz_z_full.asnumpy()
            ),
            allocator=data_alloc.allocator,
        )
        return dict(
            exner=exner,
            virtual_temperature=virtual_temperature,
            ddqz_z_full=ddqz_z_full,
            surface_pressure=surface_pressure,
            num_levels=gtx.int32(grid.num_levels),
            out=(
                data_alloc.zero_field(dims.CellDim, dims.KDim),
                data_alloc.zero_field(dims.CellDim, dims.KHalfDim),
            ),
            domain=(
                {dims.CellDim: (0, grid.num_cells), dims.KDim: (0, grid.num_levels)},
                {dims.CellDim: (0, grid.num_cells), dims.KHalfDim: (0, grid.num_levels + 1)},
            ),
        )
