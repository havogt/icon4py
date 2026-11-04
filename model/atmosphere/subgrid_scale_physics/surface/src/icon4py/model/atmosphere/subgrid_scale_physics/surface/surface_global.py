# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""
The prescribed-flux surface scheme for grids without lateral boundaries.

The same step as `surface.Surface`, as one field operator call that returns the surface fluxes
instead of writing into its arguments. It reads only the cell it computes, so it needs no halo.
"""

from __future__ import annotations

from typing import Any

import gt4py.next as gtx

from icon4py.model.atmosphere.subgrid_scale_physics.surface import (
    config as surface_config,
    surface_states,
)
from icon4py.model.atmosphere.subgrid_scale_physics.surface.stencils.prescribed_flux_stencils import (
    _compute_prescribed_surface_fluxes,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import horizontal as h_grid, icon as icon_grid
from icon4py.model.common.states.surface_flux_state import SurfaceFluxState
from icon4py.model.common.utils import data_allocation as data_alloc


class SurfaceGlobal:
    """One step of the prescribed-flux surface scheme on a grid without lateral boundaries."""

    def __init__(
        self,
        *,
        grid: icon_grid.IconGrid,
        config: surface_config.SurfaceConfig,
        allocator: gtx.typing.Allocator | None,
        backend: gtx.typing.Backend | None = None,
    ) -> None:
        if grid.limited_area:
            raise ValueError("'SurfaceGlobal' does not support limited area grids.")
        if config.surface_flux_type is not surface_config.SurfaceFluxType.FIXED_HEAT_FLUXES:
            raise NotImplementedError(f"surface flux type {config.surface_flux_type}")
        self._step = _compute_prescribed_surface_fluxes.with_grid_type(
            gtx.GridType.UNSTRUCTURED
        ).with_backend(backend)
        cell_domain = h_grid.domain(dims.CellDim)
        self._cells = (
            int(grid.start_index(cell_domain(h_grid.Zone.NUDGING))),
            int(grid.end_index(cell_domain(h_grid.Zone.LOCAL))),
        )
        self.surface_temperature = data_alloc.constant_field(
            grid,
            config.sea_surface_temperature,
            dims.CellDim,
            dtype=ta.wpfloat,
            allocator=allocator,
        )
        self._zero = data_alloc.zero_field(grid, dims.CellDim, allocator=allocator)
        self._scalar_args: dict[str, Any] = {
            "shflx": float(config.kinematic_sensible_heat_flux),
            "lhflx": float(config.kinematic_latent_heat_flux),
        }

    def run(self, *, input_state: surface_states.SurfaceInputState) -> SurfaceFluxState:
        """Return the surface fluxes; the wind stress and `q_snocpymlt` are zero, as in ICON."""
        (
            evapotranspiration,
            sensible_heat_flux,
            latent_heat_flux,
            thermal_energy_flux,
            vapor_energy_flux,
        ) = self._step(
            surface_temperature=self.surface_temperature,
            surface_pressure=input_state.surface_pressure,
            temperature=input_state.temperature,
            **self._scalar_args,
            domain=({dims.CellDim: self._cells},) * 5,
            offset_provider={},
        )
        return SurfaceFluxState(
            evapotranspiration=evapotranspiration,
            sensible_heat_flux=sensible_heat_flux,
            latent_heat_flux=latent_heat_flux,
            u_stress=self._zero,
            v_stress=self._zero,
            q_snocpymlt=self._zero,
            thermal_energy_flux=thermal_energy_flux,
            vapor_energy_flux=vapor_energy_flux,
        )
