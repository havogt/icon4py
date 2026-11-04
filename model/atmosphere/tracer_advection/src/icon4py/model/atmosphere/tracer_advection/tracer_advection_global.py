# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""
Tracer advection for grids without lateral boundaries.

The same step as `tracer_advection.GodunovSplittingAdvection`, as one field operator call that
returns the new tracer and its fluxes instead of writing into its arguments. The order of the
Godunov splitting is an argument of each call: the step keeps no state between calls.

The step does no halo exchange: on a distributed grid its inputs must already be valid on every
halo point it reads, and it computes its outputs on the owned points only.
"""

from __future__ import annotations

from typing import Any

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing

import icon4py.model.common.grid.states as grid_states
from icon4py.model.atmosphere.tracer_advection import tracer_advection, tracer_advection_states
from icon4py.model.atmosphere.tracer_advection.stencils.tracer_advection_global_step import (
    _tracer_advection_global_step,
)
from icon4py.model.common import (
    constants,
    dimension as dims,
    field_type_aliases as fa,
    type_alias as ta,
)
from icon4py.model.common.grid import horizontal as h_grid, icon as icon_grid
from icon4py.model.common.states import tracer_prep_adv_states as prep_adv_states
from icon4py.model.common.utils import data_allocation as data_alloc


class TracerAdvectionGlobal:
    """One tracer advection step on a grid without lateral boundaries."""

    def __init__(
        self,
        *,
        grid: icon_grid.IconGrid,
        config: tracer_advection.AdvectionConfig,
        interpolation_state: tracer_advection_states.AdvectionInterpolationState,
        least_squares_state: tracer_advection_states.AdvectionLeastSquaresState,
        metric_state: tracer_advection_states.AdvectionMetricState,
        edge_params: grid_states.EdgeParams,
        allocator: gtx_typing.Allocator | None,
        backend: gtx_typing.Backend | None = None,
    ) -> None:
        if grid.limited_area:
            raise ValueError("'TracerAdvectionGlobal' does not support limited area grids.")

        self._step = _tracer_advection_global_step.with_grid_type(
            gtx.GridType.UNSTRUCTURED
        ).with_backend(backend)
        self.config = config
        self._grid = grid
        self._num_levels = int(grid.num_levels)
        self._offset_provider = grid.connectivities

        def interior(dim: gtx.Dimension) -> tuple[int, int]:
            domain = h_grid.domain(dim)(h_grid.Zone.INTERIOR)
            return int(grid.start_index(domain)), int(grid.end_index(domain))

        self._cells = interior(dims.CellDim)
        self._edges = interior(dims.EdgeDim)

        self._static_args: dict[str, Any] = {
            "ddqz_z_full": metric_state.ddqz_z_full,
            "deepatmo_divh": metric_state.deepatmo_divh,
            "deepatmo_divzl": metric_state.deepatmo_divzl,
            "deepatmo_divzu": metric_state.deepatmo_divzu,
            "geofac_div": interpolation_state.geofac_div,
            "rbf_vec_coeff_e": interpolation_state.rbf_vec_coeff_e,
            "pos_on_tplane_e_1": interpolation_state.pos_on_tplane_e_1,
            "pos_on_tplane_e_2": interpolation_state.pos_on_tplane_e_2,
            "primal_normal_cell_1": edge_params.primal_normal_cell[0],
            "dual_normal_cell_1": edge_params.dual_normal_cell[0],
            "primal_normal_cell_2": edge_params.primal_normal_cell[1],
            "dual_normal_cell_2": edge_params.dual_normal_cell[1],
            "lsq_pseudoinv_1": least_squares_state.lsq_pseudoinv_1,
            "lsq_pseudoinv_2": least_squares_state.lsq_pseudoinv_2,
            "k": data_alloc.index_field(grid, dims.KDim, dtype=gtx.int32, allocator=allocator),
            "k_half": data_alloc.index_field(
                grid, dims.KHalfDim, dtype=gtx.int32, allocator=allocator
            ),
            "zero_full": data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=allocator),
            "zero_half": data_alloc.zero_field(
                grid, dims.CellDim, dims.KHalfDim, allocator=allocator
            ),
            "zero_edge": data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, allocator=allocator),
        }
        self._scalar_args: dict[str, Any] = {
            "num_levels": self._num_levels,
            "wp_eps": float(constants.WP_EPS),
            "horizontal_advection_type": config.horizontal_advection_type.value,
            "horizontal_limiter": config.horizontal_advection_limiter.value,
            "vertical_advection_type": config.vertical_advection_type.value,
            "vertical_limiter": config.vertical_advection_limiter.value,
        }

    def air_mass(self, rho: fa.CellKField[ta.wpfloat]) -> fa.CellKField[ta.wpfloat]:
        """The mass of air per cell area in each layer, as `run` takes `airmass_now/new`."""
        return rho * self._static_args["ddqz_z_full"]

    def run(
        self,
        *,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        airmass_now: fa.CellKField[ta.wpfloat],
        airmass_new: fa.CellKField[ta.wpfloat],
        prep_adv: prep_adv_states.TracerPrepAdvState,
        dtime: float,
        even_timestep: bool,
    ) -> tuple[fa.CellKField[ta.wpfloat], fa.EdgeKField[ta.wpfloat], fa.CellKHalfField[ta.wpfloat]]:
        """
        Return the new tracer mass fraction and the horizontal and vertical tracer fluxes.

        `even_timestep` selects the vertical-first splitting order (originally the Marchuk order).
        """
        num_levels = self._num_levels
        return self._step(
            p_tracer_now=p_tracer_now,
            airmass_now=airmass_now,
            airmass_new=airmass_new,
            mass_flx_ic=prep_adv.mass_flx_ic,
            mass_flx_me=prep_adv.mass_flx_me,
            vn_traj=prep_adv.vn_traj,
            **self._static_args,
            **self._scalar_args,
            p_dtime=dtime,
            even_timestep=even_timestep,
            domain=(
                {dims.CellDim: self._cells, dims.KDim: (0, num_levels)},
                {dims.EdgeDim: self._edges, dims.KDim: (0, num_levels)},
                {dims.CellDim: self._cells, dims.KHalfDim: (0, num_levels + 1)},
            ),
            offset_provider=self._offset_provider,
        )
