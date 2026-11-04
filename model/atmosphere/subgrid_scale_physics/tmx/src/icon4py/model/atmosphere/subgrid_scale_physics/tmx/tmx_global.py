# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""
tmx turbulent mixing for grids without lateral boundaries.

The same step as `tmx.Tmx`, as one field operator call that returns the new state, the tendencies
and the diagnostics instead of writing into its arguments. The step keeps no state between calls.

The step does no halo exchange: on a distributed grid its inputs must already be valid on every
halo point it reads, and it computes its outputs on the owned points only. Counted in neighbour
hops from an owned cell, the step reads:

- `u`, `v`, `w`, `rho`, `temperature`, `virtual_temperature`, `pressure`, `qv`, `qc`, `qi`, `qr`,
  `qs`, `qg` and `air_mass` on cells up to C2E2C2E2C, the vertices of their edges included: `km_ic`
  enters the w and vn stress on the vertices and edges of each cell's neighbours, and it gathers
  over V2C;
- the surface fluxes on the cells of the edges of each owned cell, through C2E2C.

This is the read set of the granule's exchanges moved to the inputs; it is not measured on a
distributed run.
"""

from __future__ import annotations

from typing import Any

import gt4py.next as gtx

import icon4py.model.common.grid.states as grid_states
from icon4py.model.atmosphere.subgrid_scale_physics.tmx import config as tmx_config, tmx_states
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils import diagnostics_stencils
from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.tmx_global_step import (
    _tmx_global_step,
)
from icon4py.model.common import constants, dimension as dims, field_type_aliases as fa
from icon4py.model.common.grid import horizontal as h_grid, icon as icon_grid
from icon4py.model.common.states.surface_flux_state import SurfaceFluxState
from icon4py.model.common.utils import data_allocation as data_alloc


class TmxGlobal:
    """One tmx step on a grid without lateral boundaries."""

    def __init__(
        self,
        *,
        grid: icon_grid.IconGrid,
        config: tmx_config.TmxConfig,
        metric_state: tmx_states.TmxMetricState,
        interpolation_state: tmx_states.TmxInterpolationState,
        edge_params: grid_states.EdgeParams,
        cell_params: grid_states.CellParams,
        allocator: gtx.typing.Allocator | None,
        backend: gtx.typing.Backend | None = None,
    ) -> None:
        if grid.limited_area:
            raise ValueError("'TmxGlobal' does not support limited area grids.")
        assert cell_params.area is not None
        assert edge_params.edge_cell_distances is not None

        def unstructured(op: Any) -> Any:
            return op.with_grid_type(gtx.GridType.UNSTRUCTURED).with_backend(backend)

        self._step = unstructured(_tmx_global_step)
        self._num_levels = int(grid.num_levels)
        self._offset_provider = grid.connectivities

        cell_domain = h_grid.domain(dims.CellDim)
        self._cells = (
            int(grid.start_index(cell_domain(h_grid.Zone.NUDGING))),
            int(grid.end_index(cell_domain(h_grid.Zone.LOCAL))),
        )
        num_levels = self._num_levels
        all_cells = (0, grid.num_cells)

        mixing_length_sq = unstructured(diagnostics_stencils._compute_smagorinsky_mixing_length)(
            metric_state.ddqz_z_half,
            metric_state.geopot_agl_ifc,
            cell_params.area,
            float(config.smag_constant),
            float(config.max_turb_scale),
            float(constants.GRAV),
            domain={dims.CellDim: all_cells, dims.KHalfDim: (0, num_levels + 1)},
            offset_provider={},
        )
        if config.use_louis:
            scaling_factor_louis = unstructured(diagnostics_stencils._compute_scaling_factor_louis)(
                cell_params.area, domain={dims.CellDim: all_cells}, offset_provider={}
            )
        else:
            scaling_factor_louis = data_alloc.zero_field(grid, dims.CellDim, allocator=allocator)
        rturb_prandtl = 1.0 / config.turb_prandtl

        self._static_args: dict[str, Any] = {
            "ddqz_z_full": metric_state.ddqz_z_full,
            "inv_ddqz_z_full": metric_state.inv_ddqz_z_full,
            "inv_ddqz_z_half": metric_state.inv_ddqz_z_half,
            "inv_ddqz_z_full_e": metric_state.inv_ddqz_z_full_e,
            "inv_ddqz_z_half_e": metric_state.inv_ddqz_z_half_e,
            "inv_ddqz_z_half_v": metric_state.inv_ddqz_z_half_v,
            "wgtfac_c": metric_state.wgtfac_c,
            "wgtfac_e": metric_state.wgtfac_e,
            "wgtfacq_c": metric_state.wgtfacq_c,
            "wgtfacq1_c": metric_state.wgtfacq1_c,
            "wgtfacq_e": metric_state.wgtfacq_e,
            "wgtfacq1_e": metric_state.wgtfacq1_e,
            "height_above_ground": metric_state.height_above_ground,
            "c_lin_e": interpolation_state.c_lin_e,
            "e_bln_c_s": interpolation_state.e_bln_c_s,
            "geofac_div": interpolation_state.geofac_div,
            "cells_aw_verts": interpolation_state.cells_aw_verts,
            "rbf_coeff_v1": interpolation_state.rbf_coeff_v1,
            "rbf_coeff_v2": interpolation_state.rbf_coeff_v2,
            "rbf_coeff_e": interpolation_state.rbf_coeff_e,
            "rbf_coeff_c1": interpolation_state.rbf_coeff_c1,
            "rbf_coeff_c2": interpolation_state.rbf_coeff_c2,
            "primal_normal_cell_x": edge_params.primal_normal_cell[0],
            "primal_normal_cell_y": edge_params.primal_normal_cell[1],
            "primal_normal_vert_x": edge_params.primal_normal_vert[0],
            "primal_normal_vert_y": edge_params.primal_normal_vert[1],
            "dual_normal_vert_x": edge_params.dual_normal_vert[0],
            "dual_normal_vert_y": edge_params.dual_normal_vert[1],
            "edge_cell_length": edge_params.edge_cell_distances,
            "tangent_orientation": edge_params.tangent_orientation,
            "inv_primal_edge_length": edge_params.inverse_primal_edge_lengths,
            "inv_vert_vert_length": edge_params.inverse_vertex_vertex_lengths,
            "inv_dual_edge_length": edge_params.inverse_dual_edge_lengths,
            "mixing_length_sq": mixing_length_sq,
            "scaling_factor_louis": scaling_factor_louis,
            # the land and sea-ice fractions have no source in the atmosphere-only port
            "fract_land": data_alloc.zero_field(grid, dims.CellDim, allocator=allocator),
            "fract_ice": data_alloc.zero_field(grid, dims.CellDim, allocator=allocator),
            "zero_full": data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=allocator),
            "zero_half": data_alloc.zero_field(
                grid, dims.CellDim, dims.KHalfDim, allocator=allocator
            ),
        }
        self._scalar_args: dict[str, Any] = {
            "rturb_prandtl": rturb_prandtl,
            "louis_constant_b": float(config.louis_constant_b),
            "km_const": float(config.km_const),
            "km_min": float(config.km_min),
            "km_surface": float(config.km_const) if config.use_km_const else 0.0,
            "kh_surface": float(config.km_const) * rturb_prandtl if config.use_km_const else 0.0,
            "energy_flux_factor": float(config.scale_turb_energy_flux)
            if config.use_scale_turb_energy_flux
            else 1.0,
            "dissipation_factor": float(config.dissipation_factor),
            "grav": float(constants.GRAV),
            "use_km_const": config.use_km_const,
            "use_louis": config.use_louis,
            "use_louis_land": config.use_louis_land,
            "use_louis_ice": config.use_louis_ice,
            "num_levels": num_levels,
        }

    def run(
        self,
        *,
        input_state: tmx_states.TmxInputState,
        surface_flux_state: SurfaceFluxState,
        dtime: float,
    ) -> tuple[tmx_states.TmxNewState, tmx_states.TmxTendencyState, dict[str, gtx.Field]]:
        """
        Return the new state, the tendencies and the diagnostics `km`, `kh`, `dissip_ke`,
        `heating`, `cptgz_vi`, `dissip_ke_vi`, `int_energy_vi` and `tend_int_energy_vi`.

        The lowest level of `km` and `kh` is `km_const` and `km_const / turb_prandtl` with
        `use_km_const`, and zero otherwise, where it is the surface exchange coefficient this step
        does not compute. w is not diffused on the top and the surface half level: its tendency is
        zero there.
        """
        num_levels = self._num_levels
        full = {dims.CellDim: self._cells, dims.KDim: (0, num_levels)}
        half = {dims.CellDim: self._cells, dims.KHalfDim: (0, num_levels + 1)}
        out = self._step(
            **{
                name: getattr(input_state, name)
                for name in (
                    "temperature",
                    "virtual_temperature",
                    "pressure",
                    "u",
                    "v",
                    "w",
                    "qv",
                    "qc",
                    "qi",
                    "qr",
                    "qs",
                    "qg",
                    "rho",
                    "air_mass",
                )
            },
            evapotranspiration=surface_flux_state.evapotranspiration,
            u_stress=surface_flux_state.u_stress,
            v_stress=surface_flux_state.v_stress,
            q_snocpymlt=surface_flux_state.q_snocpymlt,
            thermal_energy_flux=surface_flux_state.thermal_energy_flux,
            vapor_energy_flux=surface_flux_state.vapor_energy_flux,
            **self._static_args,
            **self._scalar_args,
            dtime=dtime,
            domain=(full,) * 6 + (half,) + (full,) * 6 + (half,) + (full,) * 8,
            offset_provider=self._offset_provider,
        )
        new_state = tmx_states.TmxNewState(*out[0:7])
        tendencies = tmx_states.TmxTendencyState(*out[7:14])
        surface_level = dims.KDim(num_levels - 1)
        diagnostics: dict[str, fa.CellField | fa.CellKField] = {
            "km": out[14],
            "kh": out[15],
            "dissip_ke": out[16],
            "heating": out[17],
            **{
                name: running_sum[surface_level]
                for name, running_sum in zip(
                    ("cptgz_vi", "dissip_ke_vi", "int_energy_vi", "tend_int_energy_vi"),
                    out[18:22],
                    strict=True,
                )
            },
        }
        return new_state, tendencies, diagnostics
