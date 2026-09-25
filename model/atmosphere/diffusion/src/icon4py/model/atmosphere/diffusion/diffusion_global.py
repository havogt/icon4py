# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""
Diffusion for grids without lateral boundaries, executed on the embedded backend.

The same time step as `diffusion.Diffusion`, but every stencil is a direct field operator call
instead of a program set up with `setup_program`. The diagnostics for turbulence (`div_ic`,
`hdef_ic`, `dwdx`, `dwdy`) are not computed, whatever `shear_type`, `loutshs` and `a_hshr` ask for.
"""

from __future__ import annotations

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing

import icon4py.model.common.grid.states as grid_states
import icon4py.model.common.states.prognostic_state as prognostics
from icon4py.model.atmosphere.diffusion import diffusion_states
from icon4py.model.atmosphere.diffusion.diffusion import DiffusionConfig, DiffusionParams
from icon4py.model.atmosphere.diffusion.diffusion_utils import (
    _init_diffusion_local_fields_for_regular_timestep,
    _init_nabla2_factor_in_upper_damping_zone,
    _scale_k,
    _setup_fields_for_initial_step,
)
from icon4py.model.atmosphere.diffusion.stencils.apply_diffusion_to_theta_and_exner import (
    _apply_diffusion_to_theta_and_exner,
)
from icon4py.model.atmosphere.diffusion.stencils.apply_diffusion_to_vn import _apply_diffusion_to_vn
from icon4py.model.atmosphere.diffusion.stencils.apply_nabla2_to_w import _apply_nabla2_to_w
from icon4py.model.atmosphere.diffusion.stencils.apply_nabla2_to_w_in_upper_damping_layer import (
    _apply_nabla2_to_w_in_upper_damping_layer,
)
from icon4py.model.atmosphere.diffusion.stencils.calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools import (
    _calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools,
)
from icon4py.model.atmosphere.diffusion.stencils.calculate_nabla2_and_smag_coefficients_for_vn import (
    _calculate_nabla2_and_smag_coefficients_for_vn,
)
from icon4py.model.atmosphere.diffusion.stencils.calculate_nabla2_for_w import (
    _calculate_nabla2_for_w,
)
from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.grid import horizontal as h_grid, icon as icon_grid, vertical as v_grid
from icon4py.model.common.interpolation.stencils.mo_intp_rbf_rbf_vec_interpol_vertex import (
    _mo_intp_rbf_rbf_vec_interpol_vertex,
)
from icon4py.model.common.utils import data_allocation as data_alloc


class DiffusionGlobal:
    """One diffusion step on a grid without lateral boundaries."""

    def __init__(
        self,
        *,
        grid: icon_grid.IconGrid,
        config: DiffusionConfig,
        params: DiffusionParams,
        vertical_grid: v_grid.VerticalGrid,
        metric_state: diffusion_states.DiffusionMetricState,
        interpolation_state: diffusion_states.DiffusionInterpolationState,
        edge_params: grid_states.EdgeParams,
        cell_params: grid_states.CellParams,
        allocator: gtx_typing.Allocator | None,
        ndyn_substeps: int,
    ) -> None:
        if grid.limited_area:
            raise ValueError("'DiffusionGlobal' does not support limited area grids.")
        if grid.config.distributed:
            raise ValueError(
                "'DiffusionGlobal' does not do halo exchanges; it runs on a single rank only."
            )
        assert cell_params.area is not None

        self._allocator = allocator
        self.config = config
        self._params = params
        self._grid = grid
        self._vertical_grid = vertical_grid
        self._metric_state = metric_state
        self._interpolation_state = interpolation_state
        self._edge_params = edge_params
        self._cell_params = cell_params
        self._offset_provider = grid.connectivities
        ndyn_substeps_as_float = float(ndyn_substeps)

        self.rd_o_cvd: float = constants.GAS_CONSTANT_DRY_AIR / (
            constants.CPD - constants.GAS_CONSTANT_DRY_AIR
        )
        #: threshold temperature deviation from neighboring grid points that activates extra diffusion against runaway cooling
        self.thresh_tdiff: float = -5.0
        self.smag_offset: float = 0.25 * params.K4 * ndyn_substeps_as_float
        self.diff_multfac_w: float = min(1.0 / 48.0, params.K4W * ndyn_substeps_as_float)
        self._determine_horizontal_domains()
        self._allocate_local_fields()

        _init_diffusion_local_fields_for_regular_timestep(
            params.K4,
            ndyn_substeps_as_float,
            *params.smagorinski_factor,
            *params.smagorinski_height,
            self._vertical_grid.interface_physical_height,
            out=(self.diff_multfac_vn, self.smag_limit, self.enh_smag_fac),
            offset_provider={},
        )
        end_index_of_damping_layer = self._vertical_grid.end_index_of_damping_layer
        heights = self._vertical_grid.interface_physical_height
        _init_nabla2_factor_in_upper_damping_zone(
            physical_heights=heights,
            end_index_of_damping_layer=end_index_of_damping_layer,
            nshift=0,
            heights_nrd_shift=heights.ndarray[end_index_of_damping_layer + 1].item(),
            heights_1=heights.ndarray[1].item(),
            out=self.diff_multfac_n2w,
            domain={dims.KHalfDim: (1, end_index_of_damping_layer + 1)},
            offset_provider={},
        )

    def _allocate_local_fields(self) -> None:
        allocator = self._allocator
        self.diff_multfac_vn = data_alloc.zero_field(self._grid, dims.KDim, allocator=allocator)
        self.diff_multfac_n2w = data_alloc.zero_field(
            self._grid, dims.KHalfDim, allocator=allocator
        )
        self.smag_limit = data_alloc.zero_field(self._grid, dims.KDim, allocator=allocator)
        self.enh_smag_fac = data_alloc.zero_field(self._grid, dims.KDim, allocator=allocator)
        self.u_vert = data_alloc.zero_field(
            self._grid, dims.VertexDim, dims.KDim, allocator=allocator
        )
        self.v_vert = data_alloc.zero_field(
            self._grid, dims.VertexDim, dims.KDim, allocator=allocator
        )
        self.kh_smag_e = data_alloc.zero_field(
            self._grid, dims.EdgeDim, dims.KDim, allocator=allocator
        )
        self.kh_smag_ec = data_alloc.zero_field(
            self._grid, dims.EdgeDim, dims.KDim, allocator=allocator
        )
        self.z_nabla2_e = data_alloc.zero_field(
            self._grid, dims.EdgeDim, dims.KDim, allocator=allocator
        )
        self.diff_multfac_smag = data_alloc.zero_field(self._grid, dims.KDim, allocator=allocator)
        self.z_nabla2_c = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KHalfDim, allocator=allocator
        )

    def _determine_horizontal_domains(self) -> None:
        def interior(dim: gtx.Dimension) -> tuple[gtx.int32, gtx.int32]:
            domain = h_grid.domain(dim)(h_grid.Zone.INTERIOR)
            return self._grid.start_index(domain), self._grid.end_index(domain)

        self._cells = interior(dims.CellDim)
        self._edges = interior(dims.EdgeDim)
        self._vertices = interior(dims.VertexDim)

    def run(
        self,
        prognostic_state: prognostics.PrognosticState,
        dtime: float,
        initial_run: bool = False,
    ) -> None:
        num_levels = self._grid.num_levels
        if initial_run:
            diff_multfac_vn = data_alloc.zero_field(
                self._grid, dims.KDim, allocator=self._allocator
            )
            smag_limit = data_alloc.zero_field(self._grid, dims.KDim, allocator=self._allocator)
            _setup_fields_for_initial_step(
                self._params.K4,
                self.config.hdiff_efdt_ratio,
                out=(diff_multfac_vn, smag_limit),
                offset_provider={},
            )
            smag_offset = 0.0
        else:
            diff_multfac_vn = self.diff_multfac_vn
            smag_limit = self.smag_limit
            smag_offset = self.smag_offset

        _scale_k(self.enh_smag_fac, dtime, out=self.diff_multfac_smag, offset_provider={})

        vertex_domain = {
            dims.VertexDim: self._vertices,
            dims.KDim: (0, num_levels),
        }
        _mo_intp_rbf_rbf_vec_interpol_vertex(
            p_e_in=prognostic_state.vn,
            ptr_coeff_1=self._interpolation_state.rbf_coeff_1,
            ptr_coeff_2=self._interpolation_state.rbf_coeff_2,
            out=(self.u_vert, self.v_vert),
            domain=vertex_domain,
            offset_provider=self._offset_provider,
        )

        _calculate_nabla2_and_smag_coefficients_for_vn(
            diff_multfac_smag=self.diff_multfac_smag,
            tangent_orientation=self._edge_params.tangent_orientation,
            inv_primal_edge_length=self._edge_params.inverse_primal_edge_lengths,
            inv_vert_vert_length=self._edge_params.inverse_vertex_vertex_lengths,
            u_vert=self.u_vert,
            v_vert=self.v_vert,
            primal_normal_vert_x=self._edge_params.primal_normal_vert[0],
            primal_normal_vert_y=self._edge_params.primal_normal_vert[1],
            dual_normal_vert_x=self._edge_params.dual_normal_vert[0],
            dual_normal_vert_y=self._edge_params.dual_normal_vert[1],
            vn=prognostic_state.vn,
            smag_limit=smag_limit,
            smag_offset=smag_offset,
            out=(self.kh_smag_e, self.kh_smag_ec, self.z_nabla2_e),
            domain={
                dims.EdgeDim: self._edges,
                dims.KDim: (0, num_levels),
            },
            offset_provider=self._offset_provider,
        )

        _mo_intp_rbf_rbf_vec_interpol_vertex(
            p_e_in=self.z_nabla2_e,
            ptr_coeff_1=self._interpolation_state.rbf_coeff_1,
            ptr_coeff_2=self._interpolation_state.rbf_coeff_2,
            out=(self.u_vert, self.v_vert),
            domain=vertex_domain,
            offset_provider=self._offset_provider,
        )

        _apply_diffusion_to_vn(
            u_vert=self.u_vert,
            v_vert=self.v_vert,
            primal_normal_vert_v1=self._edge_params.primal_normal_vert[0],
            primal_normal_vert_v2=self._edge_params.primal_normal_vert[1],
            z_nabla2_e=self.z_nabla2_e,
            inv_vert_vert_length=self._edge_params.inverse_vertex_vertex_lengths,
            inv_primal_edge_length=self._edge_params.inverse_primal_edge_lengths,
            area_edge=self._edge_params.edge_areas,
            kh_smag_e=self.kh_smag_e,
            diff_multfac_vn=diff_multfac_vn,
            nudgecoeff_e=self._interpolation_state.nudgecoeff_e,
            vn=prognostic_state.vn,
            nudgezone_diff=0.0,
            fac_bdydiff_v=0.0,
            start_2nd_nudge_line_idx_e=self._edges[0],
            limited_area=False,
            out=prognostic_state.vn,
            domain={
                dims.EdgeDim: self._edges,
                dims.KDim: (0, num_levels),
            },
            offset_provider=self._offset_provider,
        )

        _calculate_nabla2_for_w(
            w=prognostic_state.w,
            geofac_n2s=self._interpolation_state.geofac_n2s,
            out=self.z_nabla2_c,
            domain={
                dims.CellDim: self._cells,
                dims.KHalfDim: (0, num_levels),
            },
            offset_provider=self._offset_provider,
        )
        _apply_nabla2_to_w(
            area=self._cell_params.area,
            z_nabla2_c=self.z_nabla2_c,
            geofac_n2s=self._interpolation_state.geofac_n2s,
            w=prognostic_state.w,
            diff_multfac_w=self.diff_multfac_w,
            out=prognostic_state.w,
            domain={
                dims.CellDim: self._cells,
                dims.KHalfDim: (0, num_levels),
            },
            offset_provider=self._offset_provider,
        )
        _apply_nabla2_to_w_in_upper_damping_layer(
            w=prognostic_state.w,
            diff_multfac_n2w=self.diff_multfac_n2w,
            cell_area=self._cell_params.area,
            z_nabla2_c=self.z_nabla2_c,
            out=prognostic_state.w,
            domain={
                dims.CellDim: self._cells,
                dims.KHalfDim: (1, self._vertical_grid.end_index_of_damping_layer + 1),
            },
            offset_provider=self._offset_provider,
        )

        if self.config.apply_to_temperature:
            _calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools(
                theta_v=prognostic_state.theta_v,
                theta_ref_mc=self._metric_state.theta_ref_mc,
                thresh_tdiff=self.thresh_tdiff,
                smallest_vpfloat=constants.DBL_EPS,
                kh_smag_e=self.kh_smag_e,
                out=self.kh_smag_e,
                domain={
                    dims.EdgeDim: self._edges,
                    dims.KDim: (num_levels - 2, num_levels),
                },
                offset_provider=self._offset_provider,
            )
            _apply_diffusion_to_theta_and_exner(
                kh_smag_e=self.kh_smag_e,
                inv_dual_edge_length=self._edge_params.inverse_dual_edge_lengths,
                theta_v=prognostic_state.theta_v,
                geofac_div=self._interpolation_state.geofac_div,
                zd_vertoffset=self._metric_state.zd_vertoffset,
                zd_diffcoef=self._metric_state.zd_diffcoef,
                geofac_n2s_c=self._interpolation_state.geofac_n2s_c,
                geofac_n2s_nbh=self._interpolation_state.geofac_n2s_nbh,
                vcoef=self._metric_state.zd_intcoef,
                area=self._cell_params.area,
                exner=prognostic_state.exner,
                rd_o_cvd=self.rd_o_cvd,
                apply_zdiffusion_t=self.config.apply_zdiffusion_t,
                out=(prognostic_state.theta_v, prognostic_state.exner),
                domain={
                    dims.CellDim: self._cells,
                    dims.KDim: (0, num_levels),
                },
                offset_provider=self._offset_provider,
            )
