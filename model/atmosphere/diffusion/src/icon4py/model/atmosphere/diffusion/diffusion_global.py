# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""
Diffusion for grids without lateral boundaries.

The step does no halo exchange: on a distributed grid its inputs must already be valid on every halo
point it reads, and it computes its outputs on the owned points only.

The same time step as `diffusion.Diffusion`, but every stencil is a direct field operator call
instead of a program set up with `setup_program`. The diagnostics for turbulence (`div_ic`,
`hdef_ic`, `dwdx`, `dwdy`) are not computed, whatever `shear_type`, `loutshs` and `a_hshr` ask for.
"""

from __future__ import annotations

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing
from gt4py.next.experimental import concat_where

import icon4py.model.common.grid.states as grid_states
import icon4py.model.common.states.prognostic_state as prognostics
from icon4py.model.atmosphere.diffusion import diffusion_states
from icon4py.model.atmosphere.diffusion.diffusion import DiffusionConfig, DiffusionParams
from icon4py.model.atmosphere.diffusion.diffusion_utils import (
    _init_diffusion_local_fields_for_regular_timestep,
    _init_nabla2_factor_in_upper_damping_zone,
    _setup_fields_for_initial_step,
)
from icon4py.model.atmosphere.diffusion.stencils.diffusion_global_step import _diffusion_global_step
from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.grid import horizontal as h_grid, icon as icon_grid, vertical as v_grid
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
        backend: gtx_typing.Backend | None = None,
    ) -> None:
        if grid.limited_area:
            raise ValueError("'DiffusionGlobal' does not support limited area grids.")
        assert cell_params.area is not None

        def unstructured(op: gtx_typing.FieldOperator) -> gtx_typing.FieldOperator:
            return op.with_grid_type(gtx.GridType.UNSTRUCTURED).with_backend(backend)

        self._init_diffusion_local_fields_for_regular_timestep = unstructured(
            _init_diffusion_local_fields_for_regular_timestep
        )
        self._init_nabla2_factor_in_upper_damping_zone = unstructured(
            _init_nabla2_factor_in_upper_damping_zone
        )
        self._setup_fields_for_initial_step = unstructured(_setup_fields_for_initial_step)
        self._diffusion_global_step = unstructured(_diffusion_global_step)

        self.config = config
        self._params = params
        self._grid = grid
        self._vertical_grid = vertical_grid
        self._metric_state = metric_state
        self._interpolation_state = interpolation_state
        self._geofac_n2s_c = interpolation_state.geofac_n2s_c
        self._geofac_n2s_nbh = interpolation_state.geofac_n2s_nbh
        self._edge_params = edge_params
        self._cell_params = cell_params
        self._offset_provider = grid.connectivities
        ndyn_substeps_as_float = float(ndyn_substeps)

        # The scalars passed to the step are Python ints and floats: torch.compile traces numpy
        # scalars and the arithmetic on them into the compiled region.
        self.rd_o_cvd: float = float(
            constants.GAS_CONSTANT_DRY_AIR / (constants.CPD - constants.GAS_CONSTANT_DRY_AIR)
        )
        #: threshold temperature deviation from neighboring grid points that activates extra diffusion against runaway cooling
        self.thresh_tdiff: float = -5.0
        self.smag_offset: float = float(0.25 * params.K4 * ndyn_substeps_as_float)
        self.diff_multfac_w: float = float(min(1.0 / 48.0, params.K4W * ndyn_substeps_as_float))
        self._smallest_vpfloat = float(constants.DBL_EPS)
        self._nrdmax = int(vertical_grid.end_index_of_damping_layer) + 1
        self._num_levels = int(self._grid.num_levels)
        self._determine_horizontal_domains()

        num_levels = self._num_levels
        self.diff_multfac_vn, self.smag_limit, self.enh_smag_fac = (
            self._init_diffusion_local_fields_for_regular_timestep(
                params.K4,
                ndyn_substeps_as_float,
                *params.smagorinski_factor,
                *params.smagorinski_height,
                self._vertical_grid.interface_physical_height,
                domain={dims.KDim: (0, num_levels)},
                offset_provider={},
            )
        )
        end_index_of_damping_layer = self._vertical_grid.end_index_of_damping_layer
        heights = self._vertical_grid.interface_physical_height
        self.diff_multfac_n2w = data_alloc.zero_field(
            self._grid, dims.KHalfDim, allocator=allocator
        )
        if end_index_of_damping_layer > 0:
            self.diff_multfac_n2w = concat_where(
                (dims.KHalfDim >= 1) & (dims.KHalfDim < end_index_of_damping_layer + 1),
                self._init_nabla2_factor_in_upper_damping_zone(
                    physical_heights=heights,
                    end_index_of_damping_layer=end_index_of_damping_layer,
                    nshift=0,
                    heights_nrd_shift=heights.ndarray[end_index_of_damping_layer + 1].item(),
                    heights_1=heights.ndarray[1].item(),
                    domain={dims.KHalfDim: (1, end_index_of_damping_layer + 1)},
                    offset_provider={},
                ),
                self.diff_multfac_n2w,
            )

    def _determine_horizontal_domains(self) -> None:
        def interior(dim: gtx.Dimension) -> tuple[int, int]:
            domain = h_grid.domain(dim)(h_grid.Zone.INTERIOR)
            return int(self._grid.start_index(domain)), int(self._grid.end_index(domain))

        self._cells = interior(dims.CellDim)
        self._edges = interior(dims.EdgeDim)
        self._vertices = interior(dims.VertexDim)

    def run(
        self,
        prognostic_state: prognostics.PrognosticState,
        dtime: float,
        initial_run: bool = False,
    ) -> prognostics.PrognosticState:
        num_levels = self._num_levels
        k_domain = {dims.KDim: (0, num_levels)}

        if initial_run:
            diff_multfac_vn, smag_limit = self._setup_fields_for_initial_step(
                self._params.K4,
                self.config.hdiff_efdt_ratio,
                domain=k_domain,
                offset_provider={},
            )
            smag_offset = 0.0
        else:
            diff_multfac_vn = self.diff_multfac_vn
            smag_limit = self.smag_limit
            smag_offset = self.smag_offset

        vn, w, theta_v, exner = self._diffusion_global_step(
            vn=prognostic_state.vn,
            w=prognostic_state.w,
            theta_v=prognostic_state.theta_v,
            exner=prognostic_state.exner,
            diff_multfac_vn=diff_multfac_vn,
            smag_limit=smag_limit,
            enh_smag_fac=self.enh_smag_fac,
            diff_multfac_n2w=self.diff_multfac_n2w,
            rbf_coeff_1=self._interpolation_state.rbf_coeff_1,
            rbf_coeff_2=self._interpolation_state.rbf_coeff_2,
            tangent_orientation=self._edge_params.tangent_orientation,
            inv_primal_edge_length=self._edge_params.inverse_primal_edge_lengths,
            inv_vert_vert_length=self._edge_params.inverse_vertex_vertex_lengths,
            inv_dual_edge_length=self._edge_params.inverse_dual_edge_lengths,
            primal_normal_vert_x=self._edge_params.primal_normal_vert[0],
            primal_normal_vert_y=self._edge_params.primal_normal_vert[1],
            dual_normal_vert_x=self._edge_params.dual_normal_vert[0],
            dual_normal_vert_y=self._edge_params.dual_normal_vert[1],
            edge_area=self._edge_params.edge_areas,
            cell_area=self._cell_params.area,
            geofac_n2s=self._interpolation_state.geofac_n2s,
            geofac_div=self._interpolation_state.geofac_div,
            geofac_n2s_c=self._geofac_n2s_c,
            geofac_n2s_nbh=self._geofac_n2s_nbh,
            theta_ref_mc=self._metric_state.theta_ref_mc,
            zd_vertoffset=self._metric_state.zd_vertoffset,
            zd_diffcoef=self._metric_state.zd_diffcoef,
            zd_intcoef=self._metric_state.zd_intcoef,
            dtime=dtime,
            smag_offset=smag_offset,
            diff_multfac_w=self.diff_multfac_w,
            thresh_tdiff=self.thresh_tdiff,
            smallest_vpfloat=self._smallest_vpfloat,
            rd_o_cvd=self.rd_o_cvd,
            nrdmax=self._nrdmax,
            num_levels=num_levels,
            apply_to_temperature=self.config.apply_to_temperature,
            apply_zdiffusion_t=self.config.apply_zdiffusion_t,
            domain=(
                {dims.EdgeDim: self._edges, dims.KDim: (0, num_levels)},
                {dims.CellDim: self._cells, dims.KHalfDim: (0, num_levels + 1)},
                {dims.CellDim: self._cells, dims.KDim: (0, num_levels)},
                {dims.CellDim: self._cells, dims.KDim: (0, num_levels)},
            ),
            offset_provider=self._offset_provider,
        )

        return prognostics.PrognosticState(
            rho=prognostic_state.rho, w=w, vn=vn, exner=exner, theta_v=theta_v
        )
