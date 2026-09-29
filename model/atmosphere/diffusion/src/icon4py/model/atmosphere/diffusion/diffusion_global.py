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
from icon4py.model.atmosphere.diffusion.stencils.diffusion_global_step import (
    CellNeighbourWeights,
    DiffusionCoefficients,
    SteepPointInterpolation,
    _diffusion_global_step,
)
from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.grid import horizontal as h_grid, icon as icon_grid, vertical as v_grid
from icon4py.model.common.math.differential_operators import (
    DiamondDirection,
    DiamondLengths,
    RbfVectorCoefficients,
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
        self._edge_params = edge_params
        self._cell_params = cell_params
        self._rbf_coeff = RbfVectorCoefficients(
            u=interpolation_state.rbf_coeff_1, v=interpolation_state.rbf_coeff_2
        )
        self._primal_normal = DiamondDirection(*edge_params.primal_normal_vert)
        self._dual_normal = DiamondDirection(*edge_params.dual_normal_vert)
        self._lengths = DiamondLengths(
            inv_l_p=edge_params.inverse_primal_edge_lengths,
            inv_l_vv=edge_params.inverse_vertex_vertex_lengths,
        )
        # mypy infers other dimension types for the sparse fields of the diffusion states
        self._n2s = CellNeighbourWeights(
            center=interpolation_state.geofac_n2s_c,
            neighbours=interpolation_state.geofac_n2s_nbh,  # type: ignore[arg-type]
        )
        self._steep = SteepPointInterpolation(
            diffusion_coefficient=metric_state.zd_diffcoef,
            vertical_offset=metric_state.zd_vertoffset,  # type: ignore[arg-type]
            weight=metric_state.zd_intcoef,  # type: ignore[arg-type]
        )
        self._offset_provider = grid.connectivities
        ndyn_substeps_as_float = float(ndyn_substeps)

        self.rd_o_cvd: float = constants.GAS_CONSTANT_DRY_AIR / (
            constants.CPD - constants.GAS_CONSTANT_DRY_AIR
        )
        #: threshold temperature deviation from neighboring grid points that activates extra diffusion against runaway cooling
        self.thresh_tdiff: float = -5.0
        self._determine_horizontal_domains()

        num_levels = self._grid.num_levels
        k4_dt, kh_limit, f_s = self._init_diffusion_local_fields_for_regular_timestep(
            params.K4,
            ndyn_substeps_as_float,
            *params.smagorinski_factor,
            *params.smagorinski_height,
            self._vertical_grid.interface_physical_height,
            domain={dims.KDim: (0, num_levels)},
            offset_provider={},
        )
        end_index_of_damping_layer = self._vertical_grid.end_index_of_damping_layer
        heights = self._vertical_grid.interface_physical_height
        k2w_damping_layer = data_alloc.zero_field(self._grid, dims.KHalfDim, allocator=allocator)
        if end_index_of_damping_layer > 0:
            k2w_damping_layer = concat_where(
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
                k2w_damping_layer,
            )
        self.coefficients = DiffusionCoefficients(
            f_s=f_s,
            kh_offset=0.25 * params.K4 * ndyn_substeps_as_float,
            kh_limit=kh_limit,
            k4_dt=k4_dt,
            kw_dt=min(1.0 / 48.0, params.K4W * ndyn_substeps_as_float),
            k2w_damping_layer=k2w_damping_layer,
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
    ) -> prognostics.PrognosticState:
        num_levels = self._grid.num_levels
        k_domain = {dims.KDim: (0, num_levels)}

        coefficients = self.coefficients
        if initial_run:
            k4_dt, kh_limit = self._setup_fields_for_initial_step(
                self._params.K4,
                self.config.hdiff_efdt_ratio,
                domain=k_domain,
                offset_provider={},
            )
            coefficients = coefficients._replace(kh_offset=0.0, kh_limit=kh_limit, k4_dt=k4_dt)

        vn, w, theta_v, exner = self._diffusion_global_step(
            vn=prognostic_state.vn,
            w=prognostic_state.w,
            theta_v=prognostic_state.theta_v,
            exner=prognostic_state.exner,
            coefficients=coefficients,
            rbf_coeff=self._rbf_coeff,
            primal_normal=self._primal_normal,
            dual_normal=self._dual_normal,
            lengths=self._lengths,
            tangent_orientation=self._edge_params.tangent_orientation,
            inv_dual_edge_length=self._edge_params.inverse_dual_edge_lengths,
            edge_area=self._edge_params.edge_areas,
            cell_area=self._cell_params.area,
            geofac_div=self._interpolation_state.geofac_div,
            n2s=self._n2s,
            theta_ref_mc=self._metric_state.theta_ref_mc,
            steep=self._steep,
            dtime=dtime,
            thresh_tdiff=self.thresh_tdiff,
            smallest_coefficient=constants.DBL_EPS,
            rd_o_cvd=self.rd_o_cvd,
            num_levels=gtx.int32(num_levels),
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
