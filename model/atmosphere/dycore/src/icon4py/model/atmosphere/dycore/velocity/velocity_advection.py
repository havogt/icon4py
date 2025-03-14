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
from gt4py.next.common import Field
from gt4py.next.iterator.builtins import int32
from gt4py.next.iterator.embedded import np_as_located_field
from gt4py.next.program_processors.runners.gtfn import (
    run_gtfn,
    run_gtfn_cached,
    run_gtfn_imperative,
)

import icon4py.model.atmosphere.dycore.velocity.velocity_advection_program as velocity_prog
from icon4py.model.atmosphere.dycore.mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl import (
    mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl,
)
from icon4py.model.atmosphere.dycore.mo_math_divrot_rot_vertex_ri_dsl import (
    mo_math_divrot_rot_vertex_ri_dsl,
)
from icon4py.model.atmosphere.dycore.mo_velocity_advection_stencil_01 import (
    mo_velocity_advection_stencil_01,
)
from icon4py.model.atmosphere.dycore.mo_velocity_advection_stencil_02 import (
    mo_velocity_advection_stencil_02,
)
from icon4py.model.atmosphere.dycore.mo_velocity_advection_stencil_03 import (
    mo_velocity_advection_stencil_03,
)
from icon4py.model.atmosphere.dycore.mo_velocity_advection_stencil_07 import (
    mo_velocity_advection_stencil_07,
)
from icon4py.model.atmosphere.dycore.mo_velocity_advection_stencil_08 import (
    mo_velocity_advection_stencil_08,
)
from icon4py.model.atmosphere.dycore.mo_velocity_advection_stencil_15 import (
    mo_velocity_advection_stencil_15,
)
from icon4py.model.atmosphere.dycore.mo_velocity_advection_stencil_18 import (
    mo_velocity_advection_stencil_18,
)
from icon4py.model.atmosphere.dycore.mo_velocity_advection_stencil_19 import (
    mo_velocity_advection_stencil_19,
)
from icon4py.model.atmosphere.dycore.mo_velocity_advection_stencil_20 import (
    mo_velocity_advection_stencil_20,
)
from icon4py.model.atmosphere.dycore.state_utils.diagnostic_state import DiagnosticStateNonHydro
from icon4py.model.atmosphere.dycore.state_utils.interpolation_state import InterpolationState
from icon4py.model.atmosphere.dycore.state_utils.metric_state import MetricStateNonHydro
from icon4py.model.atmosphere.dycore.state_utils.utils import _allocate, _allocate_indices
from icon4py.model.common.dimension import CellDim, EdgeDim, KDim, VertexDim
from icon4py.model.common.grid.horizontal import EdgeParams, HorizontalMarkerIndex
from icon4py.model.common.grid.icon import IconGrid
from icon4py.model.common.grid.vertical import VerticalModelParams
from icon4py.model.common.states.prognostic_state import PrognosticState


cached_backend = run_gtfn_cached
compiled_backend = run_gtfn
imperative_backend = run_gtfn_imperative
backend = run_gtfn_cached
#


class VelocityAdvection:
    def __init__(
        self,
        grid: IconGrid,
        metric_state: MetricStateNonHydro,
        interpolation_state: InterpolationState,
        vertical_params: VerticalModelParams,
        edge_params: EdgeParams,
        owner_mask: Field[[CellDim], bool],
    ):
        self._initialized = False
        self.grid: IconGrid = grid
        self.metric_state: MetricStateNonHydro = metric_state
        self.interpolation_state: InterpolationState = interpolation_state
        self.vertical_params = vertical_params
        self.edge_params = edge_params
        self.c_owner_mask = owner_mask

        self.cfl_w_limit: float = 0.65
        self.scalfac_exdiff: float = 0.05
        self._allocate_local_fields()
        self._initialized = True

    @property
    def initialized(self):
        return self._initialized

    def _allocate_local_fields(self):
        self.z_w_v = _allocate(VertexDim, KDim, is_halfdim=True, grid=self.grid)
        self.z_v_grad_w = _allocate(EdgeDim, KDim, grid=self.grid)
        self.z_ekinh = _allocate(CellDim, KDim, grid=self.grid)
        self.z_w_concorr_mc = _allocate(CellDim, KDim, grid=self.grid)
        self.z_w_con_c = _allocate(CellDim, KDim, is_halfdim=True, grid=self.grid)
        self.zeta = _allocate(VertexDim, KDim, grid=self.grid)
        self.z_w_con_c_full = _allocate(CellDim, KDim, grid=self.grid)
        self.cfl_clipping = _allocate(CellDim, KDim, grid=self.grid, dtype=bool)
        self.levmask = _allocate(KDim, grid=self.grid, dtype=bool)
        self.vcfl_dsl = _allocate(CellDim, KDim, grid=self.grid)
        self.k_field = _allocate_indices(KDim, grid=self.grid, is_halfdim=True)

    def run_predictor_step(
        self,
        vn_only: bool,
        diagnostic_state: DiagnosticStateNonHydro,
        prognostic_state: PrognosticState,
        z_w_concorr_me: Field[[EdgeDim, KDim], float],
        z_kin_hor_e: Field[[EdgeDim, KDim], float],
        z_vt_ie: Field[[EdgeDim, KDim], float],
        dtime: float,
        ntnd: int,
        cell_areas: Field[[CellDim], float],
    ):
        cfl_w_limit, scalfac_exdiff = self._scale_factors_by_dtime(dtime)

        start_vertex_lb_plus1 = self.grid.get_start_index(
            VertexDim, HorizontalMarkerIndex.lateral_boundary(VertexDim) + 1
        )
        end_vertex_local_minus1 = self.grid.get_end_index(
            VertexDim, HorizontalMarkerIndex.local(VertexDim) - 1
        )

        start_edge_lb_plus4 = self.grid.get_start_index(
            EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 4
        )
        start_edge_lb_plus6 = self.grid.get_start_index(
            EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 6
        )
        start_edge_nudging_plus1 = self.grid.get_start_index(
            EdgeDim, HorizontalMarkerIndex.nudging(EdgeDim) + 1
        )
        end_edge_local = self.grid.get_end_index(EdgeDim, HorizontalMarkerIndex.local(EdgeDim))

        end_edge_local_minus1 = self.grid.get_end_index(
            EdgeDim, HorizontalMarkerIndex.local(EdgeDim) - 1
        )
        end_edge_local_minus2 = self.grid.get_end_index(
            EdgeDim, HorizontalMarkerIndex.local(EdgeDim) - 2
        )

        start_cell_lb_plus3 = self.grid.get_start_index(
            CellDim, HorizontalMarkerIndex.lateral_boundary(CellDim) + 3
        )
        start_cell_nudging = self.grid.get_start_index(
            CellDim, HorizontalMarkerIndex.nudging(CellDim)
        )
        end_cell_local = self.grid.get_end_index(CellDim, HorizontalMarkerIndex.local(CellDim))
        end_cell_local_minus1 = self.grid.get_end_index(
            CellDim, HorizontalMarkerIndex.local(CellDim) - 1
        )

        if not vn_only:
            mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl.with_backend(run_gtfn)(
                p_cell_in=prognostic_state.w,
                c_intp=self.interpolation_state.c_intp,
                p_vert_out=self.z_w_v,
                horizontal_start=start_vertex_lb_plus1,
                horizontal_end=end_vertex_local_minus1,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider={
                    "V2C": self.grid.get_offset_provider("V2C"),
                },
            )

        mo_math_divrot_rot_vertex_ri_dsl.with_backend(run_gtfn)(
            vec_e=prognostic_state.vn,
            geofac_rot=self.interpolation_state.geofac_rot,
            rot_vec=self.zeta,
            horizontal_start=start_vertex_lb_plus1,
            horizontal_end=end_vertex_local_minus1,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider={
                "V2E": self.grid.get_offset_provider("V2E"),
            },
        )

        mo_velocity_advection_stencil_01.with_backend(run_gtfn)(
            vn=prognostic_state.vn,
            rbf_vec_coeff_e=self.interpolation_state.rbf_vec_coeff_e,
            vt=diagnostic_state.vt,
            horizontal_start=start_edge_lb_plus4,
            horizontal_end=end_edge_local_minus2,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider={
                "E2C2E": self.grid.get_offset_provider("E2C2E"),
            },
        )

        mo_velocity_advection_stencil_02.with_backend(run_gtfn)(
            wgtfac_e=self.metric_state.wgtfac_e,
            vn=prognostic_state.vn,
            vt=diagnostic_state.vt,
            vn_ie=diagnostic_state.vn_ie,
            z_kin_hor_e=z_kin_hor_e,
            horizontal_start=start_edge_lb_plus4,
            horizontal_end=end_edge_local_minus2,
            vertical_start=1,
            vertical_end=self.grid.num_levels,
            offset_provider={
                "Koff": KDim,
            },
        )

        if not vn_only:
            mo_velocity_advection_stencil_03.with_backend(run_gtfn)(
                wgtfac_e=self.metric_state.wgtfac_e,
                vt=diagnostic_state.vt,
                z_vt_ie=z_vt_ie,
                horizontal_start=start_edge_lb_plus4,
                horizontal_end=end_edge_local_minus2,
                vertical_start=1,
                vertical_end=self.grid.num_levels,
                offset_provider={"Koff": KDim},
            )

        velocity_prog.fused_stencils_4_5_6.with_backend(run_gtfn)(
            vn=prognostic_state.vn,
            vt=diagnostic_state.vt,
            vn_ie=diagnostic_state.vn_ie,
            z_vt_ie=z_vt_ie,
            z_kin_hor_e=z_kin_hor_e,
            ddxn_z_full=self.metric_state.ddxn_z_full,
            ddxt_z_full=self.metric_state.ddxt_z_full,
            z_w_concorr_me=z_w_concorr_me,
            wgtfacq_e_dsl=self.metric_state.wgtfacq_e_dsl,
            k_field=self.k_field,
            nflatlev_startindex=self.vertical_params.nflatlev,
            nlev=self.grid.num_levels,
            horizontal_start=start_edge_lb_plus4,
            horizontal_end=end_edge_local_minus2,
            vertical_start=0,
            vertical_end=self.grid.num_levels + 1,
            offset_provider={
                "Koff": KDim,
            },
        )

        if not vn_only:
            mo_velocity_advection_stencil_07.with_backend(run_gtfn)(
                vn_ie=diagnostic_state.vn_ie,
                inv_dual_edge_length=self.edge_params.inverse_dual_edge_lengths,
                w=prognostic_state.w,
                z_vt_ie=z_vt_ie,
                inv_primal_edge_length=self.edge_params.inverse_primal_edge_lengths,
                tangent_orientation=self.edge_params.tangent_orientation,
                z_w_v=self.z_w_v,
                z_v_grad_w=self.z_v_grad_w,
                horizontal_start=start_edge_lb_plus6,
                horizontal_end=end_edge_local_minus1,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider={
                    "E2C": self.grid.get_offset_provider("E2C"),
                    "E2V": self.grid.get_offset_provider("E2V"),
                },
            )

        mo_velocity_advection_stencil_08.with_backend(run_gtfn)(
            z_kin_hor_e=z_kin_hor_e,
            e_bln_c_s=self.interpolation_state.e_bln_c_s,
            z_ekinh=self.z_ekinh,
            horizontal_start=start_cell_lb_plus3,
            horizontal_end=end_cell_local_minus1,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider={
                "C2E": self.grid.get_offset_provider("C2E"),
                "C2CE": self.grid.get_offset_provider("C2CE"),
            },
        )

        velocity_prog.fused_stencils_9_10.with_backend(run_gtfn)(
            z_w_concorr_me=z_w_concorr_me,
            e_bln_c_s=self.interpolation_state.e_bln_c_s,
            local_z_w_concorr_mc=self.z_w_concorr_mc,
            wgtfac_c=self.metric_state.wgtfac_c,
            w_concorr_c=diagnostic_state.w_concorr_c,
            k_field=self.k_field,
            nflatlev_startindex=self.vertical_params.nflatlev,
            nlev=self.grid.num_levels,
            horizontal_start=start_cell_lb_plus3,
            horizontal_end=end_cell_local_minus1,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider={
                "C2E": self.grid.get_offset_provider("C2E"),
                "C2CE": self.grid.get_offset_provider("C2CE"),
                "Koff": KDim,
            },
        )

        velocity_prog.fused_stencils_11_to_13.with_backend(run_gtfn)(
            w=prognostic_state.w,
            w_concorr_c=diagnostic_state.w_concorr_c,
            local_z_w_con_c=self.z_w_con_c,
            k_field=self.k_field,
            nflatlev_startindex=self.vertical_params.nflatlev,
            nlev=self.grid.num_levels,
            horizontal_start=start_cell_lb_plus3,
            horizontal_end=end_cell_local_minus1,
            vertical_start=0,
            vertical_end=self.grid.num_levels + 1,
            offset_provider={},
        )

        velocity_prog.fused_stencil_14.with_backend(run_gtfn)(
            ddqz_z_half=self.metric_state.ddqz_z_half,
            local_z_w_con_c=self.z_w_con_c,
            local_cfl_clipping=self.cfl_clipping,
            local_vcfl=self.vcfl_dsl,
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
            horizontal_start=start_cell_lb_plus3,
            horizontal_end=end_cell_local_minus1,
            vertical_start=int32(max(3, self.vertical_params.index_of_damping_layer - 2) - 1),
            vertical_end=int32(self.grid.num_levels - 3),
            offset_provider={},
        )

        self.levmask = np_as_located_field(KDim)(np.any(self.cfl_clipping, 0))

        mo_velocity_advection_stencil_15.with_backend(run_gtfn)(
            z_w_con_c=self.z_w_con_c,
            z_w_con_c_full=self.z_w_con_c_full,
            horizontal_start=start_cell_lb_plus3,
            horizontal_end=end_cell_local_minus1,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider={"Koff": KDim},
        )

        velocity_prog.fused_stencils_16_to_17.with_backend(run_gtfn)(
            w=prognostic_state.w,
            local_z_v_grad_w=self.z_v_grad_w,
            e_bln_c_s=self.interpolation_state.e_bln_c_s,
            local_z_w_con_c=self.z_w_con_c,
            coeff1_dwdz=self.metric_state.coeff1_dwdz,
            coeff2_dwdz=self.metric_state.coeff2_dwdz,
            ddt_w_adv=diagnostic_state.ddt_w_adv_pc[ntnd],
            horizontal_start=start_cell_nudging,
            horizontal_end=end_cell_local,
            vertical_start=1,
            vertical_end=self.grid.num_levels,
            offset_provider={
                "C2E": self.grid.get_offset_provider("C2E"),
                "C2CE": self.grid.get_offset_provider("C2CE"),
                "Koff": KDim,
            },
        )

        mo_velocity_advection_stencil_18.with_backend(run_gtfn)(
            levmask=self.levmask,
            cfl_clipping=self.cfl_clipping,
            owner_mask=self.c_owner_mask,
            z_w_con_c=self.z_w_con_c,
            ddqz_z_half=self.metric_state.ddqz_z_half,
            area=cell_areas,
            geofac_n2s=self.interpolation_state.geofac_n2s,
            w=prognostic_state.w,
            ddt_w_adv=diagnostic_state.ddt_w_adv_pc[ntnd],
            scalfac_exdiff=scalfac_exdiff,
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
            horizontal_start=start_cell_nudging,
            horizontal_end=end_cell_local,
            vertical_start=int32(max(3, self.vertical_params.index_of_damping_layer - 2) - 1),
            vertical_end=int32(self.grid.num_levels - 3),
            offset_provider={
                "C2E2CO": self.grid.get_offset_provider("C2E2CO"),
            },
        )

        # This behaviour needs to change for multiple blocks
        self.levelmask = self.levmask

        mo_velocity_advection_stencil_19.with_backend(run_gtfn)(
            z_kin_hor_e=z_kin_hor_e,
            coeff_gradekin=self.metric_state.coeff_gradekin,
            z_ekinh=self.z_ekinh,
            zeta=self.zeta,
            vt=diagnostic_state.vt,
            f_e=self.edge_params.f_e,
            c_lin_e=self.interpolation_state.c_lin_e,
            z_w_con_c_full=self.z_w_con_c_full,
            vn_ie=diagnostic_state.vn_ie,
            ddqz_z_full_e=self.metric_state.ddqz_z_full_e,
            ddt_vn_apc=diagnostic_state.ddt_vn_apc_pc[ntnd],
            horizontal_start=start_edge_nudging_plus1,
            horizontal_end=end_edge_local,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider={
                "E2C": self.grid.get_offset_provider("E2C"),
                "E2V": self.grid.get_offset_provider("E2V"),
                "E2EC": self.grid.get_offset_provider("E2EC"),
                "Koff": KDim,
            },
        )

        mo_velocity_advection_stencil_20.with_backend(run_gtfn)(
            levelmask=self.levelmask,
            c_lin_e=self.interpolation_state.c_lin_e,
            z_w_con_c_full=self.z_w_con_c_full,
            ddqz_z_full_e=self.metric_state.ddqz_z_full_e,
            area_edge=self.edge_params.edge_areas,
            tangent_orientation=self.edge_params.tangent_orientation,
            inv_primal_edge_length=self.edge_params.inverse_primal_edge_lengths,
            zeta=self.zeta,
            geofac_grdiv=self.interpolation_state.geofac_grdiv,
            vn=prognostic_state.vn,
            ddt_vn_apc=diagnostic_state.ddt_vn_apc_pc[ntnd],
            cfl_w_limit=cfl_w_limit,
            scalfac_exdiff=scalfac_exdiff,
            dtime=dtime,
            horizontal_start=start_edge_nudging_plus1,
            horizontal_end=end_edge_local,
            vertical_start=int32(max(3, self.vertical_params.index_of_damping_layer - 2) - 1),
            vertical_end=int32(self.grid.num_levels - 4),
            offset_provider={
                "E2C": self.grid.get_offset_provider("E2C"),
                "E2V": self.grid.get_offset_provider("E2V"),
                "E2C2EO": self.grid.get_offset_provider("E2C2EO"),
                "Koff": KDim,
            },
        )

    def _scale_factors_by_dtime(self, dtime):
        scaled_cfl_w_limit = self.cfl_w_limit / dtime
        scalfac_exdiff = self.scalfac_exdiff / (dtime * (0.85 - scaled_cfl_w_limit * dtime))
        return scaled_cfl_w_limit, scalfac_exdiff

    def run_corrector_step(
        self,
        vn_only: bool,
        diagnostic_state: DiagnosticStateNonHydro,
        prognostic_state: PrognosticState,
        z_kin_hor_e: Field[[EdgeDim, KDim], float],
        z_vt_ie: Field[[EdgeDim, KDim], float],
        dtime: float,
        ntnd: int,
        cell_areas: Field[[CellDim], float],
    ):
        cfl_w_limit, scalfac_exdiff = self._scale_factors_by_dtime(dtime)

        start_vertex_lb_plus1 = self.grid.get_start_index(
            VertexDim, HorizontalMarkerIndex.lateral_boundary(VertexDim) + 1
        )
        end_vertex_local_minus1 = self.grid.get_end_index(
            VertexDim, HorizontalMarkerIndex.local(VertexDim) - 1
        )

        start_edge_lb_plus6 = self.grid.get_start_index(
            EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 6
        )
        start_edge_nudging_plus1 = self.grid.get_start_index(
            EdgeDim, HorizontalMarkerIndex.nudging(EdgeDim) + 1
        )
        end_edge_local = self.grid.get_end_index(EdgeDim, HorizontalMarkerIndex.local(EdgeDim))
        end_edge_local_minus1 = self.grid.get_end_index(
            EdgeDim, HorizontalMarkerIndex.local(EdgeDim) - 1
        )

        start_cell_lb_plus3 = self.grid.get_start_index(
            CellDim, HorizontalMarkerIndex.lateral_boundary(CellDim) + 3
        )
        start_cell_nudging = self.grid.get_start_index(
            CellDim, HorizontalMarkerIndex.nudging(CellDim)
        )
        end_cell_local = self.grid.get_end_index(CellDim, HorizontalMarkerIndex.local(CellDim))
        end_cell_lb_minus1 = self.grid.get_end_index(
            CellDim, HorizontalMarkerIndex.local(CellDim) - 1
        )

        if not vn_only:
            mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl.with_backend(run_gtfn)(
                p_cell_in=prognostic_state.w,
                c_intp=self.interpolation_state.c_intp,
                p_vert_out=self.z_w_v,
                horizontal_start=start_vertex_lb_plus1,
                horizontal_end=end_vertex_local_minus1,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider={
                    "V2C": self.grid.get_offset_provider("V2C"),
                },
            )

        mo_math_divrot_rot_vertex_ri_dsl.with_backend(run_gtfn)(
            vec_e=prognostic_state.vn,
            geofac_rot=self.interpolation_state.geofac_rot,
            rot_vec=self.zeta,
            horizontal_start=start_vertex_lb_plus1,
            horizontal_end=end_vertex_local_minus1,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider={
                "V2E": self.grid.get_offset_provider("V2E"),
            },
        )

        if not vn_only:
            mo_velocity_advection_stencil_07.with_backend(run_gtfn)(
                vn_ie=diagnostic_state.vn_ie,
                inv_dual_edge_length=self.edge_params.inverse_dual_edge_lengths,
                w=prognostic_state.w,
                z_vt_ie=z_vt_ie,
                inv_primal_edge_length=self.edge_params.inverse_primal_edge_lengths,
                tangent_orientation=self.edge_params.tangent_orientation,
                z_w_v=self.z_w_v,
                z_v_grad_w=self.z_v_grad_w,
                horizontal_start=start_edge_lb_plus6,
                horizontal_end=end_edge_local_minus1,
                vertical_start=0,
                vertical_end=self.grid.num_levels,
                offset_provider={
                    "E2C": self.grid.get_offset_provider("E2C"),
                    "E2V": self.grid.get_offset_provider("E2V"),
                },
            )

        mo_velocity_advection_stencil_08.with_backend(run_gtfn)(
            z_kin_hor_e=z_kin_hor_e,
            e_bln_c_s=self.interpolation_state.e_bln_c_s,
            z_ekinh=self.z_ekinh,
            horizontal_start=start_cell_lb_plus3,
            horizontal_end=end_cell_lb_minus1,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider={
                "C2E": self.grid.get_offset_provider("C2E"),
                "C2CE": self.grid.get_offset_provider("C2CE"),
            },
        )

        velocity_prog.fused_stencils_11_to_13.with_backend(run_gtfn)(
            w=prognostic_state.w,
            w_concorr_c=diagnostic_state.w_concorr_c,
            local_z_w_con_c=self.z_w_con_c,
            k_field=self.k_field,
            nflatlev_startindex=self.vertical_params.nflatlev,
            nlev=self.grid.num_levels,
            horizontal_start=start_cell_lb_plus3,
            horizontal_end=end_cell_lb_minus1,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider={},
        )

        velocity_prog.fused_stencil_14.with_backend(run_gtfn)(
            ddqz_z_half=self.metric_state.ddqz_z_half,
            local_z_w_con_c=self.z_w_con_c,
            local_cfl_clipping=self.cfl_clipping,
            local_vcfl=self.vcfl_dsl,
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
            horizontal_start=start_cell_lb_plus3,
            horizontal_end=end_cell_lb_minus1,
            vertical_start=int32(max(3, self.vertical_params.index_of_damping_layer - 2)),
            vertical_end=int32(self.grid.num_levels - 3),
            offset_provider={},
        )

        self.levmask = np_as_located_field(KDim)(np.any(self.cfl_clipping, 0))

        mo_velocity_advection_stencil_15.with_backend(run_gtfn)(
            z_w_con_c=self.z_w_con_c,
            z_w_con_c_full=self.z_w_con_c_full,
            horizontal_start=start_cell_lb_plus3,
            horizontal_end=end_cell_lb_minus1,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider={"Koff": KDim},
        )

        velocity_prog.fused_stencils_16_to_17.with_backend(run_gtfn)(
            w=prognostic_state.w,
            local_z_v_grad_w=self.z_v_grad_w,
            e_bln_c_s=self.interpolation_state.e_bln_c_s,
            local_z_w_con_c=self.z_w_con_c,
            coeff1_dwdz=self.metric_state.coeff1_dwdz,
            coeff2_dwdz=self.metric_state.coeff2_dwdz,
            ddt_w_adv=diagnostic_state.ddt_w_adv_pc[ntnd],
            horizontal_start=start_cell_nudging,
            horizontal_end=end_cell_local,
            vertical_start=1,
            vertical_end=self.grid.num_levels,
            offset_provider={
                "C2E": self.grid.get_offset_provider("C2E"),
                "C2CE": self.grid.get_offset_provider("C2CE"),
                "Koff": KDim,
            },
        )

        mo_velocity_advection_stencil_18.with_backend(run_gtfn)(
            levmask=self.levmask,
            cfl_clipping=self.cfl_clipping,
            owner_mask=self.c_owner_mask,
            z_w_con_c=self.z_w_con_c,
            ddqz_z_half=self.metric_state.ddqz_z_half,
            area=cell_areas,
            geofac_n2s=self.interpolation_state.geofac_n2s,
            w=prognostic_state.w,
            ddt_w_adv=diagnostic_state.ddt_w_adv_pc[ntnd],
            scalfac_exdiff=scalfac_exdiff,
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
            horizontal_start=start_cell_nudging,
            horizontal_end=end_cell_local,
            vertical_start=int32(max(3, self.vertical_params.index_of_damping_layer - 2)),
            vertical_end=int32(self.grid.num_levels - 4),
            offset_provider={
                "C2E2CO": self.grid.get_offset_provider("C2E2CO"),
            },
        )

        # This behaviour needs to change for multiple blocks
        self.levelmask = self.levmask

        mo_velocity_advection_stencil_19.with_backend(run_gtfn)(
            z_kin_hor_e=z_kin_hor_e,
            coeff_gradekin=self.metric_state.coeff_gradekin,
            z_ekinh=self.z_ekinh,
            zeta=self.zeta,
            vt=diagnostic_state.vt,
            f_e=self.edge_params.f_e,
            c_lin_e=self.interpolation_state.c_lin_e,
            z_w_con_c_full=self.z_w_con_c_full,
            vn_ie=diagnostic_state.vn_ie,
            ddqz_z_full_e=self.metric_state.ddqz_z_full_e,
            ddt_vn_apc=diagnostic_state.ddt_vn_apc_pc[ntnd],
            horizontal_start=start_edge_nudging_plus1,
            horizontal_end=end_edge_local,
            vertical_start=0,
            vertical_end=self.grid.num_levels,
            offset_provider={
                "E2C": self.grid.get_offset_provider("E2C"),
                "E2V": self.grid.get_offset_provider("E2V"),
                "E2EC": self.grid.get_offset_provider("E2EC"),
                "Koff": KDim,
            },
        )

        mo_velocity_advection_stencil_20.with_backend(run_gtfn)(
            levelmask=self.levelmask,
            c_lin_e=self.interpolation_state.c_lin_e,
            z_w_con_c_full=self.z_w_con_c_full,
            ddqz_z_full_e=self.metric_state.ddqz_z_full_e,
            area_edge=self.edge_params.edge_areas,
            tangent_orientation=self.edge_params.tangent_orientation,
            inv_primal_edge_length=self.edge_params.inverse_primal_edge_lengths,
            zeta=self.zeta,
            geofac_grdiv=self.interpolation_state.geofac_grdiv,
            vn=prognostic_state.vn,
            ddt_vn_apc=diagnostic_state.ddt_vn_apc_pc[ntnd],
            cfl_w_limit=cfl_w_limit,
            scalfac_exdiff=scalfac_exdiff,
            dtime=dtime,
            horizontal_start=start_edge_nudging_plus1,
            horizontal_end=end_edge_local,
            vertical_start=int32(max(3, self.vertical_params.index_of_damping_layer - 2)),
            vertical_end=int32(self.grid.num_levels - 4),
            offset_provider={
                "E2C": self.grid.get_offset_provider("E2C"),
                "E2V": self.grid.get_offset_provider("E2V"),
                "E2C2EO": self.grid.get_offset_provider("E2C2EO"),
                "Koff": KDim,
            },
        )
