# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""
The nonhydrostatic solver for grids without lateral boundaries.

The same substep as `solve_nonhydro.SolveNonhydro.time_step`, as one field operator, and
stateless: the step returns new states and never writes into its inputs.

The step does no halo exchange: on a distributed grid its inputs must already be valid on every halo
point it reads, and it computes its outputs on the owned points only. Incremental analysis update is
not supported.
"""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing

import icon4py.model.common.grid.states as grid_states
import icon4py.model.common.utils as common_utils
from icon4py.model.atmosphere.dycore import dycore_states
from icon4py.model.atmosphere.dycore.solve_nonhydro import (
    NonHydrostaticConfig,
    NonHydrostaticParams,
)
from icon4py.model.atmosphere.dycore.stencils.solve_nonhydro_global_step import (
    _solve_nonhydro_global_step,
)
from icon4py.model.common import (
    constants,
    dimension as dims,
    field_type_aliases as fa,
    type_alias as ta,
)
from icon4py.model.common.grid import horizontal as h_grid, icon as icon_grid, vertical as v_grid
from icon4py.model.common.math import smagorinsky
from icon4py.model.common.states import nonhydro_states, prognostic_state as prognostics
from icon4py.model.common.utils import data_allocation as data_alloc


class IntermediateState(NamedTuple):
    """Fields a substep reads before it writes them, carried from the previous substep."""

    tangential_wind_on_half_levels: fa.EdgeKHalfField[ta.anyfloat]
    contravariant_correction_at_edges_on_model_levels: fa.EdgeKField[ta.anyfloat]


class SolveNonhydroGlobal:
    """One dynamical substep on a grid without lateral boundaries."""

    def __init__(
        self,
        *,
        grid: icon_grid.IconGrid,
        config: NonHydrostaticConfig,
        params: NonHydrostaticParams,
        metric_state_nonhydro: dycore_states.MetricStateNonHydro,
        interpolation_state: dycore_states.InterpolationState,
        vertical_params: v_grid.VerticalGrid,
        edge_geometry: grid_states.EdgeParams,
        cell_geometry: grid_states.CellParams,
        owner_mask: fa.CellField[bool],
        allocator: gtx_typing.Allocator | None,
        backend: gtx_typing.Backend | None = None,
    ) -> None:
        if grid.limited_area:
            raise ValueError("'SolveNonhydroGlobal' does not support limited area grids.")
        if config.iau_init:
            raise ValueError("'SolveNonhydroGlobal' does not support incremental analysis update.")
        assert cell_geometry.mean_cell_area is not None
        self._mean_cell_area: float = cell_geometry.mean_cell_area

        def unstructured(op: gtx_typing.FieldOperator) -> gtx_typing.FieldOperator:
            return op.with_grid_type(gtx.GridType.UNSTRUCTURED).with_backend(backend)

        self._solve_nonhydro_global_step = unstructured(_solve_nonhydro_global_step)

        self._grid = grid
        self._config = config
        self._params = params
        self._metric_state = metric_state_nonhydro
        self._interpolation_state = interpolation_state
        self._vertical_params = vertical_params
        self._edge_geometry = edge_geometry
        self._cell_geometry = cell_geometry
        self._owner_mask = owner_mask
        self._allocator = allocator
        self._offset_provider = grid.connectivities
        self._determine_horizontal_domains()

        self._interpolated_fourth_order_divdamp_factor = unstructured(
            smagorinsky._en_smag_fac_for_zero_nshift
        )(
            vect_a=vertical_params.interface_physical_height,
            hdiff_smag_fac=config.fourth_order_divdamp_factor,
            hdiff_smag_fac2=config.fourth_order_divdamp_factor2,
            hdiff_smag_fac3=config.fourth_order_divdamp_factor3,
            hdiff_smag_fac4=config.fourth_order_divdamp_factor4,
            hdiff_smag_z=config.fourth_order_divdamp_z,
            hdiff_smag_z2=config.fourth_order_divdamp_z2,
            hdiff_smag_z3=config.fourth_order_divdamp_z3,
            hdiff_smag_z4=config.fourth_order_divdamp_z4,
            domain={dims.KDim: (0, grid.num_levels)},
            offset_provider={},
        )
        self._zeros_cells_on_half_levels = self._zeros(dims.CellDim, dims.KHalfDim)
        self._zeros_cells_on_model_levels = self._zeros(dims.CellDim, dims.KDim)
        self._zeros_cells_on_model_levels_extended = self._zeros(
            dims.CellDim, dims.KDim, extend={dims.KDim: 1}
        )

    def _determine_horizontal_domains(self) -> None:
        def interior(dim: gtx.Dimension) -> tuple[gtx.int32, gtx.int32]:
            domain = h_grid.domain(dim)(h_grid.Zone.INTERIOR)
            start, end = self._grid.start_index(domain), self._grid.end_index(domain)
            return gtx.int32(start), gtx.int32(end)

        self._cells = interior(dims.CellDim)
        self._edges = interior(dims.EdgeDim)

    def _zeros(
        self, *field_dims: gtx.Dimension, extend: dict[gtx.Dimension, int] | None = None
    ) -> gtx.Field:
        return data_alloc.zero_field(
            self._grid, *field_dims, dtype=ta.vpfloat, extend=extend, allocator=self._allocator
        )

    def initial_intermediate_state(self) -> IntermediateState:
        return IntermediateState(
            tangential_wind_on_half_levels=self._zeros(dims.EdgeDim, dims.KHalfDim),
            contravariant_correction_at_edges_on_model_levels=self._zeros(dims.EdgeDim, dims.KDim),
        )

    def time_step(
        self,
        *,
        diagnostic_state_nh: nonhydro_states.DiagnosticStateNonHydro,
        prognostic_state: prognostics.PrognosticState,
        intermediate_state: IntermediateState,
        prep_adv: dycore_states.PrepAdvection,
        second_order_divdamp_factor: float,
        dtime: float,
        ndyn_substeps_var: int,
        at_initial_timestep: bool,
        prepare_fluxes_for_advection: bool,
        at_first_substep: bool,
        at_last_substep: bool,
    ) -> tuple[
        prognostics.PrognosticState,
        nonhydro_states.DiagnosticStateNonHydro,
        IntermediateState,
        dycore_states.PrepAdvection,
    ]:
        """
        One dynamical substep from `prognostic_state`.

        Returns the next prognostic state and the updated diagnostic, intermediate and
        advection states.
        """
        metric = self._metric_state
        interp = self._interpolation_state
        edge_geo = self._edge_geometry
        config = self._config
        params = self._params
        vertical = self._vertical_params
        nlev = self._grid.num_levels
        diag = diagnostic_state_nh
        predictor_w_advective_tendency, corrector_w_advective_tendency = (
            diag.vertical_wind_advective_tendency
        )
        predictor_vn_advective_tendency, _ = diag.normal_wind_advective_tendency

        second_order_divdamp_scaling_coeff = second_order_divdamp_factor * self._mean_cell_area
        combined_divdamp = config.divdamp_order == dycore_states.DivergenceDampingOrder.COMBINED
        apply_2nd_order_divergence_damping = (
            config.divdamp_order == dycore_states.DivergenceDampingOrder.SECOND_ORDER
            or (combined_divdamp and second_order_divdamp_scaling_coeff > 1.0e-6)
        )
        apply_4th_order_divergence_damping = (
            config.divdamp_order == dycore_states.DivergenceDampingOrder.FOURTH_ORDER
            or (
                combined_divdamp
                and second_order_divdamp_factor <= (4.0 * config.fourth_order_divdamp_factor)
            )
        )

        cells_k = {dims.CellDim: self._cells, dims.KDim: (0, nlev)}
        cells_khalf = {dims.CellDim: self._cells, dims.KHalfDim: (0, nlev + 1)}
        edges_k = {dims.EdgeDim: self._edges, dims.KDim: (0, nlev)}
        edges_khalf = {dims.EdgeDim: self._edges, dims.KHalfDim: (0, nlev + 1)}
        out_domains = (
            edges_k,
            cells_khalf,
            cells_k,
            cells_k,
            cells_k,
            edges_k,
            edges_khalf,
            cells_khalf,
            cells_khalf,
            cells_k,
            cells_khalf,
            edges_k,
            edges_k,
            edges_k,
            cells_khalf,
            cells_khalf,
            cells_k,
            edges_khalf,
            edges_k,
            edges_k,
            edges_k,
            cells_khalf,
            cells_khalf,
            {dims.CellDim: self._cells},
        )
        out = self._solve_nonhydro_global_step(
            current_vn=prognostic_state.vn,
            current_w=prognostic_state.w,
            current_rho=prognostic_state.rho,
            current_exner=prognostic_state.exner,
            current_theta_v=prognostic_state.theta_v,
            tangential_wind=diag.tangential_wind,
            vn_on_half_levels=diag.vn_on_half_levels,
            contravariant_correction_at_cells_on_half_levels=diag.contravariant_correction_at_cells_on_half_levels,
            theta_v_at_cells_on_half_levels=diag.theta_v_at_cells_on_half_levels,
            perturbed_exner_at_cells_on_model_levels=diag.perturbed_exner_at_cells_on_model_levels,
            rho_at_cells_on_half_levels=diag.rho_at_cells_on_half_levels,
            exner_tendency_due_to_slow_physics=diag.exner_tendency_due_to_slow_physics,
            normal_wind_tendency_due_to_slow_physics_process=diag.normal_wind_tendency_due_to_slow_physics_process,
            predictor_normal_wind_advective_tendency=predictor_vn_advective_tendency,
            predictor_vertical_wind_advective_tendency=predictor_w_advective_tendency,
            corrector_vertical_wind_advective_tendency=corrector_w_advective_tendency,
            exner_dynamical_increment=diag.exner_dynamical_increment,
            rho_iau_increment=diag.rho_iau_increment,
            normal_wind_iau_increment=diag.normal_wind_iau_increment,
            exner_iau_increment=diag.exner_iau_increment,
            tangential_wind_on_half_levels=intermediate_state.tangential_wind_on_half_levels,
            contravariant_correction_at_edges_on_model_levels=intermediate_state.contravariant_correction_at_edges_on_model_levels,
            vn_traj=prep_adv.vn_traj,
            mass_flx_me=prep_adv.mass_flx_me,
            dynamical_vertical_mass_flux_at_cells_on_half_levels=prep_adv.dynamical_vertical_mass_flux_at_cells_on_half_levels,
            dynamical_vertical_volumetric_flux_at_cells_on_half_levels=prep_adv.dynamical_vertical_volumetric_flux_at_cells_on_half_levels,
            zeros_cells_on_half_levels=self._zeros_cells_on_half_levels,
            zeros_cells_on_model_levels=self._zeros_cells_on_model_levels,
            zeros_cells_on_model_levels_extended=self._zeros_cells_on_model_levels_extended,
            rayleigh_w=metric.rayleigh_w,
            wgtfac_c=metric.wgtfac_c,
            wgtfacq_c=metric.wgtfacq_c,
            wgtfac_e=metric.wgtfac_e,
            wgtfacq_e=metric.wgtfacq_e,
            time_extrapolation_parameter_for_exner=metric.time_extrapolation_parameter_for_exner,
            reference_exner_at_cells_on_model_levels=metric.reference_exner_at_cells_on_model_levels,
            reference_rho_at_cells_on_model_levels=metric.reference_rho_at_cells_on_model_levels,
            reference_theta_at_cells_on_model_levels=metric.reference_theta_at_cells_on_model_levels,
            reference_rho_at_edges_on_model_levels=metric.reference_rho_at_edges_on_model_levels,
            reference_theta_at_edges_on_model_levels=metric.reference_theta_at_edges_on_model_levels,
            reference_theta_at_cells_on_half_levels=metric.reference_theta_at_cells_on_half_levels,
            ddz_of_reference_exner_at_cells_on_half_levels=metric.ddz_of_reference_exner_at_cells_on_half_levels,
            ddqz_z_half=metric.ddqz_z_half,
            d2dexdz2_fac1_mc=metric.d2dexdz2_fac1_mc,
            d2dexdz2_fac2_mc=metric.d2dexdz2_fac2_mc,
            ddxn_z_full=metric.ddxn_z_full,
            ddqz_z_full_e=metric.ddqz_z_full_e,
            ddxt_z_full=metric.ddxt_z_full,
            inv_ddqz_z_full=metric.inv_ddqz_z_full,
            vertoffset_gradp=metric.vertoffset_gradp,
            zdiff_gradp=metric.zdiff_gradp,
            pg_exdist=metric.pg_exdist,
            exner_w_explicit_weight_parameter=metric.exner_w_explicit_weight_parameter,
            exner_w_implicit_weight_parameter=metric.exner_w_implicit_weight_parameter,
            horizontal_mask_for_3d_divdamp=metric.horizontal_mask_for_3d_divdamp,
            scaling_factor_for_3d_divdamp=metric.scaling_factor_for_3d_divdamp,
            coeff1_dwdz=metric.coeff1_dwdz,
            coeff2_dwdz=metric.coeff2_dwdz,
            coeff_gradekin=metric.coeff_gradekin,
            e_bln_c_s=interp.e_bln_c_s,
            geofac_div=interp.geofac_div,
            geofac_n2s=interp.geofac_n2s,
            geofac_grg_x=interp.geofac_grg_x,
            geofac_grg_y=interp.geofac_grg_y,
            nudgecoeff_e=interp.nudgecoeff_e,
            c_lin_e=interp.c_lin_e,
            geofac_grdiv=interp.geofac_grdiv,
            rbf_vec_coeff_e=interp.rbf_vec_coeff_e,
            c_intp=interp.c_intp,
            geofac_rot=interp.geofac_rot,
            pos_on_tplane_e_1=interp.pos_on_tplane_e_1,
            pos_on_tplane_e_2=interp.pos_on_tplane_e_2,
            e_flx_avg=interp.e_flx_avg,
            inv_dual_edge_length=edge_geo.inverse_dual_edge_lengths,
            inv_primal_edge_length=edge_geo.inverse_primal_edge_lengths,
            tangent_orientation=edge_geo.tangent_orientation,
            coriolis_frequency=edge_geo.coriolis_frequency,
            edge_area=edge_geo.edge_areas,
            primal_normal_cell_x=edge_geo.primal_normal_cell[0],
            primal_normal_cell_y=edge_geo.primal_normal_cell[1],
            dual_normal_cell_x=edge_geo.dual_normal_cell[0],
            dual_normal_cell_y=edge_geo.dual_normal_cell[1],
            cell_area=self._cell_geometry.area,
            owner_mask=self._owner_mask,
            interpolated_fourth_order_divdamp_factor=self._interpolated_fourth_order_divdamp_factor,
            dtime=dtime,
            second_order_divdamp_factor=second_order_divdamp_factor,
            second_order_divdamp_scaling_coeff=second_order_divdamp_scaling_coeff,
            r_nsubsteps=1.0 / ndyn_substeps_var,
            ndyn_substeps_var=float(ndyn_substeps_var),
            mean_cell_area=self._mean_cell_area,
            advection_explicit_weight_parameter=params.advection_explicit_weight_parameter,
            advection_implicit_weight_parameter=params.advection_implicit_weight_parameter,
            rhotheta_explicit_weight_parameter=params.rhotheta_explicit_weight_parameter,
            rhotheta_implicit_weight_parameter=params.rhotheta_implicit_weight_parameter,
            grav_o_cpd=constants.GRAV_O_CPD,
            dbl_eps=constants.DBL_EPS,
            at_first_substep=at_first_substep,
            at_last_substep=at_last_substep,
            skip_compute_predictor_vertical_advection=(
                config.itime_scheme == dycore_states.TimeSteppingScheme.MOST_EFFICIENT
                and not at_initial_timestep
            ),
            prepare_fluxes_for_advection=prepare_fluxes_for_advection,
            apply_2nd_order_divergence_damping=apply_2nd_order_divergence_damping,
            apply_4th_order_divergence_damping=apply_4th_order_divergence_damping,
            igradp_method=gtx.int32(config.igradp_method),
            rayleigh_type=gtx.int32(config.rayleigh_type),
            divdamp_type=gtx.int32(config.divdamp_type),
            divdamp_order=gtx.int32(config.divdamp_order),
            nlev=gtx.int32(nlev),
            nflatlev=gtx.int32(vertical.nflatlev),
            nflat_gradp=gtx.int32(metric.nflat_gradp),
            start_of_ddz_of_exner_extrapolation=gtx.int32(max(1, vertical.nflatlev)),
            start_of_d2dz2_of_exner_extrapolation=gtx.int32(max(1, metric.nflat_gradp)),
            end_index_of_damping_layer=gtx.int32(vertical.end_index_of_damping_layer),
            kstart_moist=gtx.int32(vertical.kstart_moist),
            domain=out_domains,
            offset_provider=self._offset_provider,
        )
        (
            next_vn,
            next_w,
            next_rho,
            next_exner,
            next_theta_v,
            tangential_wind,
            vn_on_half_levels,
            contravariant_correction_at_cells_on_half_levels,
            theta_v_at_cells_on_half_levels,
            perturbed_exner_at_cells_on_model_levels,
            rho_at_cells_on_half_levels,
            mass_flux_at_edges_on_model_levels,
            predictor_vn_advective_tendency,
            corrector_vn_advective_tendency,
            predictor_w_advective_tendency,
            corrector_w_advective_tendency,
            exner_dynamical_increment,
            tangential_wind_on_half_levels,
            contravariant_correction_at_edges_on_model_levels,
            vn_traj,
            mass_flx_me,
            dynamical_vertical_mass_flux_at_cells_on_half_levels,
            dynamical_vertical_volumetric_flux_at_cells_on_half_levels,
            max_vertical_cfl_on_cells,
        ) = out

        xp = data_alloc.array_namespace(max_vertical_cfl_on_cells.ndarray)
        next_diagnostic_state = dataclasses.replace(
            diag,
            max_vertical_cfl=xp.asarray(
                xp.maximum(xp.max(max_vertical_cfl_on_cells.ndarray), diag.max_vertical_cfl)
            ),
            tangential_wind=tangential_wind,
            vn_on_half_levels=vn_on_half_levels,
            contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
            theta_v_at_cells_on_half_levels=theta_v_at_cells_on_half_levels,
            perturbed_exner_at_cells_on_model_levels=perturbed_exner_at_cells_on_model_levels,
            rho_at_cells_on_half_levels=rho_at_cells_on_half_levels,
            mass_flux_at_edges_on_model_levels=mass_flux_at_edges_on_model_levels,
            normal_wind_advective_tendency=common_utils.PredictorCorrectorPair(
                predictor_vn_advective_tendency, corrector_vn_advective_tendency
            ),
            vertical_wind_advective_tendency=common_utils.PredictorCorrectorPair(
                predictor_w_advective_tendency, corrector_w_advective_tendency
            ),
            exner_dynamical_increment=exner_dynamical_increment,
        )
        return (
            prognostics.PrognosticState(
                rho=next_rho, w=next_w, vn=next_vn, exner=next_exner, theta_v=next_theta_v
            ),
            next_diagnostic_state,
            IntermediateState(
                tangential_wind_on_half_levels=tangential_wind_on_half_levels,
                contravariant_correction_at_edges_on_model_levels=contravariant_correction_at_edges_on_model_levels,
            ),
            dycore_states.PrepAdvection(
                vn_traj=vn_traj,
                mass_flx_me=mass_flx_me,
                dynamical_vertical_mass_flux_at_cells_on_half_levels=dynamical_vertical_mass_flux_at_cells_on_half_levels,
                dynamical_vertical_volumetric_flux_at_cells_on_half_levels=dynamical_vertical_volumetric_flux_at_cells_on_half_levels,
            ),
        )
