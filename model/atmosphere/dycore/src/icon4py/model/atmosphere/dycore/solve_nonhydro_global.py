# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""
The nonhydrostatic solver for grids without lateral boundaries.

The same substep as `solve_nonhydro.SolveNonhydro.time_step`, but every stencil is a direct field
operator call instead of a program set up with `setup_program`, and the step is stateless: it
returns new states and never writes into its inputs.

The step does no halo exchange. Incremental analysis update is not supported.
"""

from __future__ import annotations

import dataclasses
from typing import Any, NamedTuple

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing
from gt4py.next.experimental import concat_where

import icon4py.model.common.grid.states as grid_states
import icon4py.model.common.utils as common_utils
from icon4py.model.atmosphere.dycore import dycore_states, dycore_utils
from icon4py.model.atmosphere.dycore.solve_nonhydro import (
    NonHydrostaticConfig,
    NonHydrostaticParams,
)
from icon4py.model.atmosphere.dycore.stencils.compute_cell_diagnostics_for_dycore import (
    _compute_interpolation_and_nonhydro_buoy,
    _compute_perturbed_quantities_and_interpolation,
)
from icon4py.model.atmosphere.dycore.stencils.compute_edge_diagnostics_for_dycore_and_update_vn import (
    _apply_divergence_damping_and_update_vn,
    _compute_rho_theta_pgrad_and_update_vn,
)
from icon4py.model.atmosphere.dycore.stencils.compute_horizontal_velocity_quantities import (
    _compute_averaged_vn_and_fluxes,
    _compute_horizontal_velocity_quantities_and_fluxes,
)
from icon4py.model.atmosphere.dycore.stencils.compute_hydrostatic_correction_term import (
    _compute_hydrostatic_correction_term,
)
from icon4py.model.atmosphere.dycore.stencils.extrapolate_at_top import _extrapolate_at_top
from icon4py.model.atmosphere.dycore.stencils.velocity_advection_corrector import (
    _compute_velocity_advection_in_corrector_step,
)
from icon4py.model.atmosphere.dycore.stencils.velocity_advection_predictor import (
    _compute_velocity_advection_in_predictor_step,
)
from icon4py.model.atmosphere.dycore.stencils.vertically_implicit_dycore_solver import (
    _compute_thermodynamic_variables_at_corrector_step,
    _compute_thermodynamic_variables_at_predictor_step,
    _interpolate_contravariant_correction_from_edges_on_model_levels_to_cells_on_half_levels,
    _set_surface_boundary_condition_for_computation_of_w,
    _set_top_boundary_condition_for_computation_of_w,
    _solve_w_and_update_vertical_fluxes_at_corrector_step,
    _solve_w_at_predictor_step,
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


def _merge(new: gtx.Field, old: gtx.Field, dim: gtx.Dimension, start: int, stop: int) -> gtx.Field:
    """`new` on the levels [start, stop) of `dim`, `old` elsewhere."""
    old_range = old.domain[dim].unit_range
    if start <= old_range.start and old_range.stop <= stop:
        return new
    return concat_where((dim >= start) & (dim < stop), new, old)


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
        if grid.config.distributed:
            raise ValueError("'SolveNonhydroGlobal' does not support distributed grids.")
        if config.iau_init:
            raise ValueError("'SolveNonhydroGlobal' does not support incremental analysis update.")
        assert cell_geometry.mean_cell_area is not None
        self._mean_cell_area: float = cell_geometry.mean_cell_area

        def unstructured(op: gtx_typing.FieldOperator) -> gtx_typing.FieldOperator:
            return op.with_grid_type(gtx.GridType.UNSTRUCTURED).with_backend(backend)

        self._velocity_advection_predictor = unstructured(
            _compute_velocity_advection_in_predictor_step
        )
        self._velocity_advection_corrector = unstructured(
            _compute_velocity_advection_in_corrector_step
        )
        self._perturbed_quantities_and_interpolation = unstructured(
            _compute_perturbed_quantities_and_interpolation
        )
        self._hydrostatic_correction_term = unstructured(_compute_hydrostatic_correction_term)
        self._rho_theta_pgrad_and_update_vn = unstructured(_compute_rho_theta_pgrad_and_update_vn)
        self._horizontal_velocity_quantities_and_fluxes = unstructured(
            _compute_horizontal_velocity_quantities_and_fluxes
        )
        self._extrapolate_at_top = unstructured(_extrapolate_at_top)
        self._interpolate_contravariant_correction_to_cells = unstructured(
            _interpolate_contravariant_correction_from_edges_on_model_levels_to_cells_on_half_levels
        )
        self._surface_boundary_condition_for_w = unstructured(
            _set_surface_boundary_condition_for_computation_of_w
        )
        self._top_boundary_condition_for_w = unstructured(
            _set_top_boundary_condition_for_computation_of_w
        )
        self._solve_w_at_predictor_step = unstructured(_solve_w_at_predictor_step)
        self._thermodynamic_variables_at_predictor_step = unstructured(
            _compute_thermodynamic_variables_at_predictor_step
        )
        self._interpolation_and_nonhydro_buoy = unstructured(
            _compute_interpolation_and_nonhydro_buoy
        )
        self._divergence_damping_and_update_vn = unstructured(
            _apply_divergence_damping_and_update_vn
        )
        self._averaged_vn_and_fluxes = unstructured(_compute_averaged_vn_and_fluxes)
        self._solve_w_at_corrector_step = unstructured(
            _solve_w_and_update_vertical_fluxes_at_corrector_step
        )
        self._thermodynamic_variables_at_corrector_step = unstructured(
            _compute_thermodynamic_variables_at_corrector_step
        )
        self._rayleigh_damping_factor = unstructured(dycore_utils._compute_rayleigh_damping_factor)

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

        num_levels = grid.num_levels
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
            domain={dims.KDim: (0, num_levels)},
            offset_provider={},
        )

    def _determine_horizontal_domains(self) -> None:
        # Every horizontal domain of the program-based solver covers the whole grid here, so each
        # output below is written over all points and only its vertical range needs a merge.
        def interior(dim: gtx.Dimension) -> tuple[gtx.int32, gtx.int32]:
            domain = h_grid.domain(dim)(h_grid.Zone.INTERIOR)
            start, end = self._grid.start_index(domain), self._grid.end_index(domain)
            assert (start, end) == (0, self._grid.size[dim])
            return gtx.int32(start), gtx.int32(end)

        self._cells = interior(dims.CellDim)
        self._edges = interior(dims.EdgeDim)

    def initial_intermediate_state(self) -> IntermediateState:
        return IntermediateState(
            tangential_wind_on_half_levels=data_alloc.zero_field(
                self._grid,
                dims.EdgeDim,
                dims.KHalfDim,
                dtype=ta.vpfloat,
                allocator=self._allocator,
            ),
            contravariant_correction_at_edges_on_model_levels=data_alloc.zero_field(
                self._grid, dims.EdgeDim, dims.KDim, dtype=ta.vpfloat, allocator=self._allocator
            ),
        )

    def _zeros(self, *field_dims: gtx.Dimension, **kwargs: Any) -> gtx.Field:
        return data_alloc.zero_field(
            self._grid, *field_dims, dtype=ta.vpfloat, allocator=self._allocator, **kwargs
        )

    def _cell_domain(self, dim: gtx.Dimension, start: int, stop: int) -> dict:
        return {dims.CellDim: self._cells, dim: (gtx.int32(start), gtx.int32(stop))}

    def _edge_domain(self, dim: gtx.Dimension, start: int, stop: int) -> dict:
        return {dims.EdgeDim: self._edges, dim: (gtx.int32(start), gtx.int32(stop))}

    def time_step(  # noqa: PLR0915 [too-many-statements]
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
        nflatlev = vertical.nflatlev
        damping_end = gtx.int32(vertical.end_index_of_damping_layer)
        kstart_moist = gtx.int32(vertical.kstart_moist)
        op = self._offset_provider
        current = prognostic_state
        diag = diagnostic_state_nh
        xp = data_alloc.array_namespace(current.vn.ndarray)

        rayleigh_damping_factor = self._rayleigh_damping_factor(
            metric.rayleigh_w, dtime, domain={dims.KHalfDim: (0, nlev + 1)}, offset_provider={}
        )

        tangential_wind = diag.tangential_wind
        tangential_wind_on_half_levels = intermediate_state.tangential_wind_on_half_levels
        vn_on_half_levels = diag.vn_on_half_levels
        contravariant_correction_at_edges = (
            intermediate_state.contravariant_correction_at_edges_on_model_levels
        )
        contravariant_correction_at_cells = diag.contravariant_correction_at_cells_on_half_levels
        predictor_w_advective_tendency, corrector_w_advective_tendency = (
            diag.vertical_wind_advective_tendency
        )
        predictor_vn_advective_tendency, _ = diag.normal_wind_advective_tendency
        max_vertical_cfl = diag.max_vertical_cfl

        # --- predictor ---
        if at_first_substep:
            skip_compute_predictor_vertical_advection = (
                config.itime_scheme == dycore_states.TimeSteppingScheme.MOST_EFFICIENT
                and not at_initial_timestep
            )
            (
                tangential_wind,
                new_tangential_wind_on_half_levels,
                vn_on_half_levels,
                _horizontal_kinetic_energy,
                new_contravariant_correction_at_edges,
                new_contravariant_correction_at_cells,
                new_predictor_w_advective_tendency,
                vertical_cfl,
                predictor_vn_advective_tendency,
            ) = self._velocity_advection_predictor(
                tangential_wind_on_half_levels=tangential_wind_on_half_levels,
                vertical_wind_advective_tendency=predictor_w_advective_tendency,
                vn=current.vn,
                w=current.w,
                rbf_vec_coeff_e=interp.rbf_vec_coeff_e,
                wgtfac_e=metric.wgtfac_e,
                wgtfacq_e=metric.wgtfacq_e,
                ddxn_z_full=metric.ddxn_z_full,
                ddxt_z_full=metric.ddxt_z_full,
                coeff1_dwdz=metric.coeff1_dwdz,
                coeff2_dwdz=metric.coeff2_dwdz,
                c_intp=interp.c_intp,
                inv_dual_edge_length=edge_geo.inverse_dual_edge_lengths,
                inv_primal_edge_length=edge_geo.inverse_primal_edge_lengths,
                tangent_orientation=edge_geo.tangent_orientation,
                e_bln_c_s=interp.e_bln_c_s,
                wgtfac_c=metric.wgtfac_c,
                ddqz_z_half=metric.ddqz_z_half,
                area=self._cell_geometry.area,
                geofac_n2s=interp.geofac_n2s,
                owner_mask=self._owner_mask,
                coriolis_frequency=edge_geo.coriolis_frequency,
                geofac_rot=interp.geofac_rot,
                coeff_gradekin=metric.coeff_gradekin,
                c_lin_e=interp.c_lin_e,
                ddqz_z_full_e=metric.ddqz_z_full_e,
                area_edge=edge_geo.edge_areas,
                geofac_grdiv=interp.geofac_grdiv,
                dtime=dtime,
                skip_compute_predictor_vertical_advection=skip_compute_predictor_vertical_advection,
                apply_extra_diffusion_on_vn=True,
                nflatlev=gtx.int32(nflatlev),
                nlev=gtx.int32(nlev),
                end_index_of_damping_layer=damping_end,
                domain=(
                    self._edge_domain(dims.KDim, 0, nlev),
                    self._edge_domain(dims.KHalfDim, 0, nlev),
                    self._edge_domain(dims.KHalfDim, 0, nlev + 1),
                    self._edge_domain(dims.KDim, 0, nlev),
                    self._edge_domain(dims.KDim, nflatlev, nlev),
                    self._cell_domain(dims.KHalfDim, 0, nlev),
                    self._cell_domain(dims.KHalfDim, 1, nlev),
                    self._cell_domain(dims.KHalfDim, 0, nlev),
                    self._edge_domain(dims.KDim, 0, nlev),
                ),
                offset_provider=op,
            )
            tangential_wind_on_half_levels = _merge(
                new_tangential_wind_on_half_levels,
                tangential_wind_on_half_levels,
                dims.KHalfDim,
                0,
                nlev,
            )
            contravariant_correction_at_edges = _merge(
                new_contravariant_correction_at_edges,
                contravariant_correction_at_edges,
                dims.KDim,
                nflatlev,
                nlev,
            )
            contravariant_correction_at_cells = _merge(
                new_contravariant_correction_at_cells,
                contravariant_correction_at_cells,
                dims.KHalfDim,
                0,
                nlev,
            )
            predictor_w_advective_tendency = _merge(
                new_predictor_w_advective_tendency,
                predictor_w_advective_tendency,
                dims.KHalfDim,
                1,
                nlev,
            )
            max_vertical_cfl = xp.maximum(xp.max(xp.abs(vertical_cfl.ndarray)), max_vertical_cfl)

        (
            perturbed_rho,
            perturbed_theta_v,
            new_perturbed_exner,
            new_rho_ic,
            new_theta_v_ic,
            new_nonhydro_buoy,
            new_exner_extrapolation,
            new_ddz_exner_extrapolation,
            new_d2dz2_exner_extrapolation,
        ) = self._perturbed_quantities_and_interpolation(
            time_extrapolation_parameter_for_exner=metric.time_extrapolation_parameter_for_exner,
            current_exner=current.exner,
            reference_exner_at_cells_on_model_levels=metric.reference_exner_at_cells_on_model_levels,
            current_rho=current.rho,
            reference_rho_at_cells_on_model_levels=metric.reference_rho_at_cells_on_model_levels,
            current_theta_v=current.theta_v,
            reference_theta_at_cells_on_model_levels=metric.reference_theta_at_cells_on_model_levels,
            wgtfac_c=metric.wgtfac_c,
            exner_w_explicit_weight_parameter=metric.exner_w_explicit_weight_parameter,
            perturbed_exner_at_cells_on_model_levels=diag.perturbed_exner_at_cells_on_model_levels,
            ddz_of_reference_exner_at_cells_on_half_levels=metric.ddz_of_reference_exner_at_cells_on_half_levels,
            ddqz_z_half=metric.ddqz_z_half,
            wgtfacq_c=metric.wgtfacq_c,
            reference_theta_at_cells_on_half_levels=metric.reference_theta_at_cells_on_half_levels,
            inv_ddqz_z_full=metric.inv_ddqz_z_full,
            d2dexdz2_fac1_mc=metric.d2dexdz2_fac1_mc,
            d2dexdz2_fac2_mc=metric.d2dexdz2_fac2_mc,
            ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels=self._zeros(
                dims.CellDim, dims.KDim
            ),
            d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels=self._zeros(
                dims.CellDim, dims.KDim
            ),
            igradp_method=gtx.int32(config.igradp_method),
            surface_level=gtx.int32(nlev + 1),
            domain=(
                self._cell_domain(dims.KDim, 0, nlev),
                self._cell_domain(dims.KDim, 0, nlev),
                self._cell_domain(dims.KDim, 0, nlev),
                self._cell_domain(dims.KHalfDim, 1, nlev),
                self._cell_domain(dims.KHalfDim, 1, nlev + 1),
                self._cell_domain(dims.KHalfDim, 1, nlev),
                self._cell_domain(dims.KDim, 0, nlev),
                self._cell_domain(dims.KDim, max(1, nflatlev), nlev),
                self._cell_domain(dims.KDim, max(1, metric.nflat_gradp), nlev),
            ),
            offset_provider=op,
        )
        perturbed_exner = new_perturbed_exner
        rho_ic = _merge(new_rho_ic, diag.rho_at_cells_on_half_levels, dims.KHalfDim, 1, nlev)
        theta_v_ic = _merge(
            new_theta_v_ic, diag.theta_v_at_cells_on_half_levels, dims.KHalfDim, 1, nlev + 1
        )
        nonhydro_buoy = _merge(
            new_nonhydro_buoy, self._zeros(dims.CellDim, dims.KHalfDim), dims.KHalfDim, 1, nlev
        )
        exner_extrapolation = _merge(
            new_exner_extrapolation,
            self._zeros(dims.CellDim, dims.KDim, extend={dims.KDim: 1}),
            dims.KDim,
            0,
            nlev,
        )
        ddz_exner_extrapolation = _merge(
            new_ddz_exner_extrapolation,
            self._zeros(dims.CellDim, dims.KDim),
            dims.KDim,
            max(1, nflatlev),
            nlev,
        )
        d2dz2_exner_extrapolation = _merge(
            new_d2dz2_exner_extrapolation,
            self._zeros(dims.CellDim, dims.KDim),
            dims.KDim,
            max(1, metric.nflat_gradp),
            nlev,
        )

        hydrostatic_correction = self._hydrostatic_correction_term(
            theta_v=current.theta_v,
            ikoffset=metric.vertoffset_gradp,
            zdiff_gradp=metric.zdiff_gradp,
            theta_v_ic=theta_v_ic,
            inv_ddqz_z_full=metric.inv_ddqz_z_full,
            inv_dual_edge_length=edge_geo.inverse_dual_edge_lengths,
            grav_o_cpd=constants.GRAV_O_CPD,
            domain=self._edge_domain(dims.KDim, nlev - 1, nlev),
            offset_provider=op,
        )
        hydrostatic_correction_on_lowest_level = gtx.as_field(
            (dims.EdgeDim,), hydrostatic_correction.ndarray[:, 0], allocator=self._allocator
        )

        (
            rho_at_edges,
            theta_v_at_edges,
            horizontal_pressure_gradient,
            next_vn,
        ) = self._rho_theta_pgrad_and_update_vn(
            next_vn=current.vn,
            current_vn=current.vn,
            tangential_wind=tangential_wind,
            reference_rho_at_edges_on_model_levels=metric.reference_rho_at_edges_on_model_levels,
            reference_theta_at_edges_on_model_levels=metric.reference_theta_at_edges_on_model_levels,
            perturbed_rho_at_cells_on_model_levels=perturbed_rho,
            perturbed_theta_v_at_cells_on_model_levels=perturbed_theta_v,
            temporal_extrapolation_of_perturbed_exner=exner_extrapolation,
            ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels=ddz_exner_extrapolation,
            d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels=d2dz2_exner_extrapolation,
            hydrostatic_correction_on_lowest_level=hydrostatic_correction_on_lowest_level,
            predictor_normal_wind_advective_tendency=predictor_vn_advective_tendency,
            normal_wind_tendency_due_to_slow_physics_process=diag.normal_wind_tendency_due_to_slow_physics_process,
            normal_wind_iau_increment=diag.normal_wind_iau_increment,
            grf_tend_vn=diag.grf_tend_vn,
            geofac_grg_x=interp.geofac_grg_x,
            geofac_grg_y=interp.geofac_grg_y,
            pos_on_tplane_e_x=interp.pos_on_tplane_e_1,
            pos_on_tplane_e_y=interp.pos_on_tplane_e_2,
            primal_normal_cell_x=edge_geo.primal_normal_cell[0],
            dual_normal_cell_x=edge_geo.dual_normal_cell[0],
            primal_normal_cell_y=edge_geo.primal_normal_cell[1],
            dual_normal_cell_y=edge_geo.dual_normal_cell[1],
            ddxn_z_full=metric.ddxn_z_full,
            c_lin_e=interp.c_lin_e,
            ikoffset=metric.vertoffset_gradp,
            zdiff_gradp=metric.zdiff_gradp,
            pg_exdist=metric.pg_exdist,
            inv_dual_edge_length=edge_geo.inverse_dual_edge_lengths,
            dtime=dtime,
            is_iau_active=False,
            iau_wgt_dyn=0.0,
            limited_area=False,
            nflatlev=gtx.int32(nflatlev),
            nflat_gradp=gtx.int32(metric.nflat_gradp),
            start_edge_lateral_boundary=self._edges[0],
            start_edge_lateral_boundary_level_7=self._edges[0],
            start_edge_nudging_level_2=self._edges[0],
            end_edge_nudging=self._edges[0],
            end_edge_local=self._edges[1],
            end_edge_halo=self._edges[1],
            domain=self._edge_domain(dims.KDim, 0, nlev),
            offset_provider=op,
        )

        (
            _spatially_averaged_vn,
            horizontal_gradient_of_normal_wind_divergence,
            tangential_wind,
            mass_flux_at_edges,
            theta_v_flux_at_edges,
            new_vn_on_half_levels,
            new_tangential_wind_on_half_levels,
            horizontal_kinetic_energy,
            contravariant_correction_at_edges,
        ) = self._horizontal_velocity_quantities_and_fluxes(
            contravariant_correction_at_edges_on_model_levels=contravariant_correction_at_edges,
            vn=next_vn,
            e_flx_avg=interp.e_flx_avg,
            geofac_grdiv=interp.geofac_grdiv,
            rbf_vec_coeff_e=interp.rbf_vec_coeff_e,
            rho_at_edges_on_model_levels=rho_at_edges,
            theta_v_at_edges_on_model_levels=theta_v_at_edges,
            ddqz_z_full_e=metric.ddqz_z_full_e,
            ddxn_z_full=metric.ddxn_z_full,
            ddxt_z_full=metric.ddxt_z_full,
            wgtfac_e=metric.wgtfac_e,
            nflatlev=gtx.int32(nflatlev),
            domain=(
                *(self._edge_domain(dims.KDim, 0, nlev),) * 5,
                *(self._edge_domain(dims.KHalfDim, 0, nlev),) * 2,
                *(self._edge_domain(dims.KDim, 0, nlev),) * 2,
            ),
            offset_provider=op,
        )
        vn_on_half_levels = concat_where(
            dims.KHalfDim < nlev,
            new_vn_on_half_levels,
            self._extrapolate_at_top(
                next_vn,
                metric.wgtfacq_e,
                domain=self._edge_domain(dims.KHalfDim, nlev, nlev + 1),
                offset_provider=op,
            ),
        )
        tangential_wind_on_half_levels = _merge(
            new_tangential_wind_on_half_levels,
            tangential_wind_on_half_levels,
            dims.KHalfDim,
            0,
            nlev,
        )

        contravariant_correction_at_cells = _merge(
            self._interpolate_contravariant_correction_to_cells(
                contravariant_correction_at_edges_on_model_levels=contravariant_correction_at_edges,
                e_bln_c_s=interp.e_bln_c_s,
                wgtfac_c=metric.wgtfac_c,
                wgtfacq_c=metric.wgtfacq_c,
                nlev=gtx.int32(nlev),
                domain=self._cell_domain(dims.KHalfDim, nflatlev + 1, nlev + 1),
                offset_provider=op,
            ),
            contravariant_correction_at_cells,
            dims.KHalfDim,
            nflatlev + 1,
            nlev + 1,
        )
        w_boundary = self._w_with_boundary_conditions(current.w, contravariant_correction_at_cells)
        next_w = self._solve_w_at_predictor_step(
            next_w=w_boundary,
            geofac_div=interp.geofac_div,
            mass_flux_at_edges_on_model_levels=mass_flux_at_edges,
            theta_v_flux_at_edges_on_model_levels=theta_v_flux_at_edges,
            predictor_vertical_wind_advective_tendency=predictor_w_advective_tendency,
            nonhydro_buoy_at_cells_on_half_levels=nonhydro_buoy,
            rho_at_cells_on_half_levels=rho_ic,
            contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells,
            exner_w_explicit_weight_parameter=metric.exner_w_explicit_weight_parameter,
            current_exner=current.exner,
            current_rho=current.rho,
            current_theta_v=current.theta_v,
            current_w=current.w,
            inv_ddqz_z_full=metric.inv_ddqz_z_full,
            exner_w_implicit_weight_parameter=metric.exner_w_implicit_weight_parameter,
            theta_v_at_cells_on_half_levels=theta_v_ic,
            perturbed_exner_at_cells_on_model_levels=perturbed_exner,
            exner_tendency_due_to_slow_physics=diag.exner_tendency_due_to_slow_physics,
            exner_iau_increment=diag.exner_iau_increment,
            ddqz_z_half=metric.ddqz_z_half,
            rayleigh_damping_factor=rayleigh_damping_factor,
            iau_wgt_dyn=0.0,
            dtime=dtime,
            rayleigh_type=gtx.int32(config.rayleigh_type),
            is_iau_active=False,
            end_index_of_damping_layer=damping_end,
            n_lev=gtx.int32(nlev),
            domain=self._cell_domain(dims.KHalfDim, 1, nlev),
            offset_provider=op,
        )
        next_w = _merge(next_w, w_boundary, dims.KHalfDim, 1, nlev)
        (
            next_rho,
            next_exner,
            next_theta_v,
            dwdz_at_cells,
            exner_dynamical_increment,
        ) = self._thermodynamic_variables_at_predictor_step(
            next_w=next_w,
            geofac_div=interp.geofac_div,
            mass_flux_at_edges_on_model_levels=mass_flux_at_edges,
            theta_v_flux_at_edges_on_model_levels=theta_v_flux_at_edges,
            rho_at_cells_on_half_levels=rho_ic,
            contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells,
            exner_w_explicit_weight_parameter=metric.exner_w_explicit_weight_parameter,
            current_exner=current.exner,
            current_rho=current.rho,
            current_theta_v=current.theta_v,
            current_w=current.w,
            inv_ddqz_z_full=metric.inv_ddqz_z_full,
            exner_w_implicit_weight_parameter=metric.exner_w_implicit_weight_parameter,
            theta_v_at_cells_on_half_levels=theta_v_ic,
            perturbed_exner_at_cells_on_model_levels=perturbed_exner,
            exner_tendency_due_to_slow_physics=diag.exner_tendency_due_to_slow_physics,
            rho_iau_increment=diag.rho_iau_increment,
            exner_iau_increment=diag.exner_iau_increment,
            exner_dynamical_increment=diag.exner_dynamical_increment,
            dwdz_at_cells_on_model_levels=self._zeros(dims.CellDim, dims.KDim),
            reference_exner_at_cells_on_model_levels=metric.reference_exner_at_cells_on_model_levels,
            iau_wgt_dyn=0.0,
            dtime=dtime,
            divdamp_type=gtx.int32(config.divdamp_type),
            is_iau_active=False,
            at_first_substep=at_first_substep,
            kstart_moist=kstart_moist,
            n_lev=gtx.int32(nlev),
            domain=self._cell_domain(dims.KDim, 0, nlev),
            offset_provider=op,
        )

        # --- corrector ---
        r_nsubsteps = 1.0 / ndyn_substeps_var
        second_order_divdamp_scaling_coeff = second_order_divdamp_factor * self._mean_cell_area

        (
            new_corrector_w_advective_tendency,
            vertical_cfl,
            corrector_vn_advective_tendency,
        ) = self._velocity_advection_corrector(
            vn=next_vn,
            w=next_w,
            tangential_wind=tangential_wind,
            tangential_wind_on_half_levels=tangential_wind_on_half_levels,
            vn_on_half_levels=vn_on_half_levels,
            horizontal_kinetic_energy_at_edges_on_model_levels=horizontal_kinetic_energy,
            contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells,
            coeff1_dwdz=metric.coeff1_dwdz,
            coeff2_dwdz=metric.coeff2_dwdz,
            c_intp=interp.c_intp,
            inv_dual_edge_length=edge_geo.inverse_dual_edge_lengths,
            inv_primal_edge_length=edge_geo.inverse_primal_edge_lengths,
            tangent_orientation=edge_geo.tangent_orientation,
            e_bln_c_s=interp.e_bln_c_s,
            ddqz_z_half=metric.ddqz_z_half,
            area=self._cell_geometry.area,
            geofac_n2s=interp.geofac_n2s,
            owner_mask=self._owner_mask,
            coriolis_frequency=edge_geo.coriolis_frequency,
            geofac_rot=interp.geofac_rot,
            coeff_gradekin=metric.coeff_gradekin,
            c_lin_e=interp.c_lin_e,
            ddqz_z_full_e=metric.ddqz_z_full_e,
            area_edge=edge_geo.edge_areas,
            geofac_grdiv=interp.geofac_grdiv,
            dtime=dtime,
            apply_extra_diffusion_on_vn=True,
            nlev=gtx.int32(nlev),
            end_index_of_damping_layer=damping_end,
            domain=(
                self._cell_domain(dims.KHalfDim, 1, nlev),
                self._cell_domain(dims.KHalfDim, 0, nlev),
                self._edge_domain(dims.KDim, 0, nlev),
            ),
            offset_provider=op,
        )
        corrector_w_advective_tendency = _merge(
            new_corrector_w_advective_tendency,
            corrector_w_advective_tendency,
            dims.KHalfDim,
            1,
            nlev,
        )
        max_vertical_cfl = xp.maximum(xp.max(xp.abs(vertical_cfl.ndarray)), max_vertical_cfl)

        new_rho_ic, new_theta_v_ic, new_nonhydro_buoy = self._interpolation_and_nonhydro_buoy(
            w=next_w,
            contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells,
            current_rho=current.rho,
            next_rho=next_rho,
            current_theta_v=current.theta_v,
            next_theta_v=next_theta_v,
            perturbed_exner_at_cells_on_model_levels=perturbed_exner,
            reference_theta_at_cells_on_model_levels=metric.reference_theta_at_cells_on_model_levels,
            ddz_of_reference_exner_at_cells_on_half_levels=metric.ddz_of_reference_exner_at_cells_on_half_levels,
            ddqz_z_half=metric.ddqz_z_half,
            wgtfac_c=metric.wgtfac_c,
            exner_w_explicit_weight_parameter=metric.exner_w_explicit_weight_parameter,
            dtime=dtime,
            rhotheta_explicit_weight_parameter=params.rhotheta_explicit_weight_parameter,
            rhotheta_implicit_weight_parameter=params.rhotheta_implicit_weight_parameter,
            domain=self._cell_domain(dims.KHalfDim, 1, nlev),
            offset_provider=op,
        )
        rho_ic = _merge(new_rho_ic, rho_ic, dims.KHalfDim, 1, nlev)
        theta_v_ic = _merge(new_theta_v_ic, theta_v_ic, dims.KHalfDim, 1, nlev)
        nonhydro_buoy = _merge(new_nonhydro_buoy, nonhydro_buoy, dims.KHalfDim, 1, nlev)

        apply_2nd_order_divergence_damping = (
            config.divdamp_order == dycore_states.DivergenceDampingOrder.SECOND_ORDER
            or (
                config.divdamp_order == dycore_states.DivergenceDampingOrder.COMBINED
                and second_order_divdamp_scaling_coeff > 1.0e-6
            )
        )
        apply_4th_order_divergence_damping = (
            config.divdamp_order == dycore_states.DivergenceDampingOrder.FOURTH_ORDER
            or (
                config.divdamp_order == dycore_states.DivergenceDampingOrder.COMBINED
                and second_order_divdamp_factor <= (4.0 * config.fourth_order_divdamp_factor)
            )
        )
        next_vn = self._divergence_damping_and_update_vn(
            horizontal_gradient_of_normal_wind_divergence=horizontal_gradient_of_normal_wind_divergence,
            next_vn=next_vn,
            current_vn=current.vn,
            dwdz_at_cells_on_model_levels=dwdz_at_cells,
            predictor_normal_wind_advective_tendency=predictor_vn_advective_tendency,
            corrector_normal_wind_advective_tendency=corrector_vn_advective_tendency,
            normal_wind_tendency_due_to_slow_physics_process=diag.normal_wind_tendency_due_to_slow_physics_process,
            normal_wind_iau_increment=diag.normal_wind_iau_increment,
            second_order_divdamp_scaling_coeff=second_order_divdamp_scaling_coeff,
            theta_v_at_edges_on_model_levels=theta_v_at_edges,
            horizontal_pressure_gradient=horizontal_pressure_gradient,
            horizontal_mask_for_3d_divdamp=metric.horizontal_mask_for_3d_divdamp,
            scaling_factor_for_3d_divdamp=metric.scaling_factor_for_3d_divdamp,
            inv_dual_edge_length=edge_geo.inverse_dual_edge_lengths,
            nudgecoeff_e=interp.nudgecoeff_e,
            geofac_grdiv=interp.geofac_grdiv,
            interpolated_fourth_order_divdamp_factor=self._interpolated_fourth_order_divdamp_factor,
            advection_explicit_weight_parameter=params.advection_explicit_weight_parameter,
            advection_implicit_weight_parameter=params.advection_implicit_weight_parameter,
            dtime=dtime,
            is_iau_active=False,
            iau_wgt_dyn=0.0,
            limited_area=False,
            apply_2nd_order_divergence_damping=apply_2nd_order_divergence_damping,
            apply_4th_order_divergence_damping=apply_4th_order_divergence_damping,
            divdamp_order=gtx.int32(config.divdamp_order),
            mean_cell_area=self._mean_cell_area,
            second_order_divdamp_factor=second_order_divdamp_factor,
            max_nudging_coefficient=0.0,
            dbl_eps=constants.DBL_EPS,
            domain=self._edge_domain(dims.KDim, 0, nlev),
            offset_provider=op,
        )

        (
            _spatially_averaged_vn,
            mass_flux_at_edges,
            theta_v_flux_at_edges,
            vn_traj,
            mass_flx_me,
        ) = self._averaged_vn_and_fluxes(
            substep_and_spatially_averaged_vn=prep_adv.vn_traj,
            substep_averaged_mass_flux=prep_adv.mass_flx_me,
            e_flx_avg=interp.e_flx_avg,
            vn=next_vn,
            rho_at_edges_on_model_levels=rho_at_edges,
            ddqz_z_full_e=metric.ddqz_z_full_e,
            theta_v_at_edges_on_model_levels=theta_v_at_edges,
            prepare_fluxes_for_advection=prepare_fluxes_for_advection,
            at_first_substep=at_first_substep,
            r_nsubsteps=r_nsubsteps,
            domain=self._edge_domain(dims.KDim, 0, nlev),
            offset_provider=op,
        )

        w_boundary = self._w_with_boundary_conditions(next_w, contravariant_correction_at_cells)
        (
            new_next_w,
            new_dynamical_mass_flux,
            new_dynamical_volumetric_flux,
        ) = self._solve_w_at_corrector_step(
            next_w=w_boundary,
            dynamical_vertical_mass_flux_at_cells_on_half_levels=prep_adv.dynamical_vertical_mass_flux_at_cells_on_half_levels,
            dynamical_vertical_volumetric_flux_at_cells_on_half_levels=prep_adv.dynamical_vertical_volumetric_flux_at_cells_on_half_levels,
            geofac_div=interp.geofac_div,
            mass_flux_at_edges_on_model_levels=mass_flux_at_edges,
            theta_v_flux_at_edges_on_model_levels=theta_v_flux_at_edges,
            predictor_vertical_wind_advective_tendency=predictor_w_advective_tendency,
            corrector_vertical_wind_advective_tendency=corrector_w_advective_tendency,
            nonhydro_buoy_at_cells_on_half_levels=nonhydro_buoy,
            rho_at_cells_on_half_levels=rho_ic,
            contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells,
            exner_w_explicit_weight_parameter=metric.exner_w_explicit_weight_parameter,
            current_exner=current.exner,
            current_rho=current.rho,
            current_theta_v=current.theta_v,
            current_w=current.w,
            inv_ddqz_z_full=metric.inv_ddqz_z_full,
            exner_w_implicit_weight_parameter=metric.exner_w_implicit_weight_parameter,
            theta_v_at_cells_on_half_levels=theta_v_ic,
            perturbed_exner_at_cells_on_model_levels=perturbed_exner,
            exner_tendency_due_to_slow_physics=diag.exner_tendency_due_to_slow_physics,
            exner_iau_increment=diag.exner_iau_increment,
            ddqz_z_half=metric.ddqz_z_half,
            rayleigh_damping_factor=rayleigh_damping_factor,
            advection_explicit_weight_parameter=params.advection_explicit_weight_parameter,
            advection_implicit_weight_parameter=params.advection_implicit_weight_parameter,
            prepare_fluxes_for_advection=prepare_fluxes_for_advection,
            r_nsubsteps=r_nsubsteps,
            iau_wgt_dyn=0.0,
            dtime=dtime,
            is_iau_active=False,
            rayleigh_type=gtx.int32(config.rayleigh_type),
            at_first_substep=at_first_substep,
            end_index_of_damping_layer=damping_end,
            n_lev=gtx.int32(nlev),
            domain=self._cell_domain(dims.KHalfDim, 1, nlev),
            offset_provider=op,
        )
        next_w = _merge(new_next_w, w_boundary, dims.KHalfDim, 1, nlev)
        dynamical_mass_flux = _merge(
            new_dynamical_mass_flux,
            prep_adv.dynamical_vertical_mass_flux_at_cells_on_half_levels,
            dims.KHalfDim,
            1,
            nlev,
        )
        dynamical_volumetric_flux = _merge(
            new_dynamical_volumetric_flux,
            prep_adv.dynamical_vertical_volumetric_flux_at_cells_on_half_levels,
            dims.KHalfDim,
            1,
            nlev,
        )
        (
            next_rho,
            next_exner,
            next_theta_v,
            exner_dynamical_increment,
        ) = self._thermodynamic_variables_at_corrector_step(
            next_w=next_w,
            exner_dynamical_increment=exner_dynamical_increment,
            geofac_div=interp.geofac_div,
            mass_flux_at_edges_on_model_levels=mass_flux_at_edges,
            theta_v_flux_at_edges_on_model_levels=theta_v_flux_at_edges,
            rho_at_cells_on_half_levels=rho_ic,
            contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells,
            exner_w_explicit_weight_parameter=metric.exner_w_explicit_weight_parameter,
            current_exner=current.exner,
            current_rho=current.rho,
            current_theta_v=current.theta_v,
            current_w=current.w,
            inv_ddqz_z_full=metric.inv_ddqz_z_full,
            exner_w_implicit_weight_parameter=metric.exner_w_implicit_weight_parameter,
            theta_v_at_cells_on_half_levels=theta_v_ic,
            perturbed_exner_at_cells_on_model_levels=perturbed_exner,
            exner_tendency_due_to_slow_physics=diag.exner_tendency_due_to_slow_physics,
            rho_iau_increment=diag.rho_iau_increment,
            exner_iau_increment=diag.exner_iau_increment,
            reference_exner_at_cells_on_model_levels=metric.reference_exner_at_cells_on_model_levels,
            ndyn_substeps_var=float(ndyn_substeps_var),
            iau_wgt_dyn=0.0,
            dtime=dtime,
            is_iau_active=False,
            at_last_substep=at_last_substep,
            kstart_moist=kstart_moist,
            n_lev=gtx.int32(nlev),
            domain=self._cell_domain(dims.KDim, 0, nlev),
            offset_provider=op,
        )

        next_state = prognostics.PrognosticState(
            rho=next_rho, w=next_w, vn=next_vn, exner=next_exner, theta_v=next_theta_v
        )
        next_diagnostic_state = dataclasses.replace(
            diag,
            max_vertical_cfl=xp.asarray(max_vertical_cfl),
            tangential_wind=tangential_wind,
            vn_on_half_levels=vn_on_half_levels,
            contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells,
            theta_v_at_cells_on_half_levels=theta_v_ic,
            perturbed_exner_at_cells_on_model_levels=perturbed_exner,
            rho_at_cells_on_half_levels=rho_ic,
            mass_flux_at_edges_on_model_levels=mass_flux_at_edges,
            normal_wind_advective_tendency=common_utils.PredictorCorrectorPair(
                predictor_vn_advective_tendency, corrector_vn_advective_tendency
            ),
            vertical_wind_advective_tendency=common_utils.PredictorCorrectorPair(
                predictor_w_advective_tendency, corrector_w_advective_tendency
            ),
            exner_dynamical_increment=exner_dynamical_increment,
        )
        next_intermediate_state = IntermediateState(
            tangential_wind_on_half_levels=tangential_wind_on_half_levels,
            contravariant_correction_at_edges_on_model_levels=contravariant_correction_at_edges,
        )
        next_prep_adv = dycore_states.PrepAdvection(
            vn_traj=vn_traj,
            mass_flx_me=mass_flx_me,
            dynamical_vertical_mass_flux_at_cells_on_half_levels=dynamical_mass_flux,
            dynamical_vertical_volumetric_flux_at_cells_on_half_levels=dynamical_volumetric_flux,
        )
        return next_state, next_diagnostic_state, next_intermediate_state, next_prep_adv

    def _w_with_boundary_conditions(
        self,
        w: fa.CellKHalfField[ta.wpfloat],
        contravariant_correction_at_cells: fa.CellKHalfField[ta.anyfloat],
    ) -> fa.CellKHalfField[ta.wpfloat]:
        """`w` with zero at the model top and the surface boundary condition at the bottom."""
        nlev = self._grid.num_levels
        surface = self._surface_boundary_condition_for_w(
            contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells,
            domain=self._cell_domain(dims.KHalfDim, nlev, nlev + 1),
            offset_provider=self._offset_provider,
        )
        top = self._top_boundary_condition_for_w(
            domain=self._cell_domain(dims.KHalfDim, 0, 1), offset_provider={}
        )
        return concat_where(dims.KHalfDim < 1, top, concat_where(dims.KHalfDim < nlev, w, surface))
