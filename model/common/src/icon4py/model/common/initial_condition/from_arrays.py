# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""
An initial state from column arrays, balanced in the dynamical core's discrete form.

The arrays are on the grid file's cells and the model's full levels. As in the
Jablonowski-Williamson initial condition, the thermodynamic state is brought into the discrete
hydrostatic balance of the dynamical core (`apply_hydrostatic_adjustment_ndarray`), so the
temperature of the balanced state, a diagnostic of theta_v and exner, differs from the given one
at the level of the discretisation. The surface pressure that icon4py diagnoses from the balanced
state (`compute_pressure._compute_surface_pressure`) is the given one.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from icon4py.model.common import (
    constants as phy_const,
    dimension as dims,
    model_backends,
    type_alias as ta,
)
from icon4py.model.common.grid import geometry_attributes as geometry_meta, vertical as v_grid
from icon4py.model.common.initial_condition.analytical import utils as testcases_utils
from icon4py.model.common.interpolation import interpolation_attributes
from icon4py.model.common.metrics import metrics_attributes
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.decomposition import definitions as decomposition_defs
    from icon4py.model.common.grid import icon as icon_grid
    from icon4py.model.common.states import (
        prognostic_state as prognostics,
        static_fields,
        tracer_states,
    )


MOISTURE = ("qv", "qc", "qi", "qr", "qs", "qg")


@dataclasses.dataclass(frozen=True)
class ColumnInitialState:
    """The state on the grid file's cells (axis 0) and the model's full levels (axis 1)."""

    #: eastward and northward wind at the cell centres [m/s]
    u: data_alloc.NDArray
    v: data_alloc.NDArray
    #: air temperature [K]
    temperature: data_alloc.NDArray
    #: specific masses of water vapour and the hydrometeors, of total air [kg/kg], by tracer name
    q: dict[str, data_alloc.NDArray]
    #: surface pressure, per cell [Pa]
    surface_pressure: data_alloc.NDArray


def _diagnosed_surface_pressure(exner, virtual_temperature, ddqz_z_full):  # type: ignore[no-untyped-def]
    """`compute_pressure._compute_surface_pressure` on arrays."""
    array_ns = data_alloc.array_namespace(exner)
    return phy_const.P0REF * array_ns.exp(
        phy_const.CPD_O_RD * array_ns.log(exner[:, -3])
        + phy_const.GRAV_O_RD
        * (
            ddqz_z_full[:, -1] / virtual_temperature[:, -1]
            + ddqz_z_full[:, -2] / virtual_temperature[:, -2]
            + 0.5 * ddqz_z_full[:, -3] / virtual_temperature[:, -3]
        )
    )


def initialize_from_arrays(
    *,
    state: ColumnInitialState,
    vertical_grid_config: v_grid.VerticalGridConfig,
    grid: icon_grid.IconGrid,
    decomposition_info: decomposition_defs.DecompositionInfo,
    static_fields: static_fields.StaticFieldFactories,
    prognostic_state_now: prognostics.PrognosticState,
    tracer_state_now: tracer_states.TracerState,
    backend: gtx_typing.Backend | None,
    exchange: decomposition_defs.ExchangeRuntime,
    num_iterations: int = 20,
) -> None:
    """
    Fill the prognostic and tracer states from `state`.

    1. The virtual temperature follows from the temperature and the moisture.
    2. A first exner profile integrates the hydrostatic equation from the surface pressure over
       the model's heights.
    3. The discrete hydrostatic adjustment of the dynamical core then integrates exner upward from
       the lowest level, at fixed virtual temperature, with theta_v and rho following; the lowest
       exner is iterated (secant) until the surface pressure icon4py diagnoses is the given one.
    4. vn is the cell wind interpolated to the edges (c_lin_e) projected on the edge normals, w
       follows from vn over the topography as in the analytic initial conditions.
    """
    metrics = static_fields.metrics
    geometry = static_fields.geometry
    interpolation = static_fields.interpolation
    allocator = model_backends.get_allocator(backend)
    array_ns = data_alloc.import_array_ns(allocator)

    cells = array_ns.asarray(
        decomposition_info.global_index(dims.CellDim, decomposition_info.EntryType.ALL)
    )

    def local(array: data_alloc.NDArray) -> data_alloc.NDArray:
        return array_ns.asarray(array, dtype=array_ns.float64)[cells]

    missing = [name for name in MOISTURE if name not in state.q]
    if missing:
        raise ValueError(f"The initial state needs all moisture species, missing: {missing}.")
    q = {name: local(state.q[name]) for name in MOISTURE}
    temperature = local(state.temperature)
    surface_pressure = local(state.surface_pressure)
    virtual_temperature = temperature * (
        1.0
        + phy_const.RV_O_RD_MINUS_1 * q["qv"]
        - (q["qc"] + q["qi"] + q["qr"] + q["qs"] + q["qg"])
    )

    z_ifc = metrics.get_full_precision(metrics_attributes.CELL_HEIGHT_ON_HALF_LEVEL).ndarray
    z_mc = metrics.get_full_precision(metrics_attributes.Z_MC).ndarray
    ddqz_z_full = metrics.get_full_precision(metrics_attributes.DDQZ_Z_FULL).ndarray
    adjustment_metrics = {
        "exner_ref_mc": metrics.get_full_precision(metrics_attributes.EXNER_REF_MC).ndarray,
        "d_exner_dz_ref_ic": metrics.get_full_precision(
            metrics_attributes.D_EXNER_DZ_REF_IC
        ).ndarray,
        "theta_ref_mc": metrics.get_full_precision(metrics_attributes.THETA_REF_MC).ndarray,
        "theta_ref_ic": metrics.get_full_precision(metrics_attributes.THETA_REF_IC).ndarray,
        "wgtfac_c": metrics.get_full_precision(metrics_attributes.WGTFAC_C).ndarray,
        "ddqz_z_half": metrics.get_full_precision(metrics_attributes.DDQZ_Z_HALF).ndarray,
    }
    num_levels = grid.num_levels

    # the hydrostatic equation integrated from the surface pressure, layer by layer
    pressure = array_ns.empty_like(virtual_temperature)
    pressure[:, -1] = surface_pressure * array_ns.exp(
        -phy_const.GRAV_O_RD * (z_mc[:, -1] - z_ifc[:, -1]) / virtual_temperature[:, -1]
    )
    for k in range(num_levels - 2, -1, -1):
        mean_virtual_temperature = 0.5 * (virtual_temperature[:, k] + virtual_temperature[:, k + 1])
        pressure[:, k] = pressure[:, k + 1] * array_ns.exp(
            -phy_const.GRAV_O_RD * (z_mc[:, k] - z_mc[:, k + 1]) / mean_virtual_temperature
        )
    first_guess = (pressure / phy_const.P0REF) ** phy_const.RD_O_CPD

    def balanced(lowest_exner):  # type: ignore[no-untyped-def]
        exner = first_guess.copy()
        exner[:, -1] = lowest_exner
        theta_v = virtual_temperature / exner
        rho = exner**phy_const.CVD_O_RD * phy_const.P0REF / (phy_const.RD * theta_v)
        testcases_utils.apply_hydrostatic_adjustment_ndarray(
            rho=rho,
            exner=exner,
            theta_v=theta_v,
            num_levels=num_levels,
            **adjustment_metrics,
        )
        return exner, theta_v, rho

    def misfit(lowest_exner):  # type: ignore[no-untyped-def]
        exner, theta_v, _ = balanced(lowest_exner)
        return array_ns.log(
            _diagnosed_surface_pressure(exner, theta_v * exner, ddqz_z_full) / surface_pressure
        )

    previous, current = first_guess[:, -1], first_guess[:, -1] * (1.0 + 1.0e-4)
    previous_misfit, current_misfit = misfit(previous), misfit(current)
    for _ in range(num_iterations):
        if float(array_ns.max(array_ns.abs(current_misfit))) < 1.0e-15:
            break
        step = array_ns.where(
            current_misfit != previous_misfit,
            current_misfit * (current - previous) / (current_misfit - previous_misfit),
            0.0,
        )
        previous, previous_misfit = current, current_misfit
        current = current - step
        current_misfit = misfit(current)
    exner, theta_v, rho = balanced(current)

    e2c = grid.get_connectivity(dims.E2C).ndarray
    c_lin_e = interpolation.get_full_precision(interpolation_attributes.C_LIN_E).ndarray
    u, v = local(state.u), local(state.v)
    u_edge = c_lin_e[:, 0:1] * u[e2c[:, 0]] + c_lin_e[:, 1:2] * u[e2c[:, 1]]
    v_edge = c_lin_e[:, 0:1] * v[e2c[:, 0]] + c_lin_e[:, 1:2] * v[e2c[:, 1]]
    normal_u = geometry.get_full_precision(geometry_meta.EDGE_NORMAL_U).ndarray
    normal_v = geometry.get_full_precision(geometry_meta.EDGE_NORMAL_V).ndarray
    vn = u_edge * normal_u[:, None] + v_edge * normal_v[:, None]

    _, vct_b = v_grid.get_vct_a_and_vct_b(vertical_grid_config, allocator)
    w = testcases_utils.init_w(
        grid=grid,
        z_ifc=z_ifc,
        inv_dual_edge_length=geometry.get_full_precision(
            f"inverse_of_{geometry_meta.DUAL_EDGE_LENGTH}"
        ).ndarray,
        edge_cell_distance=geometry.get_full_precision(geometry_meta.EDGE_CELL_DISTANCE).ndarray,
        primal_edge_length=geometry.get_full_precision(geometry_meta.EDGE_LENGTH).ndarray,
        cell_area=geometry.get_full_precision(geometry_meta.CELL_AREA).ndarray,
        vn=vn,
        vct_b=vct_b.ndarray,
        nlev=num_levels,
    )

    prognostic_state_now.vn.ndarray[:] = vn.astype(ta.wpfloat)
    prognostic_state_now.w.ndarray[:] = w.astype(ta.wpfloat)
    prognostic_state_now.exner.ndarray[:] = exner.astype(ta.wpfloat)
    prognostic_state_now.theta_v.ndarray[:] = theta_v.astype(ta.wpfloat)
    prognostic_state_now.rho.ndarray[:] = rho.astype(ta.wpfloat)
    for name, field in tracer_state_now.active_fields():
        field.ndarray[:] = q[name].astype(ta.wpfloat)
    exchange.exchange(dims.EdgeDim, prognostic_state_now.vn)
    exchange.exchange(
        dims.CellDim,
        prognostic_state_now.w,
        prognostic_state_now.exner,
        prognostic_state_now.theta_v,
        prognostic_state_now.rho,
    )
