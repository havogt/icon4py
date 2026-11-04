# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""
The physics of one time step for grids without lateral boundaries, without state between calls.

The same coupling as `physics_driver.PhysicsDriver` with its `PhysicsState`: the state is diagnosed
once (dyn2phy), muphys, the surface scheme and tmx run in turn, each process's tendencies are added
to the state before the next one reads it, and exner, theta_v, the perturbed exner and vn follow
from the final state (phy2dyn). Every process runs on every step. The step does no halo exchange.
"""

from __future__ import annotations

from typing import Any

import gt4py.next as gtx
from gt4py.next import exp, scan, sqrt
from gt4py.next.experimental import concat_where

import icon4py.model.common.grid.states as grid_states
from icon4py.model.atmosphere.subgrid_scale_physics.muphys import muphys_global
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.definitions import Q
from icon4py.model.atmosphere.subgrid_scale_physics.surface import surface_global, surface_states
from icon4py.model.atmosphere.subgrid_scale_physics.tmx import tmx_global, tmx_states
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.constants import PhysicsConstants
from icon4py.model.common.grid import icon as icon_grid
from icon4py.model.common.interpolation.stencils.compute_vn_from_uv import _compute_vn_from_uv
from icon4py.model.common.interpolation.stencils.edge_2_cell_vector_rbf_interpolation import (
    _edge_2_cell_vector_rbf_interpolation,
)
from icon4py.model.common.physics.thermodynamics.compute_pressure import _compute_surface_pressure
from icon4py.model.common.physics.thermodynamics.compute_temperature import (
    _compute_virtual_temperature_and_temperature,
    _update_exner_theta_v_and_perturbed_exner,
)
from icon4py.model.common.physics.thermodynamics.compute_tendencies import (
    _compute_virtual_temperature_tendency,
)


_MOISTURE = ("qv", "qc", "qi", "qr", "qs", "qg")


@gtx.field_operator
def _pressure_pass(
    state: tuple[ta.wpfloat, ta.wpfloat], ddqz_z_full: ta.wpfloat, virtual_temperature: ta.wpfloat
) -> tuple[ta.wpfloat, ta.wpfloat]:
    pressure_interface = state[1] * exp(
        -PhysicsConstants.grav_o_rd * ddqz_z_full / virtual_temperature
    )
    return sqrt(state[1] * pressure_interface), pressure_interface


@gtx.field_operator
def _surface_and_hydrostatic_pressure(
    exner: fa.CellKField[ta.wpfloat],
    virtual_temperature: fa.CellKField[ta.wpfloat],
    ddqz_z_full: fa.CellKField[ta.wpfloat],
    surface_pressure: fa.CellField[ta.wpfloat],
    num_levels: gtx.int32,
) -> tuple[fa.CellKField[ta.wpfloat], fa.CellKHalfField[ta.wpfloat]]:
    # `compute_pressure.compute_surface_and_hydrostatic_pressure`, with the surface pressure as
    # the scan's start instead of a flag in its carry: a scan that branches on a carried value
    # runs per column in embedded execution, which JAX cannot trace
    pressure, pressure_ifc_on_model_levels = scan(
        _pressure_pass,
        range=(dims.KDim, 0, num_levels),
        forward=False,
        init=(0.0, surface_pressure),
    )(ddqz_z_full, virtual_temperature)
    pressure_ifc = concat_where(
        dims.KHalfDim < num_levels,
        pressure_ifc_on_model_levels(dims.KHalfDim + 0.5),
        _compute_surface_pressure(exner, virtual_temperature, ddqz_z_full),
    )
    return pressure, pressure_ifc


class PhysicsGlobal:
    """muphys, the surface scheme and tmx, each optional, coupled as in ICON AES."""

    def __init__(
        self,
        *,
        grid: icon_grid.IconGrid,
        ddqz_z_full: fa.CellKField[ta.wpfloat],
        rbf_vec_coeff_c1: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2EDim], ta.wpfloat],
        rbf_vec_coeff_c2: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2EDim], ta.wpfloat],
        c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
        edge_params: grid_states.EdgeParams,
        muphys: muphys_global.MuphysGlobal | None = None,
        surface: surface_global.SurfaceGlobal | None = None,
        tmx: tmx_global.TmxGlobal | None = None,
    ) -> None:
        if tmx is not None and surface is None:
            raise ValueError(
                "tmx needs the surface fluxes: configure 'surface' together with 'tmx'."
            )
        self.muphys = muphys
        self.surface = surface
        self.tmx = tmx
        self._num_levels = int(grid.num_levels)
        self._num_cells = int(grid.num_cells)
        self._num_edges = int(grid.num_edges)
        self._offset_provider = grid.connectivities

        def unstructured(op: Any) -> Any:
            return op.with_grid_type(gtx.GridType.UNSTRUCTURED)

        self._temperature = unstructured(_compute_virtual_temperature_and_temperature)
        self._surface_pressure = unstructured(_compute_surface_pressure)
        self._pressure = unstructured(_surface_and_hydrostatic_pressure)
        self._rbf = unstructured(_edge_2_cell_vector_rbf_interpolation)
        self._tv_tendency = unstructured(_compute_virtual_temperature_tendency)
        self._exner = unstructured(_update_exner_theta_v_and_perturbed_exner)
        self._vn_from_uv = unstructured(_compute_vn_from_uv)
        self.ddqz_z_full = ddqz_z_full
        self._static_args: dict[str, gtx.Field] = {
            "ddqz_z_full": ddqz_z_full,
            "rbf_vec_coeff_c1": rbf_vec_coeff_c1,
            "rbf_vec_coeff_c2": rbf_vec_coeff_c2,
            "c_lin_e": c_lin_e,
            "primal_normal_cell_x": edge_params.primal_normal_cell[0],
            "primal_normal_cell_y": edge_params.primal_normal_cell[1],
        }

    def run(
        self,
        *,
        prognostic: dict[str, gtx.Field],
        tracers: dict[str, gtx.Field],
        perturbed_exner: fa.CellKField[ta.wpfloat],
        air_mass: fa.CellKField[ta.wpfloat],
        dtime: float,
    ) -> tuple[dict[str, gtx.Field], dict[str, gtx.Field], fa.CellKField[ta.wpfloat]]:
        """Return the new prognostics (exner, theta_v, vn, w), tracers and perturbed exner."""
        if missing := [name for name in _MOISTURE if name not in tracers]:
            raise ValueError(f"Physics requires all moisture species, missing: {missing}.")
        static = self._static_args
        nlev = self._num_levels
        cell_k = {dims.CellDim: (0, self._num_cells), dims.KDim: (0, nlev)}
        cell_khalf = {dims.CellDim: (0, self._num_cells), dims.KHalfDim: (0, nlev + 1)}
        rho, w = prognostic["rho"], prognostic["w"]
        q = dict(tracers)

        # dyn2phy
        virtual_temperature, temperature = self._temperature(
            *(q[name] for name in _MOISTURE),
            prognostic["theta_v"],
            prognostic["exner"],
            domain=cell_k,
            offset_provider={},
        )
        surface_pressure = self._surface_pressure(
            prognostic["exner"],
            virtual_temperature,
            static["ddqz_z_full"],
            domain={dims.CellDim: (0, self._num_cells), dims.KHalfDim: (nlev, nlev + 1)},
            offset_provider={},
        )[dims.KHalfDim(nlev)]
        pressure, pressure_ifc = self._pressure(
            prognostic["exner"],
            virtual_temperature,
            static["ddqz_z_full"],
            surface_pressure,
            nlev,
            domain=(cell_k, cell_khalf),
            offset_provider={},
        )
        u, v = self._rbf(
            prognostic["vn"],
            static["rbf_vec_coeff_c1"],
            static["rbf_vec_coeff_c2"],
            domain=cell_k,
            offset_provider=self._offset_provider,
        )

        # the processes, each reading the state the previous ones advanced
        wind_tendencies = None
        if self.muphys is not None:
            new_temperature, new_q, *_ = self.muphys.run(
                dz=static["ddqz_z_full"],
                te=temperature,
                p=pressure,
                rho=rho,
                q=Q(v=q["qv"], c=q["qc"], r=q["qr"], s=q["qs"], i=q["qi"], g=q["qg"]),
                dtime=dtime,
            )
            temperature = temperature + dtime * ((new_temperature - temperature) / dtime)
            for species, new in new_q._asdict().items():
                name = f"q{species}"
                q[name] = q[name] + dtime * ((new - q[name]) / dtime)
        if self.surface is not None:
            fluxes = self.surface.run(
                input_state=surface_states.SurfaceInputState(
                    surface_pressure=pressure_ifc[dims.KHalfDim(nlev)],
                    temperature=temperature[dims.KDim(nlev - 1)],
                )
            )
        if self.tmx is not None:
            _, tendencies, _ = self.tmx.run(
                input_state=tmx_states.TmxInputState(
                    temperature=temperature,
                    virtual_temperature=virtual_temperature,
                    pressure=pressure,
                    u=u,
                    v=v,
                    w=w,
                    **{name: q[name] for name in _MOISTURE},
                    rho=rho,
                    air_mass=air_mass,
                ),
                surface_flux_state=fluxes,
                dtime=dtime,
            )
            temperature = temperature + dtime * tendencies.tend_temperature
            w = w + dtime * tendencies.tend_w
            for name in ("qv", "qc", "qi"):
                q[name] = q[name] + dtime * getattr(tendencies, f"tend_{name}")
            wind_tendencies = (tendencies.tend_u, tendencies.tend_v)

        # phy2dyn
        exner, theta_v, perturbed_exner = self._exner(
            rho,
            virtual_temperature,
            self._tv_tendency(
                dtime,
                *(q[name] for name in _MOISTURE),
                temperature,
                virtual_temperature,
                domain=cell_k,
                offset_provider={},
            ),
            prognostic["exner"],
            perturbed_exner,
            dtime,
            domain=cell_k,
            offset_provider={},
        )
        vn = prognostic["vn"]
        if wind_tendencies is not None:
            vn = vn + dtime * self._vn_from_uv(
                *wind_tendencies,
                static["primal_normal_cell_x"],
                static["primal_normal_cell_y"],
                static["c_lin_e"],
                domain={dims.EdgeDim: (0, self._num_edges), dims.KDim: (0, nlev)},
                offset_provider=self._offset_provider,
            )
        return (
            {**prognostic, "exner": exner, "theta_v": theta_v, "vn": vn, "w": w},
            q,
            perturbed_exner,
        )
