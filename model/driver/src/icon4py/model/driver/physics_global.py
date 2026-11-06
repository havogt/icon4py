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

Each of these is a method of its own, on `ColumnState` and `ColumnTendencies`; `PhysicsGlobal.run`
composes them.
"""

from __future__ import annotations

import dataclasses
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
from icon4py.model.common.states.surface_flux_state import SurfaceFluxState


_MOISTURE = ("qv", "qc", "qi", "qr", "qs", "qg")
_MUPHYS_DIAGNOSTICS = ("pflx", "pr", "ps", "pi", "pg", "pre")


@dataclasses.dataclass(frozen=True)
class ColumnState:
    """The cell-centred state the physics processes read, on full levels unless noted."""

    temperature: fa.CellKField[ta.wpfloat]
    virtual_temperature: fa.CellKField[ta.wpfloat]
    pressure: fa.CellKField[ta.wpfloat]
    pressure_ifc: fa.CellKHalfField[ta.wpfloat]
    rho: fa.CellKField[ta.wpfloat]
    u: fa.CellKField[ta.wpfloat]
    v: fa.CellKField[ta.wpfloat]
    w: fa.CellKHalfField[ta.wpfloat]
    q: dict[str, fa.CellKField[ta.wpfloat]]
    air_mass: fa.CellKField[ta.wpfloat]
    dz: fa.CellKField[ta.wpfloat]


@dataclasses.dataclass(frozen=True)
class ColumnTendencies:
    """Tendencies per second of the fields of `ColumnState`; None, or a missing tracer, is zero."""

    temperature: fa.CellKField[ta.wpfloat] | None = None
    q: dict[str, fa.CellKField[ta.wpfloat]] = dataclasses.field(default_factory=dict)
    u: fa.CellKField[ta.wpfloat] | None = None
    v: fa.CellKField[ta.wpfloat] | None = None
    w: fa.CellKHalfField[ta.wpfloat] | None = None


def _add(a: gtx.Field | None, b: gtx.Field | None) -> gtx.Field | None:
    if a is None:
        return b
    return a if b is None else a + b


def sum_column_tendencies(*tendencies: ColumnTendencies) -> ColumnTendencies:
    total = ColumnTendencies()
    for t in tendencies:
        total = ColumnTendencies(
            temperature=_add(total.temperature, t.temperature),
            q={
                name: summed
                for name in total.q.keys() | t.q.keys()
                if (summed := _add(total.q.get(name), t.q.get(name))) is not None
            },
            u=_add(total.u, t.u),
            v=_add(total.v, t.v),
            w=_add(total.w, t.w),
        )
    return total


def apply_column_tendencies(
    col: ColumnState,
    tend: ColumnTendencies,
    dtime: float,
    *,
    recompute_virtual_temperature: bool = False,
) -> ColumnState:
    """
    `col` advanced by `dtime` times `tend`, as one step of the sequential splitting.

    The pressure and the virtual temperature stay at their values from dyn2phy, as in
    `physics_driver.PhysicsState.update`.

    recompute_virtual_temperature: changes results, up to ~1e-2 relative in qv/qc after one step.
    """

    def advanced(field: gtx.Field, tendency: gtx.Field | None) -> gtx.Field:
        return field if tendency is None else field + dtime * tendency

    temperature = advanced(col.temperature, tend.temperature)
    q = {name: advanced(field, tend.q.get(name)) for name, field in col.q.items()}
    virtual_temperature = col.virtual_temperature
    if recompute_virtual_temperature:
        virtual_temperature = temperature * (
            1.0
            + PhysicsConstants.rv_o_rd_minus_1 * q["qv"]
            - (q["qc"] + q["qi"] + q["qr"] + q["qs"] + q["qg"])
        )
    return dataclasses.replace(
        col,
        temperature=temperature,
        virtual_temperature=virtual_temperature,
        u=advanced(col.u, tend.u),
        v=advanced(col.v, tend.v),
        w=advanced(col.w, tend.w),
        q=q,
    )


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
    pressure, pressure_ifc_on_model_levels = scan(  # type: ignore[misc]  # the scan of a tuple carry returns a tuple
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
        recompute_virtual_temperature: bool = False,
    ) -> None:
        if tmx is not None and surface is None:
            raise ValueError(
                "tmx needs the surface fluxes: configure 'surface' together with 'tmx'."
            )
        self.muphys = muphys
        self.surface = surface
        self.tmx = tmx
        self.recompute_virtual_temperature = recompute_virtual_temperature
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
        self._static_args: dict[str, gtx.Field] = {
            "ddqz_z_full": ddqz_z_full,
            "rbf_vec_coeff_c1": rbf_vec_coeff_c1,
            "rbf_vec_coeff_c2": rbf_vec_coeff_c2,
            "c_lin_e": c_lin_e,
            "primal_normal_cell_x": edge_params.primal_normal_cell[0],
            "primal_normal_cell_y": edge_params.primal_normal_cell[1],
        }

    @property
    def _cell_k(self) -> dict:
        return {dims.CellDim: (0, self._num_cells), dims.KDim: (0, self._num_levels)}

    def _virtual_temperature_and_temperature(
        self, prognostic: dict[str, gtx.Field], q: dict[str, gtx.Field]
    ) -> tuple[gtx.Field, gtx.Field]:
        return self._temperature(
            *(q[name] for name in _MOISTURE),
            prognostic["theta_v"],
            prognostic["exner"],
            domain=self._cell_k,
            offset_provider={},
        )

    def dyn2phy(
        self, prognostic: dict[str, gtx.Field], tracers: dict[str, gtx.Field]
    ) -> ColumnState:
        """The column state of the prognostics and tracers; the tracers pass through by name."""
        if missing := [name for name in _MOISTURE if name not in tracers]:
            raise ValueError(f"Physics requires all moisture species, missing: {missing}.")
        static = self._static_args
        nlev = self._num_levels
        cell_k = self._cell_k
        cell_khalf = {dims.CellDim: (0, self._num_cells), dims.KHalfDim: (0, nlev + 1)}
        virtual_temperature, temperature = self._virtual_temperature_and_temperature(
            prognostic, tracers
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
        return ColumnState(
            temperature=temperature,
            virtual_temperature=virtual_temperature,
            pressure=pressure,
            pressure_ifc=pressure_ifc,
            rho=prognostic["rho"],
            u=u,
            v=v,
            w=prognostic["w"],
            q=dict(tracers),
            # ICON's ``mair``
            air_mass=prognostic["rho"] * static["ddqz_z_full"],
            dz=static["ddqz_z_full"],
        )

    def run_muphys(
        self, col: ColumnState, dtime: float
    ) -> tuple[ColumnTendencies, dict[str, gtx.Field]]:
        """The tendencies of temperature and moisture, and the precipitation fluxes."""
        assert self.muphys is not None
        q = col.q
        new_temperature, new_q, *fluxes = self.muphys.run(
            dz=col.dz,
            te=col.temperature,
            p=col.pressure,
            rho=col.rho,
            q=Q(v=q["qv"], c=q["qc"], r=q["qr"], s=q["qs"], i=q["qi"], g=q["qg"]),
            dtime=dtime,
        )
        return (
            ColumnTendencies(
                temperature=(new_temperature - col.temperature) / dtime,
                q={
                    f"q{species}": (new - q[f"q{species}"]) / dtime
                    for species, new in new_q._asdict().items()
                },
            ),
            dict(zip(_MUPHYS_DIAGNOSTICS, fluxes, strict=True)),
        )

    def surface_input(self, col: ColumnState) -> surface_states.SurfaceInputState:
        return surface_states.SurfaceInputState(
            surface_pressure=col.pressure_ifc[dims.KHalfDim(self._num_levels)],  # type: ignore[arg-type]  # indexing a level drops the vertical dimension
            temperature=col.temperature[dims.KDim(self._num_levels - 1)],  # type: ignore[arg-type]  # indexing a level drops the vertical dimension
        )

    def run_surface(self, input_state: surface_states.SurfaceInputState) -> SurfaceFluxState:
        assert self.surface is not None
        return self.surface.run(input_state=input_state)

    def run_tmx(
        self, col: ColumnState, fluxes: SurfaceFluxState, dtime: float
    ) -> tuple[ColumnTendencies, dict[str, gtx.Field]]:
        """The tendencies of temperature, qv, qc, qi, u, v and w, and the tmx diagnostics."""
        assert self.tmx is not None
        _, tendencies, diagnostics = self.tmx.run(
            input_state=tmx_states.TmxInputState(
                temperature=col.temperature,
                virtual_temperature=col.virtual_temperature,
                pressure=col.pressure,
                u=col.u,
                v=col.v,
                w=col.w,
                **{name: col.q[name] for name in _MOISTURE},
                rho=col.rho,
                air_mass=col.air_mass,
            ),
            surface_flux_state=fluxes,
            dtime=dtime,
        )
        return (
            ColumnTendencies(
                temperature=tendencies.tend_temperature,
                q={name: getattr(tendencies, f"tend_{name}") for name in ("qv", "qc", "qi")},
                u=tendencies.tend_u,
                v=tendencies.tend_v,
                w=tendencies.tend_w,
            ),
            diagnostics,
        )

    def phy2dyn(
        self,
        prognostic: dict[str, gtx.Field],
        perturbed_exner: fa.CellKField[ta.wpfloat],
        col: ColumnState,
        tend: ColumnTendencies,
        dtime: float,
    ) -> tuple[dict[str, gtx.Field], dict[str, gtx.Field], fa.CellKField[ta.wpfloat]]:
        """
        The prognostics, tracers and perturbed exner of the advanced column `col`.

        At constant volume: rho stays, exner and theta_v follow from the virtual temperature of
        `col`'s temperature and moisture, w and the tracers are `col`'s, and vn takes the wind
        tendencies of `tend`, the sum over the processes.
        """
        nlev = self._num_levels
        q = col.q
        virtual_temperature, _ = self._virtual_temperature_and_temperature(prognostic, q)
        exner, theta_v, perturbed_exner = self._exner(
            prognostic["rho"],
            virtual_temperature,
            self._tv_tendency(
                dtime,
                *(q[name] for name in _MOISTURE),
                col.temperature,
                virtual_temperature,
                domain=self._cell_k,
                offset_provider={},
            ),
            prognostic["exner"],
            perturbed_exner,
            dtime,
            domain=self._cell_k,
            offset_provider={},
        )
        vn = prognostic["vn"]
        if tend.u is not None or tend.v is not None:
            assert tend.u is not None and tend.v is not None, "the wind tendencies come as a pair"
            vn = vn + dtime * self._vn_from_uv(
                tend.u,
                tend.v,
                self._static_args["primal_normal_cell_x"],
                self._static_args["primal_normal_cell_y"],
                self._static_args["c_lin_e"],
                domain={dims.EdgeDim: (0, self._num_edges), dims.KDim: (0, nlev)},
                offset_provider=self._offset_provider,
            )
        return (
            {**prognostic, "exner": exner, "theta_v": theta_v, "vn": vn, "w": col.w},
            dict(q),
            perturbed_exner,
        )

    def run(
        self,
        *,
        prognostic: dict[str, gtx.Field],
        tracers: dict[str, gtx.Field],
        perturbed_exner: fa.CellKField[ta.wpfloat],
        dtime: float,
    ) -> tuple[dict[str, gtx.Field], dict[str, gtx.Field], fa.CellKField[ta.wpfloat]]:
        """Return the new prognostics (exner, theta_v, vn, w), tracers and perturbed exner."""
        col = self.dyn2phy(prognostic, tracers)
        applied = []
        fluxes = None
        if self.muphys is not None:
            tend, _ = self.run_muphys(col, dtime)
            col = self._apply(col, tend, dtime)
            applied.append(tend)
        if self.surface is not None:
            fluxes = self.run_surface(self.surface_input(col))
        if self.tmx is not None:
            assert fluxes is not None
            tend, _ = self.run_tmx(col, fluxes, dtime)
            col = self._apply(col, tend, dtime)
            applied.append(tend)
        return self.phy2dyn(
            prognostic, perturbed_exner, col, sum_column_tendencies(*applied), dtime
        )

    def _apply(self, col: ColumnState, tend: ColumnTendencies, dtime: float) -> ColumnState:
        return apply_column_tendencies(
            col,
            tend,
            dtime,
            recompute_virtual_temperature=self.recompute_virtual_temperature,
        )
