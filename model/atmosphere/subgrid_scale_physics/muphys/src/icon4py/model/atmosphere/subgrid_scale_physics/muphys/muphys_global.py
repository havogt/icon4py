# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""
Muphys microphysics for grids without lateral boundaries.

The same step as `component.MuphysComponent`, as one field operator call that returns the new
state instead of tendencies. The surface precipitation rates and the energy flux are the last
level of `pr`, `ps`, `pi`, `pg` and `pre`.
"""

from __future__ import annotations

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.definitions import Q
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.implementations.muphys_global_step import (
    _muphys_global_step,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.grid import horizontal as h_grid, icon as icon_grid


class MuphysGlobal:
    """One muphys step (saturation adjustment, AES graupel, saturation adjustment)."""

    def __init__(
        self,
        *,
        grid: icon_grid.IconGrid,
        qnc: float,
        backend: gtx_typing.Backend | None = None,
    ) -> None:
        if grid.limited_area:
            raise ValueError("'MuphysGlobal' does not support limited area grids.")
        self._step = _muphys_global_step.with_grid_type(gtx.GridType.UNSTRUCTURED).with_backend(
            backend
        )
        self._qnc = float(qnc)
        self._num_levels = int(grid.num_levels)
        cell_domain = h_grid.domain(dims.CellDim)
        # every local cell, the halo included: on a distributed grid the next process and the
        # dynamics read the outputs there, and the outermost rows that read past the local grid
        # are never read back
        self._cells = (
            int(grid.start_index(cell_domain(h_grid.Zone.NUDGING))),
            int(grid.end_index(cell_domain(h_grid.Zone.END))),
        )

    def run(
        self,
        *,
        dz: fa.CellKField[ta.wpfloat],
        te: fa.CellKField[ta.wpfloat],
        p: fa.CellKField[ta.wpfloat],
        rho: fa.CellKField[ta.wpfloat],
        q: Q,
        dtime: float,
    ) -> tuple[
        fa.CellKField[ta.wpfloat],
        Q,
        fa.CellKField[ta.wpfloat],
        fa.CellKField[ta.wpfloat],
        fa.CellKField[ta.wpfloat],
        fa.CellKField[ta.wpfloat],
        fa.CellKField[ta.wpfloat],
        fa.CellKField[ta.wpfloat],
    ]:
        """Return the new temperature and tracers, then `pflx`, `pr`, `ps`, `pi`, `pg` and `pre`."""
        cell_k = {dims.CellDim: self._cells, dims.KDim: (0, self._num_levels)}
        te_new, q_new, *fluxes = self._step(
            num_levels=self._num_levels,
            dz=dz,
            te=te,
            p=p,
            rho=rho,
            q_in=q,
            dt=dtime,
            qnc=self._qnc,
            domain=(cell_k, (cell_k,) * 6, cell_k, cell_k, cell_k, cell_k, cell_k, cell_k),
            offset_provider={},
        )
        return (te_new, Q(*q_new), *fluxes)
