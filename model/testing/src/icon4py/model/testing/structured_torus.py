# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""
A single field-operator call on an ICON torus in three layouts, for comparing memory order and structured indexing.

- "icon": ICON's entity numbering with its neighbour tables.
- "reordered": entities renumbered to the flat order of the (i, j, X) lattice, with the tables renumbered accordingly.
- "structured": fields on (I, J, X) with a periodic halo and a `StructuredConnectivity` offset provider.

The torus vertices sit on a sheared lattice: vertex (i, j) is at x = (i - j) a + j a / 2, y = j a sqrt(3) / 2. Cells and
edges are anchored at a vertex and coloured by their shape (X). Crossing the J boundary shifts I by the torus twist.
Tables that are not uniform per colour (V2E, V2C, ...) are brought into a canonical slot order per vertex, and every
field with that local dimension is permuted the same way, in both non-ICON layouts.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable, Mapping
from typing import Any

import gt4py.next as gtx
import numpy as np
from gt4py.next import common as gtx_common

from icon4py.model.common import dimension as dims


I = gtx.Dimension("I")  # noqa: E741 [ambiguous-variable-name]
J = gtx.Dimension("J")
X = gtx.Dimension("X")
LAYOUTS = ("icon", "reordered", "structured")
_N_COLOURS = {dims.CellDim: 2, dims.EdgeDim: 3, dims.VertexDim: 1}
_EDGE_COLOUR = {(1, 0): 0, (0, 1): 1, (1, 1): 2}
_UP = sorted([(0, 0), (0, 1), (1, 1)])
_DOWN = sorted([(0, 0), (1, 0), (1, 1)])


def _horizontal_dim(field_dims: tuple[gtx.Dimension, ...]) -> gtx.Dimension | None:
    return next((d for d in field_dims if d in _N_COLOURS), None)


@dataclasses.dataclass(frozen=True)
class TorusLayout:
    M: int
    N: int
    twist: int
    ijx: Mapping[gtx.Dimension, np.ndarray]

    @classmethod
    def from_grid_file(cls, grid_file: str) -> TorusLayout:
        import netCDF4  # noqa: PLC0415 [import-outside-top-level]

        with netCDF4.Dataset(grid_file) as nc:
            a = float(nc.getncattr("mean_edge_length"))
            height = float(nc.getncattr("domain_height"))
            length = float(nc.getncattr("domain_length"))
            vx = np.asarray(nc["cartesian_x_vertices"][:])
            vy = np.asarray(nc["cartesian_y_vertices"][:])
            e2v = np.asarray(nc["edge_vertices"][:]).T.astype(np.int64) - 1
            c2v = np.asarray(nc["vertex_of_cell"][:]).T.astype(np.int64) - 1
        dy = a * np.sqrt(3) / 2
        n, m = round(height / dy), round(length / a)
        jv = np.rint((vy - vy.min()) / dy).astype(np.int64)
        i_x = np.rint((vx - jv * a / 2 - vx[jv == 0].min()) / a).astype(np.int64) % m
        iv = (i_x + jv) % m
        vertex = np.stack([iv, jv], axis=1)

        def edge_deltas(twist: int) -> np.ndarray:
            return _minimal_image(vertex[e2v[:, 0]], vertex[e2v[:, 1]], m, n, twist)

        twist = min(
            (s for s in range(-m // 2, m // 2) if _all_lattice_steps(edge_deltas(s))), key=abs
        )
        d = edge_deltas(twist)
        forward = np.array([tuple(x) in _EDGE_COLOUR for x in d])
        anchor = np.where(forward, e2v[:, 0], e2v[:, 1])
        edge_colour = np.array(
            [_EDGE_COLOUR[tuple(x) if f else (-x[0], -x[1])] for x, f in zip(d, forward)]
        )
        edge = np.column_stack([vertex[anchor], edge_colour])

        cell = np.full((c2v.shape[0], 3), -1, dtype=np.int64)
        for k in range(3):
            rel = np.stack(
                [
                    _minimal_image(vertex[c2v[:, k]], vertex[c2v[:, v]], m, n, twist)
                    for v in range(3)
                ],
                axis=1,
            )
            rel_sorted = [sorted(map(tuple, r)) for r in rel]
            for colour, shape in ((0, _UP), (1, _DOWN)):
                hit = np.array([r == shape for r in rel_sorted]) & (cell[:, 0] < 0)
                cell[hit] = np.column_stack([vertex[c2v[hit, k]], np.full(hit.sum(), colour)])
        assert (cell[:, 0] >= 0).all(), "a cell is neither an up nor a down triangle"
        ijx = {
            dims.CellDim: cell,
            dims.EdgeDim: edge,
            dims.VertexDim: np.column_stack([vertex, np.zeros_like(iv)]),
        }
        layout = cls(m, n, twist, ijx)
        for dim, arr in ijx.items():
            assert np.unique(layout.flat_index(dim)).size == arr.shape[0], (
                f"{dim} is not a bijection"
            )
        return layout

    def flat_index(self, dim: gtx.Dimension) -> np.ndarray:
        """Per ICON entity, its index in the flattened (I, J, X) lattice."""
        i, j, x = self.ijx[dim].T
        return (i * self.N + j) * _N_COLOURS[dim] + x

    def to_structured(self, dim: gtx.Dimension, flat: np.ndarray, halo: int) -> np.ndarray:
        i, j, x = self.ijx[dim].T
        core = np.empty((self.M, self.N, _N_COLOURS[dim], *flat.shape[1:]), dtype=flat.dtype)
        core[i, j, x] = flat
        ii = np.arange(-halo, self.M + halo)
        jj = np.arange(-halo, self.N + halo)
        src_i = (ii[:, None] - self.twist * (jj // self.N)[None, :]) % self.M
        return core[src_i, (jj % self.N)[None, :]]

    def core_to_icon(self, dim: gtx.Dimension, core: np.ndarray) -> np.ndarray:
        i, j, x = self.ijx[dim].T
        return core[i, j, x]

    def offsets(self, src: gtx.Dimension, tgt: gtx.Dimension, table: np.ndarray) -> np.ndarray:
        """(n_src, n_slots, 3) (di, dj, dX) from each source entity to each neighbour."""
        a = np.repeat(self.ijx[src][:, None, :], table.shape[1], axis=1).reshape(-1, 3)
        b = self.ijx[tgt][table.reshape(-1)]
        dij = _minimal_image(a[:, :2], b[:, :2], self.M, self.N, self.twist)
        return np.column_stack([dij, b[:, 2] - a[:, 2]]).reshape(*table.shape, 3)


def _minimal_image(a: np.ndarray, b: np.ndarray, m: int, n: int, twist: int) -> np.ndarray:
    """(di, dj) from a to b on the torus (i, j) ~ (i + m, j) ~ (i + twist, j + n), smallest |di| + |dj|."""
    cands = []
    for w in (-1, 0, 1):
        dj = b[:, 1] - a[:, 1] + w * n
        di = (b[:, 0] - a[:, 0] + w * twist + m // 2) % m - m // 2
        cands.append(np.stack([di, dj], axis=1))
    stacked = np.stack(cands)
    best = np.argmin(np.abs(stacked).sum(axis=2), axis=0)
    return stacked[best, np.arange(a.shape[0])]


def _all_lattice_steps(d: np.ndarray) -> bool:
    return bool(np.isin(d[:, 0] * 10 + d[:, 1], [10, -10, 1, -1, 11, -11]).all())


@dataclasses.dataclass(frozen=True)
class _Tables:
    """Per connectivity: per-colour offsets, and for non-uniform tables the ICON slot for each canonical slot."""

    offsets: dict[str, dict[int, list[tuple[int, int, int]]]]
    slot_perm: dict[gtx.Dimension, np.ndarray]


def _analyse_tables(layout: TorusLayout, connectivities: Mapping[str, Any]) -> _Tables:
    offsets: dict[str, dict[int, list[tuple[int, int, int]]]] = {}
    slot_perm: dict[gtx.Dimension, np.ndarray] = {}
    for name, conn in connectivities.items():
        if not isinstance(conn, gtx_common.Connectivity):
            continue
        src, local = conn.domain.dims
        tgt = conn.codomain
        if src not in _N_COLOURS or tgt not in _N_COLOURS:
            continue
        table = conn.asnumpy()
        assert (table >= 0).all(), f"{name} has skip values; the torus has none"
        offs = layout.offsets(src, tgt, table)
        colour = layout.ijx[src][:, 2]
        code = ((offs[..., 0] + 64) * 128 + (offs[..., 1] + 64)) * 8 + (offs[..., 2] + 4)
        per: dict[int, list[tuple[int, int, int]]] = {}
        uniform = True
        for c in np.unique(colour):
            rows = code[colour == c]
            if not (rows == rows[0]).all():
                uniform = False
        if uniform:
            for c in np.unique(colour):
                per[int(c)] = [tuple(map(int, o)) for o in offs[np.argmax(colour == c)]]
        else:
            perm = np.argsort(code, axis=1, kind="stable")
            sorted_code = np.take_along_axis(code, perm, axis=1)
            for c in np.unique(colour):
                rows = sorted_code[colour == c]
                assert (rows == rows[0]).all(), (
                    f"{name}: the neighbour sets differ within colour {c}"
                )
                e = int(np.argmax(colour == c))
                per[int(c)] = [tuple(map(int, offs[e, k])) for k in perm[e]]
            assert local not in slot_perm, local
            slot_perm[local] = perm
        offsets[name] = per
    return _Tables(offsets, slot_perm)


@dataclasses.dataclass(frozen=True)
class LayoutCall:
    """A jitted call of the operator in one layout, its prognostic-free arguments bound, and the way back to ICON order."""

    run: Callable[[], tuple[Any, ...]]
    to_icon: Callable[[tuple[Any, ...]], list[np.ndarray]]
    extent: dict[str, tuple[int, ...]]


def make_layout_call(  # noqa: PLR0915 [too-many-statements]
    op: Any,
    kwargs: Mapping[str, Any],
    *,
    domain: tuple[Mapping[gtx.Dimension, tuple[int, int]], ...],
    connectivities: Mapping[str, Any],
    layout: TorusLayout,
    which: str,
    halo: int,
) -> LayoutCall:
    import jax  # noqa: PLC0415 [import-outside-top-level]

    jax.config.update("jax_enable_x64", True)
    jnp = jax.numpy
    tables = _analyse_tables(layout, connectivities)
    sizes = {dim: arr.shape[0] for dim, arr in layout.ijx.items()}
    for d in domain:
        hdim = _horizontal_dim(tuple(d))
        assert d[hdim] == (0, sizes[hdim]), f"{hdim} output domain {d[hdim]} is not the whole torus"
    out_dims = [_horizontal_dim(tuple(d)) for d in domain]

    def as_jax(f: Any) -> Any:
        if isinstance(f, gtx_common.Connectivity):
            return gtx.as_connectivity(
                f.domain,
                f.codomain,
                jnp.asarray(f.asnumpy()),
                skip_value=f.skip_value,
                allocator=jnp,
            )
        if isinstance(f, gtx.Field):
            return gtx.as_field(f.domain, jnp.asarray(f.asnumpy()), allocator=jnp)
        return f

    def slot_permuted(f: gtx.Field) -> np.ndarray:
        arr = f.asnumpy()
        for ax, d in enumerate(f.domain.dims):
            if d in tables.slot_perm:
                perm = tables.slot_perm[d]
                arr = np.take_along_axis(
                    arr, perm.reshape(perm.shape + (1,) * (arr.ndim - 2)), axis=ax
                )
        return arr

    if which == "icon":
        args = {k: as_jax(v) for k, v in kwargs.items()}
        provider = {k: as_jax(v) for k, v in connectivities.items()}
        out_domain = domain

        def to_icon(outs: tuple[Any, ...]) -> list[np.ndarray]:
            return [np.asarray(o.asnumpy()) for o in outs]

        extent = {dim.value: (n,) for dim, n in sizes.items()}

    elif which == "reordered":
        new = {dim: layout.flat_index(dim) for dim in sizes}
        old_of_new = {dim: np.argsort(idx) for dim, idx in new.items()}

        def reorder(f: Any) -> Any:
            if not isinstance(f, gtx.Field):
                return f
            hdim = _horizontal_dim(f.domain.dims)
            if hdim is None:
                return as_jax(f)
            assert f.domain.dims[0] is hdim, f.domain
            arr = slot_permuted(f)[old_of_new[hdim]]
            return gtx.as_field(f.domain, jnp.asarray(arr), allocator=jnp)

        args = {k: reorder(v) for k, v in kwargs.items()}
        provider = {}
        for name, conn in connectivities.items():
            if name not in tables.offsets:
                provider[name] = as_jax(conn)
                continue
            src, local = conn.domain.dims
            table = conn.asnumpy()
            if local in tables.slot_perm:
                table = np.take_along_axis(table, tables.slot_perm[local], axis=1)
            renumbered = new[conn.codomain][table][old_of_new[src]].astype(table.dtype)
            provider[name] = gtx.as_connectivity(
                conn.domain,
                conn.codomain,
                jnp.asarray(renumbered),
                skip_value=conn.skip_value,
                allocator=jnp,
            )
        out_domain = domain

        def to_icon(outs: tuple[Any, ...]) -> list[np.ndarray]:
            return [np.asarray(o.asnumpy())[new[d]] for o, d in zip(outs, out_dims)]

        extent = {dim.value: (n,) for dim, n in sizes.items()}

    elif which == "structured":
        from gt4py.next.embedded.structured_connectivity import (  # noqa: PLC0415 [import-outside-top-level]
            StructuredConnectivity,
        )

        def structure(f: Any) -> Any:
            if not isinstance(f, gtx.Field):
                return f
            hdim = _horizontal_dim(f.domain.dims)
            if hdim is None:
                return as_jax(f)
            s = layout.to_structured(hdim, slot_permuted(f), halo)
            rest = {
                d: (r.start, r.stop)
                for d, r in zip(f.domain.dims, f.domain.ranges)
                if d is not hdim
            }
            dom = gtx.domain(
                {
                    I: (-halo, layout.M + halo),
                    J: (-halo, layout.N + halo),
                    X: (0, _N_COLOURS[hdim]),
                    **rest,
                }
            )
            return gtx.as_field(dom, jnp.asarray(np.ascontiguousarray(s)), allocator=jnp)

        args = {k: structure(v) for k, v in kwargs.items()}
        provider = {k: as_jax(v) for k, v in connectivities.items() if k not in tables.offsets}
        for name, per in tables.offsets.items():
            conn = connectivities[name]
            provider[name] = StructuredConnectivity(
                source_dim=conn.domain.dims[0],
                codomain=conn.codomain,
                color_dim=X,
                local_dim=conn.domain.dims[1],
                offsets={
                    c: [{d: v for d, v in zip((I, J, X), o) if v != 0} for o in slots]
                    for c, slots in per.items()
                },
            )
        out_domain = tuple(
            {
                I: (0, layout.M),
                J: (0, layout.N),
                X: (0, _N_COLOURS[hdim]),
                **{k: v for k, v in d.items() if k is not hdim},
            }
            for d, hdim in zip(domain, out_dims)
        )

        def to_icon(outs: tuple[Any, ...]) -> list[np.ndarray]:
            return [layout.core_to_icon(d, np.asarray(o.asnumpy())) for o, d in zip(outs, out_dims)]

        extent = {
            dim.value: (layout.M + 2 * halo, layout.N + 2 * halo, n)
            for dim, n in _N_COLOURS.items()
        }
    else:
        raise ValueError(which)

    jitted = jax.jit(lambda a: op(**a, domain=out_domain, offset_provider=provider))

    def run() -> tuple[Any, ...]:
        return jax.block_until_ready(jitted(args))

    return LayoutCall(run, to_icon, extent)


class Recorded(Exception):
    """Raised by `record_call` to stop the caller after the fused call's arguments are captured."""


def record_call(owner: Any, attribute: str, call: Callable[[], Any]) -> dict[str, Any]:
    """Capture the keyword arguments of `owner.<attribute>(...)` during `call()`, without running it."""
    captured: dict[str, Any] = {}
    original = getattr(owner, attribute)

    def recorder(**kwargs: Any) -> Any:
        captured.update(kwargs)
        raise Recorded

    setattr(owner, attribute, recorder)
    try:
        call()
    except Recorded:
        pass
    finally:
        setattr(owner, attribute, original)
    assert captured, f"{attribute} was not called"
    return captured


@functools.cache
def torus_layout(grid_file: str) -> TorusLayout:
    return TorusLayout.from_grid_file(grid_file)


def is_torus(grid_file: str) -> bool:
    import netCDF4  # noqa: PLC0415 [import-outside-top-level]

    with netCDF4.Dataset(grid_file) as nc:
        return "domain_length" in nc.ncattrs() and "domain_height" in nc.ncattrs()


def check_layouts(
    calls: Mapping[str, LayoutCall], names: tuple[str, ...], rtol: float = 1e-12
) -> dict[str, dict[str, tuple[float, float]]]:
    """
    One call per layout, outputs mapped back to ICON order, against the "icon" layout.

    Prints and returns (max abs diff, max abs diff / max |icon|) per layout and output; asserts the
    relative one is below `rtol`. The structured layout is compared on its core only: its halo
    points are not outputs.
    """
    outs = {which: call.to_icon(call.run()) for which, call in calls.items()}
    reference = outs["icon"]
    result: dict[str, dict[str, tuple[float, float]]] = {}
    for which, got in outs.items():
        if which == "icon":
            continue
        result[which] = {}
        for name, g, r in zip(names, got, reference):
            assert g.shape == r.shape, (which, name, g.shape, r.shape)
            diff = float(np.max(np.abs(g - r)))
            rel = diff / float(np.max(np.abs(r)))
            result[which][name] = (diff, rel)
            print(f"LAYOUT_CHECK {which} vs icon {name}: max abs {diff:.3e}, rel {rel:.3e}")
    for which, per in result.items():
        for name, (_, rel) in per.items():
            assert rel <= rtol, f"{which} {name}: relative difference {rel:.3e} > {rtol}"
    return result
