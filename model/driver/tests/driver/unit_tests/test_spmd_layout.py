# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import concurrent.futures
import dataclasses
import functools
import threading
from typing import Any

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.common import dimension as dims
from icon4py.model.common.decomposition import decomposer as decomp, definitions as decomp_defs
from icon4py.model.common.grid import grid_manager as gm, icon, vertical as v_grid
from icon4py.model.driver import spmd_layout
from icon4py.model.testing import datatest_utils as dt_utils, definitions as test_defs


@dataclasses.dataclass(frozen=True)
class DummyProps(decomp_defs.ProcessProperties):
    comm: object
    comm_name: str
    comm_size: int
    rank: int


def _grid_manager(props: decomp_defs.ProcessProperties, decomposer: Any) -> gm.GridManager:
    manager = gm.GridManager(
        config=v_grid.VerticalGridConfig(num_levels=2),
        grid_file=dt_utils.get_grid_filepath(test_defs.Grids.R02B04_GLOBAL),
    )
    manager(
        decomposer=decomposer,
        allocator=None,
        keep_skip_values=True,
        process_props=props,
        extra_halo_rings=2,
    )
    return manager


def _build_layouts(infos: list[decomp_defs.DecompositionInfo]) -> list[spmd_layout.PaddedLayout]:
    """Runs `build_padded_layout` on one thread per rank, with a barrier-based allgather."""
    num_ranks = len(infos)
    slots: list[Any] = [None] * num_ranks
    barrier = threading.Barrier(num_ranks)

    def allgather(rank: int, obj: Any) -> list[Any]:
        slots[rank] = obj
        barrier.wait()
        gathered = list(slots)
        barrier.wait()
        return gathered

    with concurrent.futures.ThreadPoolExecutor(num_ranks) as pool:
        futures = [
            pool.submit(
                spmd_layout.build_padded_layout,
                info,
                rank,
                num_ranks,
                functools.partial(allgather, rank),
            )
            for rank, info in enumerate(infos)
        ]
        return [f.result() for f in futures]


@pytest.fixture(scope="module")
def global_grid() -> icon.IconGrid:
    return _grid_manager(
        decomp_defs.SingleNodeProcessProperties(), decomp.SingleNodeDecomposer()
    ).grid


@pytest.fixture(scope="module", params=[2, 4], ids=lambda n: f"ranks{n}")
def ranks(
    request: pytest.FixtureRequest,
) -> tuple[list[gm.GridManager], list[spmd_layout.PaddedLayout]]:
    num_ranks = request.param
    managers = [
        _grid_manager(DummyProps(None, "dummy", num_ranks, r), decomp.MetisDecomposer())
        for r in range(num_ranks)
    ]
    return managers, _build_layouts([m.decomposition_info for m in managers])


def _global_index(manager: gm.GridManager, dim: gtx.Dimension) -> np.ndarray:
    return np.asarray(manager.decomposition_info.global_index(dim))


def _to_global(global_index: np.ndarray, table: np.ndarray) -> np.ndarray:
    return np.where(table < 0, -1, global_index[table])


def _halo_mismatches(
    managers: list[gm.GridManager],
    layouts: list[spmd_layout.PaddedLayout],
    dim: gtx.Dimension,
    num_global: int,
) -> int:
    """Number of real rows that differ from a global field after exchanging only its owned values."""
    g = np.random.default_rng(42).random(num_global)
    local = []
    for manager, layout in zip(managers, layouts, strict=True):
        values = g[_global_index(manager, dim)]
        values[layout.num_owned[dim] :] = np.nan
        local.append(values)
    padded = [spmd_layout.pad_rows(lt, dim, a) for lt, a in zip(layouts, local, strict=True)]
    exchanged = spmd_layout.exchange_numpy(layouts, padded, dim)

    mismatches = 0
    for manager, layout, before, after in zip(managers, layouts, padded, exchanged, strict=True):
        real = after[layout.padded_index[dim]]
        mismatches += int(np.count_nonzero(real != g[_global_index(manager, dim)]))
        is_padding = np.ones(layout.padded_local[dim], dtype=bool)
        is_padding[layout.padded_index[dim]] = False
        mismatches += int(np.count_nonzero(after[is_padding] != before[is_padding]))
        np.testing.assert_array_equal(
            spmd_layout.owned_rows(layout, dim, after),
            g[_global_index(manager, dim)][: layout.num_owned[dim]],
        )
    return mismatches


@pytest.mark.datatest
@pytest.mark.parametrize("dim", list(dims.horizontal_dims()), ids=lambda d: d.value)
def test_layout_invariants(ranks: Any, dim: gtx.Dimension) -> None:
    managers, layouts = ranks
    assert len({lt.padded_owned[dim] for lt in layouts}) == 1
    assert len({lt.padded_local[dim] for lt in layouts}) == 1
    assert len({lt.send_index[dim].shape for lt in layouts}) == 1
    assert len({lt.recv_index[dim].shape for lt in layouts}) == 1
    assert layouts[0].padded_owned[dim] == max(lt.num_owned[dim] for lt in layouts)
    assert layouts[0].padded_local[dim] - layouts[0].padded_owned[dim] == max(
        lt.num_local[dim] - lt.num_owned[dim] for lt in layouts
    )
    for manager, layout in zip(managers, layouts, strict=True):
        num_owned, num_local = layout.num_owned[dim], layout.num_local[dim]
        assert num_owned == int(np.asarray(manager.decomposition_info.owner_mask(dim)).sum())
        assert num_local == _global_index(manager, dim).size
        padded_index = layout.padded_index[dim]
        assert len(set(padded_index.tolist())) == num_local
        np.testing.assert_array_equal(padded_index[:num_owned], np.arange(num_owned))
        assert (padded_index[num_owned:] >= layout.padded_owned[dim]).all()
        assert padded_index.max() < layout.padded_local[dim]
        source_index = layout.source_index[dim]
        np.testing.assert_array_equal(source_index[padded_index], np.arange(num_local))
        is_padding = np.ones(layout.padded_local[dim], dtype=bool)
        is_padding[padded_index] = False
        assert (source_index[is_padding] == 0).all()
        assert (layout.send_index[dim] < layout.padded_owned[dim]).all()


@pytest.mark.datatest
@pytest.mark.parametrize("dim", list(dims.horizontal_dims()), ids=lambda d: d.value)
def test_exchange_matches_global_field(
    ranks: Any, global_grid: icon.IconGrid, dim: gtx.Dimension
) -> None:
    managers, layouts = ranks
    assert _halo_mismatches(managers, layouts, dim, global_grid.size[dim]) == 0


@pytest.mark.datatest
@pytest.mark.parametrize("dim", list(dims.horizontal_dims()), ids=lambda d: d.value)
def test_exchange_detects_corrupted_recv_index(
    ranks: Any, global_grid: icon.IconGrid, dim: gtx.Dimension
) -> None:
    managers, layouts = ranks
    target = layouts[1]
    recv = target.recv_index[dim].copy()
    valid = np.flatnonzero(recv[0] < target.padded_local[dim])
    assert valid.size >= 2
    recv[0, valid[:2]] = recv[0, valid[1::-1]]
    corrupted = dataclasses.replace(target, recv_index={**target.recv_index, dim: recv})
    assert (
        _halo_mismatches(
            managers, [layouts[0], corrupted, *layouts[2:]], dim, global_grid.size[dim]
        )
        > 0
    )


def _horizontal_neighbour_tables(
    grid: icon.IconGrid,
) -> dict[str, tuple[gtx.Dimension, gtx.Dimension, np.ndarray]]:
    tables = {}
    for name, connectivity in grid.connectivities.items():
        from_dim, to_dim = connectivity.domain.dims[0], connectivity.codomain
        if (
            from_dim.kind == gtx.DimensionKind.HORIZONTAL
            and to_dim.kind == gtx.DimensionKind.HORIZONTAL
        ):
            tables[name] = (from_dim, to_dim, np.asarray(connectivity.ndarray))
    return tables


@pytest.mark.datatest
def test_pad_connectivity_matches_global_grid(ranks: Any, global_grid: icon.IconGrid) -> None:
    managers, layouts = ranks
    global_tables = _horizontal_neighbour_tables(global_grid)
    assert {"C2E", "E2C", "V2E", "C2E2C", "E2V", "C2V"} <= global_tables.keys()
    for manager, layout in zip(managers, layouts, strict=True):
        local_tables = _horizontal_neighbour_tables(manager.grid)
        assert local_tables.keys() == global_tables.keys()
        for name, (from_dim, to_dim, table) in local_tables.items():
            num_owned = layout.num_owned[from_dim]
            from_global = _global_index(manager, from_dim)
            to_global = _global_index(manager, to_dim)
            expected = global_tables[name][2][from_global[:num_owned]]

            np.testing.assert_array_equal(
                _to_global(to_global, table[:num_owned]), expected, err_msg=name
            )

            padded = spmd_layout.pad_connectivity(layout, from_dim, to_dim, table)
            assert padded.shape == (layout.padded_local[from_dim], table.shape[1])
            owned = padded[layout.padded_index[from_dim][:num_owned]]
            np.testing.assert_array_equal(
                _to_global(to_global, _to_global(layout.source_index[to_dim], owned)),
                expected,
                err_msg=name,
            )
            np.testing.assert_array_equal(
                _to_global(layout.source_index[to_dim], padded[layout.padded_index[from_dim]]),
                table,
                err_msg=name,
            )
            np.testing.assert_array_equal(
                padded[layout.source_index[from_dim] == 0],
                np.broadcast_to(padded[0], padded[layout.source_index[from_dim] == 0].shape),
                err_msg=name,
            )
