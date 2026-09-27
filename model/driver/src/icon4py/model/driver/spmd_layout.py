# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Padded per-rank layout and halo-exchange tables for running the distributed driver as one SPMD
program.

In SPMD every rank runs the same program, so the owned and local sizes of each horizontal
dimension must be the same on all ranks. Each rank's local rows are therefore re-laid out per
dimension as

    [owned | owned padding up to padded_owned | halo | halo padding up to padded_local]

where the padding rows replicate local row 0, so that they compute finite values.

The halo exchange is an all-to-all of fixed-size buffers: rank r sends
``padded[send_index[p]]`` to every rank p, and writes what it receives from p into
``padded[recv_index[p]]``, dropping the slots that are out of bounds.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import gt4py.next as gtx
import numpy as np

from icon4py.model.common import dimension as dims
from icon4py.model.common.decomposition import definitions as decomposition_defs
from icon4py.model.common.utils import data_allocation as data_alloc


@dataclasses.dataclass(frozen=True)
class PaddedLayout:
    rank: int
    num_ranks: int
    #: this rank's real owned and local counts
    num_owned: dict[gtx.Dimension, int]
    num_local: dict[gtx.Dimension, int]
    #: the padded owned and local sizes, identical on all ranks
    padded_owned: dict[gtx.Dimension, int]
    padded_local: dict[gtx.Dimension, int]
    #: (padded_local,): the local row of each padded row, 0 for the padding rows
    source_index: dict[gtx.Dimension, np.ndarray]
    #: (num_local,): the padded row of each local row
    padded_index: dict[gtx.Dimension, np.ndarray]
    #: (num_ranks, max sent points): the padded owned rows sent to each rank, unused slots 0
    send_index: dict[gtx.Dimension, np.ndarray]
    #: (num_ranks, max sent points): the padded halo rows receiving from each rank, in the order
    #: that rank sends them; unused slots are padded_local, out of bounds
    recv_index: dict[gtx.Dimension, np.ndarray]


def _padded_index(num_owned: int, num_local: int, padded_owned: int) -> np.ndarray:
    index = np.arange(num_local)
    return np.where(index < num_owned, index, index + padded_owned - num_owned)


def build_padded_layout(
    decomposition_info: decomposition_defs.DecompositionInfo,
    rank: int,
    num_ranks: int,
    allgather: Callable[[Any], list[Any]],
) -> PaddedLayout:
    """
    The padded layout of `rank`, collective over all ranks.

    `allgather(obj)` returns the list of every rank's `obj`, ordered by rank.
    """
    local = {}
    for dim in dims.horizontal_dims():
        owner_mask = data_alloc.as_numpy(decomposition_info.owner_mask(dim))
        owned = int(owner_mask.sum())
        assert owner_mask[:owned].all(), f"the owned {dim.value}s are not first"
        local[dim] = (owned, data_alloc.as_numpy(decomposition_info.global_index(dim)))
    gathered = allgather(local)
    assert len(gathered) == num_ranks

    num_owned: dict[gtx.Dimension, int] = {}
    num_local: dict[gtx.Dimension, int] = {}
    padded_owned: dict[gtx.Dimension, int] = {}
    padded_local: dict[gtx.Dimension, int] = {}
    source_index: dict[gtx.Dimension, np.ndarray] = {}
    padded_index: dict[gtx.Dimension, np.ndarray] = {}
    send_index: dict[gtx.Dimension, np.ndarray] = {}
    recv_index: dict[gtx.Dimension, np.ndarray] = {}
    for dim in local:
        owned_counts = [g[dim][0] for g in gathered]
        global_indices = [g[dim][1] for g in gathered]
        local_counts = [gi.size for gi in global_indices]
        num_owned[dim] = owned_counts[rank]
        num_local[dim] = local_counts[rank]
        padded_owned[dim] = max(owned_counts)
        padded_local[dim] = padded_owned[dim] + max(
            n - o for n, o in zip(local_counts, owned_counts, strict=True)
        )

        num_global = 1 + max(int(gi.max()) for gi in global_indices)
        owner = np.full(num_global, -1)
        owner_row = np.full(num_global, -1)
        for q, (gi, o) in enumerate(zip(global_indices, owned_counts, strict=True)):
            assert (owner[gi[:o]] == -1).all(), f"{dim.value}s owned by more than one rank"
            owner[gi[:o]] = q
            owner_row[gi[:o]] = np.arange(o)

        # halo_owner[q]: the owner of each halo point of rank q, in q's local order
        halo_owner = []
        for gi, o in zip(global_indices, owned_counts, strict=True):
            halo_owner.append(owner[gi[o:]])
            assert (halo_owner[-1] >= 0).all(), f"halo {dim.value}s owned by no rank"
        max_sent = max(
            (int(np.bincount(h, minlength=num_ranks).max()) for h in halo_owner if h.size),
            default=0,
        )

        padded_index[dim] = _padded_index(num_owned[dim], num_local[dim], padded_owned[dim])
        source_index[dim] = np.zeros(padded_local[dim], dtype=padded_index[dim].dtype)
        source_index[dim][padded_index[dim]] = np.arange(num_local[dim])

        send = np.zeros((num_ranks, max_sent), dtype=padded_index[dim].dtype)
        recv = np.full((num_ranks, max_sent), padded_local[dim], dtype=padded_index[dim].dtype)
        my_halo_rows = padded_index[dim][num_owned[dim] :]
        for p in range(num_ranks):
            received = my_halo_rows[halo_owner[rank] == p]
            recv[p, : received.size] = received
            # owned rows are not shifted by the padding, so the owner's local row is its padded row
            sent = owner_row[global_indices[p][owned_counts[p] :][halo_owner[p] == rank]]
            send[p, : sent.size] = sent
        send_index[dim] = send
        recv_index[dim] = recv

    return PaddedLayout(
        rank=rank,
        num_ranks=num_ranks,
        num_owned=num_owned,
        num_local=num_local,
        padded_owned=padded_owned,
        padded_local=padded_local,
        source_index=source_index,
        padded_index=padded_index,
        send_index=send_index,
        recv_index=recv_index,
    )


def pad_rows(layout: PaddedLayout, dim: gtx.Dimension, array: Any) -> Any:
    """`array` with axis 0 re-laid out from the local to the padded rows of `dim`."""
    return array[layout.source_index[dim]]


def pad_connectivity(
    layout: PaddedLayout, from_dim: gtx.Dimension, to_dim: gtx.Dimension, table: np.ndarray
) -> np.ndarray:
    """A local `from_dim` -> `to_dim` neighbour table in the padded rows and indices; -1 is kept."""
    rows = np.asarray(table)[layout.source_index[from_dim]]
    return np.where(rows < 0, -1, layout.padded_index[to_dim][rows])


def owned_rows(layout: PaddedLayout, dim: gtx.Dimension, padded_array: Any) -> Any:
    return padded_array[: layout.num_owned[dim]]


def exchange_numpy(
    layouts: list[PaddedLayout], padded_arrays: list[np.ndarray], dim: gtx.Dimension
) -> list[np.ndarray]:
    """Reference all-to-all halo exchange of one padded array per rank."""
    result = [a.copy() for a in padded_arrays]
    for r, layout in enumerate(layouts):
        for p, target in enumerate(layouts):
            buffer = padded_arrays[r][layout.send_index[dim][p]]
            slots = target.recv_index[dim][r]
            valid = slots < target.padded_local[dim]
            result[p][slots[valid]] = buffer[valid]
    return result
