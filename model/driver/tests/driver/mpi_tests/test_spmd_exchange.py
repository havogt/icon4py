# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import numpy as np
import pytest

from icon4py.model.common import dimension as dims, model_backends
from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.driver import driver_utils, jax_utils, spmd_layout
from icon4py.model.testing import definitions as test_defs, grid_utils
from icon4py.model.testing.fixtures.datatest import backend, backend_like, process_props


@pytest.fixture(scope="module")
def spmd_layout_and_info(
    process_props: decomp_defs.ProcessProperties,
) -> tuple[spmd_layout.PaddedLayout, decomp_defs.DecompositionInfo]:
    pytest.importorskip("jax")
    from jax._src import (  # type: ignore[import-not-found]  # noqa: PLC0415 [import-outside-top-level]
        xla_bridge,
    )

    if xla_bridge.backends_are_initialized():
        pytest.skip(
            "jax.distributed must start before JAX runs anything in the process: run this module "
            "on its own."
        )
    jax_utils.initialize_distributed(process_props.comm)
    grid_manager = driver_utils.create_grid_manager(
        grid_file_path=grid_utils._download_grid_file(test_defs.Grids.R02B04_GLOBAL),
        vertical_grid_config=v_grid.VerticalGridConfig(num_levels=2),
        allocator=model_backends.get_allocator(None),
        process_props=process_props,
        extra_halo_rings=driver_utils.JAX_EXTRA_HALO_RINGS,
    )
    info = grid_manager.decomposition_info
    layout = spmd_layout.build_padded_layout(
        info, process_props.rank, process_props.comm_size, process_props.comm.allgather
    )
    return layout, info


@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.parametrize("process_props", [True], indirect=True)
@pytest.mark.parametrize("transport", list(driver_utils.SpmdTransport))
def test_spmd_exchange_and_adjoint(
    process_props: decomp_defs.ProcessProperties,
    spmd_layout_and_info: tuple[spmd_layout.PaddedLayout, decomp_defs.DecompositionInfo],
    transport: driver_utils.SpmdTransport,
) -> None:
    """
    The collective halo exchange of the SPMD JAX driver on a decomposed R02B04 grid: the halo
    against the global field, and the adjoint JAX computes against the exchange (dot-product test).
    """
    from mpi4py import MPI  # noqa: PLC0415 [import-outside-top-level]

    from icon4py.model.driver import jax_driver  # noqa: PLC0415 [import-outside-top-level]

    comm = process_props.comm
    layout, info = spmd_layout_and_info
    jax = jax_utils.import_jax()
    jnp = jax.numpy
    coloured = transport == driver_utils.SpmdTransport.PPERMUTE
    mesh = jax.make_mesh((layout.num_ranks,), ("rank",))
    sharded = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("rank"))

    def to_global(x: np.ndarray) -> object:
        return jax.make_array_from_process_local_data(
            sharded, x, (layout.num_ranks * x.shape[0], *x.shape[1:])
        )

    num_levels = 3
    rng = np.random.default_rng(0)
    for dim in dims.horizontal_dims():
        global_index = np.asarray(info.global_index(dim))
        num_global = comm.allreduce(int(global_index.max()) + 1, op=MPI.MAX)
        global_field = rng.standard_normal((num_global, num_levels))
        padded_owned, padded_local = layout.padded_owned[dim], layout.padded_local[dim]
        owned = spmd_layout.pad_rows(layout, dim, global_field[global_index])[:padded_owned]
        # the real rows of the padded layout: owned ones and received halo ones
        real = np.zeros(padded_local, bool)
        real[layout.padded_index[dim]] = True

        def exchange(
            x: Any, send: Any, recv: Any, real: Any, padded_local: int = padded_local
        ) -> Any:
            exchanged = (
                jax_driver.exchange_coloured(x, send, recv, layout.rounds, padded_local)
                if coloured
                else jax_driver.exchange_collective(x, send, recv, padded_local)
            )
            return jnp.where(real[:, None], exchanged, 0.0)

        def run(body: Any, *args: Any) -> Any:
            return jax.jit(
                jax.shard_map(
                    body,
                    mesh=mesh,
                    in_specs=(jax.sharding.PartitionSpec("rank"),) * len(args),
                    out_specs=jax.sharding.PartitionSpec("rank"),
                )
            )(*(to_global(a) for a in args))

        tables = (
            (layout.send_round if coloured else layout.send_index)[dim],
            (layout.recv_round if coloured else layout.recv_index)[dim],
            real,
        )
        local = np.asarray(run(exchange, owned, *tables).addressable_shards[0].data)
        expected = global_field[global_index]
        np.testing.assert_array_equal(local[layout.padded_index[dim]], expected)

        x = rng.standard_normal((padded_owned, num_levels))
        y = rng.standard_normal((padded_local, num_levels))

        def dot_test(x: Any, y: Any, send: Any, recv: Any, real: Any) -> Any:
            lx, pullback = jax.vjp(lambda v: exchange(v, send, recv, real), x)
            (lty,) = pullback(y)
            return jnp.stack(
                [jax.lax.psum(jnp.sum(lx * y), "rank"), jax.lax.psum(jnp.sum(x * lty), "rank")]
            )[None]

        lhs, rhs = np.asarray(run(dot_test, x, y, *tables).addressable_shards[0].data)[0]
        assert abs(lhs - rhs) <= 1e-12 * abs(lhs), (dim, lhs, rhs)
        print(
            f"{transport} {dim.value}: halo == global; <Lx, y> = {lhs:.15e}, <x, L^T y> = {rhs:.15e}"
        )
