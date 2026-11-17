# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.driver import jax_utils
from icon4py.model.testing.fixtures.datatest import process_props


@pytest.mark.mpi
@pytest.mark.parametrize("process_props", [True], indirect=True)
def test_initialize_distributed_twice(process_props: decomp_defs.ProcessProperties) -> None:
    pytest.importorskip("jax")
    from jax._src import (  # type: ignore[import-not-found]  # noqa: PLC0415 [import-outside-top-level]
        xla_bridge,
    )

    if xla_bridge.backends_are_initialized():
        pytest.skip(
            "jax.distributed must start before JAX runs anything in the process: run this module "
            "on its own."
        )
    comm = process_props.comm
    jax_utils.initialize_distributed(comm)
    jax_utils.initialize_distributed(comm)
    jax = jax_utils.import_jax()
    mesh = jax_utils.rank_mesh("rank")
    sharded = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("rank"))
    ranks = jax.make_array_from_process_local_data(
        sharded, np.array([comm.rank], dtype=np.int64), (comm.size,)
    )
    total = jax.jit(
        lambda x: x.sum(),
        out_shardings=jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()),
    )(ranks)

    assert jax.process_count() == comm.size
    assert int(total.addressable_data(0)) == comm.allreduce(comm.rank)
