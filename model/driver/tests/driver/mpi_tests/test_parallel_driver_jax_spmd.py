# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pathlib

import pytest

from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.driver import driver_utils
from icon4py.model.testing.fixtures.datatest import backend, backend_like, process_props

from .test_parallel_driver_jax import _JW_STEP_DATES_EXIT, check_against_savepoints, run_jw_gathered


@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.embedded_only
@pytest.mark.parametrize("process_props", [True], indirect=True)
def test_parallel_driver_jax_spmd(
    tmp_path: pathlib.Path,
    process_props: decomp_defs.ProcessProperties,
) -> None:
    """The JAX driver on JW as one SPMD program over all ranks, against the savepoints."""
    pytest.importorskip("jax")
    from jax._src import (  # type: ignore[import-not-found]  # noqa: PLC0415 [import-outside-top-level]
        xla_bridge,
    )

    if xla_bridge.backends_are_initialized():
        pytest.skip(
            "jax.distributed must start before JAX runs anything in the process: run this module "
            "on its own."
        )
    computed = run_jw_gathered(
        process_props,
        tmp_path,
        num_steps=len(_JW_STEP_DATES_EXIT),
        extra_halo_rings=driver_utils.JAX_EXTRA_HALO_RINGS,
        exchange_read_fields_only=True,
        spmd=True,
    )
    check_against_savepoints(process_props, computed)
