# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import datetime
import gc
import pathlib
from typing import Any

import numpy as np
import pytest

from icon4py.model.common import dimension as dims, model_backends
from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.common.states import tracer_states
from icon4py.model.driver import config as driver_config, driver, driver_utils
from icon4py.model.testing import datatest_utils as dt_utils, definitions as test_defs, grid_utils
from icon4py.model.testing.fixtures.datatest import backend, backend_like, process_props


_FIELDS = ("vn", "w", "rho", "exner", "theta_v")


def _run(  # noqa: PLR0917 [too-many-positional-arguments]
    experiment: test_defs.ExperimentDescription,
    start: datetime.datetime,
    num_steps: int,
    process_props: decomp_defs.ProcessProperties,
    tmp_path: pathlib.Path,
    tracer_config: tracer_states.TracerConfig | None,
) -> tuple[dict[Any, np.ndarray], dict[str, np.ndarray]]:
    """The JAX driver's state after `num_steps` steps: the global index and fields of the owned points."""
    config = driver_config.read_experiment_config_from_yaml(
        dt_utils.get_path_for_experiment(experiment, decomp_defs.SingleNodeProcessProperties())
        / "config.yml"
    )
    config = config.with_overrides(
        driver={
            "output_path": tmp_path
            / f"output_{process_props.comm_size}_ranks_rank_{process_props.rank}",
            "start_of_timestepping": start,
            "end_of_simulation": num_steps,
        }
    )
    if tracer_config is not None:
        config = config.with_overrides(tracer_config=tracer_config)
    distributed = not process_props.is_single_rank()
    grid_manager = driver_utils.create_grid_manager(
        grid_file_path=grid_utils._download_grid_file(experiment.grid),
        vertical_grid_config=config.vertical_grid,
        allocator=model_backends.get_allocator(None),
        process_props=process_props,
        extra_halo_rings=driver_utils.JAX_EXTRA_HALO_RINGS if distributed else 0,
    )
    info = grid_manager.decomposition_info
    owned = {
        dim: np.asarray(info.global_index(dim, decomp_defs.DecompositionInfo.EntryType.OWNED))
        for dim in (dims.CellDim, dims.EdgeDim)
    }
    ds, _ = driver.run_driver(
        config=config,
        grid_manager=grid_manager,
        process_props=process_props,
        backend=None,
        jax=True,
        jax_exchange_read_fields_only=distributed,
    )
    fields = {}
    for name in _FIELDS:
        field = getattr(ds.prognostics.current, name)
        fields[name] = field.asnumpy()[: owned[field.domain.dims[0]].size]
    for tracer in ds.tracers.current.active_fields():
        fields[tracer.name] = tracer.field.asnumpy()[: owned[dims.CellDim].size]
    return owned, fields


@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.embedded_only
@pytest.mark.parametrize("process_props", [True], indirect=True)
@pytest.mark.parametrize(
    "experiment, start, num_steps, tracer_config",
    [
        (
            test_defs.Experiments.GAUSS3D,
            datetime.datetime(2001, 1, 1, tzinfo=datetime.UTC),
            2,
            None,
        ),
    ],
)
def test_parallel_driver_jax_compare_single_multi_rank(  # noqa: PLR0917 [too-many-positional-arguments]
    experiment: test_defs.ExperimentDescription,
    start: datetime.datetime,
    num_steps: int,
    tracer_config: tracer_states.TracerConfig | None,
    tmp_path: pathlib.Path,
    process_props: decomp_defs.ProcessProperties,
) -> None:
    """The JAX driver on several ranks against the JAX driver on one rank, as `test_parallel_driver`."""
    jax = pytest.importorskip("jax")
    single = (
        _run(
            experiment,
            start,
            num_steps,
            decomp_defs.SingleNodeProcessProperties(),
            tmp_path,
            tracer_config,
        )[1]
        if process_props.rank == 0
        else None
    )
    # the single-rank driver's compiled steps and constants would stay alive next to the
    # multi-rank ones on rank 0
    jax.clear_caches()
    gc.collect()
    owned, fields = _run(experiment, start, num_steps, process_props, tmp_path, tracer_config)
    gathered = process_props.comm.gather((owned, fields), root=0)
    failures = []
    if process_props.rank == 0:
        assert single is not None
        for name in single:
            dim = dims.EdgeDim if name == "vn" else dims.CellDim
            index = np.concatenate([rank_owned[dim] for rank_owned, _ in gathered])
            assert np.array_equal(np.sort(index), np.arange(index.size)), (
                "owned points are no partition"
            )
            multi = np.concatenate([rank_fields[name] for _, rank_fields in gathered])[
                np.argsort(index)
            ]
            error = float(np.max(np.abs(multi - single[name])))
            print(
                f"{experiment.name} {name}: max |{process_props.comm_size} ranks - 1 rank| = {error:.2e}"
            )
            if not np.allclose(multi, single[name], atol=1e-9, rtol=0.0):
                failures.append(name)
    failures = process_props.comm.bcast(failures, root=0)
    assert not failures, f"multi-rank differs from single-rank: {failures}"
