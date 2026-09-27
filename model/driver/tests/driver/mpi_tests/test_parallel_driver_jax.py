# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import datetime
import pathlib

import numpy as np
import pytest

from icon4py.model.common import dimension as dims, model_backends
from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.common.states import prognostic_state as prognostics
from icon4py.model.driver import config as driver_config, driver, driver_utils
from icon4py.model.testing import (
    datatest_utils as dt_utils,
    definitions as test_defs,
    grid_utils,
    test_utils,
)
from icon4py.model.testing.fixtures.datatest import backend, backend_like, process_props

from ..integration_tests.test_driver import _TOLERANCES


_FIELDS = ("vn", "w", "rho", "exner", "theta_v")
_JW_STEP_DATES_EXIT = (
    "2008-09-01T00:05:00.000",
    "2008-09-01T00:10:00.000",
    "2008-09-01T00:15:00.000",
)


def run_jw_gathered(
    process_props: decomp_defs.ProcessProperties,
    tmp_path: pathlib.Path,
    num_steps: int,
    extra_halo_rings: int,
    exchange_read_fields_only: bool = False,
) -> list[dict[str, np.ndarray]] | None:
    """
    Run the JAX driver on JW for `num_steps` steps and return the prognostic fields after every
    step on the global grid, assembled on rank 0 from the owned points of every rank.
    """
    experiment = test_defs.Experiments.JW
    grid_file_path = grid_utils._download_grid_file(experiment.grid)
    config = driver_config.read_experiment_config_from_fortran(
        dt_utils.get_path_for_experiment(experiment, decomp_defs.SingleNodeProcessProperties())
    )
    config = config.with_overrides(
        driver={
            "output_path": tmp_path / f"ci_driver_output_rank_{process_props.rank}",
            "start_of_timestepping": datetime.datetime(2008, 9, 1, tzinfo=datetime.UTC),
            "end_of_simulation": num_steps,
        }
    )
    grid_manager = driver_utils.create_grid_manager(
        grid_file_path=grid_file_path,
        vertical_grid_config=config.vertical_grid,
        allocator=model_backends.get_allocator(None),
        process_props=process_props,
        extra_halo_rings=extra_halo_rings,
    )
    info = grid_manager.decomposition_info
    owned_global_index = {
        dim: np.asarray(info.global_index(dim, decomp_defs.DecompositionInfo.EntryType.OWNED))
        for dim in (dims.CellDim, dims.EdgeDim)
    }

    owned_per_step: list[dict[str, np.ndarray]] = []

    def record(time_step: int, state: prognostics.PrognosticState) -> None:
        fields = {}
        for name in _FIELDS:
            field = getattr(state, name)
            num_owned = owned_global_index[field.domain.dims[0]].size
            fields[name] = field.asnumpy()[:num_owned]
        owned_per_step.append(fields)

    driver.run_driver(
        config=config,
        grid_manager=grid_manager,
        process_props=process_props,
        backend=None,
        jax=True,
        jax_exchange_read_fields_only=exchange_read_fields_only,
        on_step_end=record,
    )

    gathered = process_props.comm.gather((owned_global_index, owned_per_step), root=0)
    if process_props.rank != 0:
        return None
    global_index = {
        dim: np.concatenate([rank_global_index[dim] for rank_global_index, _ in gathered])
        for dim in owned_global_index
    }
    for index in global_index.values():
        assert np.array_equal(np.sort(index), np.arange(index.size)), (
            "owned points are no partition"
        )
    return [
        {
            name: np.concatenate([rank_per_step[step][name] for _, rank_per_step in gathered])[
                np.argsort(global_index[dims.EdgeDim if name == "vn" else dims.CellDim])
            ]
            for name in _FIELDS
        }
        for step in range(num_steps)
    ]


@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.embedded_only
@pytest.mark.parametrize("process_props", [True], indirect=True)
def test_parallel_driver_jax(
    tmp_path: pathlib.Path,
    process_props: decomp_defs.ProcessProperties,
) -> None:
    """
    The JAX driver on JW on several ranks: the assembled global fields after every time step
    against the end-of-time-step savepoints of the single-rank Fortran run, with the tolerances
    of `test_driver`.
    """
    pytest.importorskip("jax")
    computed = run_jw_gathered(
        process_props,
        tmp_path,
        num_steps=len(_JW_STEP_DATES_EXIT),
        extra_halo_rings=driver_utils.JAX_EXTRA_HALO_RINGS,
        exchange_read_fields_only=True,
    )
    failures = []
    if computed is not None:
        experiment = test_defs.Experiments.JW
        data_provider = dt_utils.create_icon_serial_data_provider(
            dt_utils.get_datapath_for_experiment(
                experiment, decomp_defs.SingleNodeProcessProperties()
            ),
            rank=0,
            backend=None,
        )
        tolerances = _TOLERANCES[experiment]
        for step, date in enumerate(_JW_STEP_DATES_EXIT):
            reference = data_provider.from_savepoint_time_step_exit(date=date)
            errors = {}
            for name in _FIELDS:
                expected = getattr(reference, name)().asnumpy()
                assert computed[step][name].shape == expected.shape
                errors[name] = np.max(np.abs(computed[step][name] - expected))
                atol, rtol = tolerances[name]
                if not test_utils.dallclose(computed[step][name], expected, atol=atol, rtol=rtol):
                    failures.append(f"{name} at {date}")
            print(
                f"{date}: max |jax on {process_props.comm_size} ranks - savepoint| "
                + ", ".join(f"{name}={error:.2e}" for name, error in errors.items())
            )
    failures = process_props.comm.bcast(failures, root=0)
    assert not failures, f"outside the tolerances: {failures}"
