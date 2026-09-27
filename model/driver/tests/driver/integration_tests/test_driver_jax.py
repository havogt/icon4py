# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import datetime
import pathlib

import gt4py.next.typing as gtx_typing
import numpy as np
import pytest

from icon4py.model.common import model_backends
from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.common.states import prognostic_state as prognostics
from icon4py.model.driver import config as driver_config, driver, driver_utils
from icon4py.model.testing import (
    datatest_utils as dt_utils,
    definitions as test_defs,
    grid_utils,
    serialbox as sb,
    test_utils,
)

from ..fixtures import *  # noqa: F403
from .test_driver import _TOLERANCES


_FIELDS = ("vn", "w", "rho", "exner", "theta_v")


@pytest.mark.datatest
@pytest.mark.embedded_only
@pytest.mark.parametrize(
    "experiment_description, timeloop_date_init, step_dates_exit",
    [
        (
            test_defs.Experiments.JW,
            "2008-09-01T00:00:00.000",
            ("2008-09-01T00:05:00.000", "2008-09-01T00:10:00.000", "2008-09-01T00:15:00.000"),
        ),
    ],
)
@pytest.mark.parametrize("jit_time_step", [False, True])
def test_driver_jax(
    experiment_description: test_defs.ExperimentDescription,
    timeloop_date_init: str,
    step_dates_exit: tuple[str, ...],
    *,
    tmp_path: pathlib.Path,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend | None,
    data_provider: sb.IconSerialDataProvider,
    jit_time_step: bool,
) -> None:
    """
    The driver with the global steps under jax.jit, validated after every time step against
    the end-of-time-step savepoints, with the tolerances of `test_driver`.
    """
    pytest.importorskip("jax")
    grid_file_path = grid_utils._download_grid_file(experiment_description.grid)
    config_file_path = dt_utils.get_path_for_experiment(experiment_description, process_props)
    config = driver_config.read_experiment_config_from_fortran(config_file_path)
    config = config.with_overrides(
        driver={
            "output_path": tmp_path / "ci_driver_output",
            "start_of_timestepping": datetime.datetime.fromisoformat(timeloop_date_init).replace(
                tzinfo=datetime.UTC
            ),
            "end_of_simulation": len(step_dates_exit),
        }
    )
    grid_manager = driver_utils.create_grid_manager(
        grid_file_path=grid_file_path,
        vertical_grid_config=config.vertical_grid,
        allocator=model_backends.get_allocator(backend),
        process_props=process_props,
    )

    computed: list[dict[str, np.ndarray]] = []

    def record(time_step: int, state: prognostics.PrognosticState) -> None:
        computed.append({name: getattr(state, name).asnumpy() for name in _FIELDS})

    ds, _ = driver.run_driver(
        config=config,
        grid_manager=grid_manager,
        process_props=process_props,
        backend=None,
        jax=True,
        jax_jit_time_step=jit_time_step,
        on_step_end=record,
    )

    assert len(computed) == len(step_dates_exit)
    for name in _FIELDS:
        np.testing.assert_array_equal(
            getattr(ds.prognostics.current, name).asnumpy(), computed[-1][name]
        )
    tolerances = _TOLERANCES[experiment_description]
    references = [
        {
            name: getattr(data_provider.from_savepoint_time_step_exit(date=date), name)().asnumpy()
            for name in _FIELDS
        }
        for date in step_dates_exit
    ]
    for step, date in enumerate(step_dates_exit):
        print(
            f"{date}: max |jax - savepoint| "
            + ", ".join(
                f"{name}={np.max(np.abs(computed[step][name] - references[step][name])):.2e}"
                for name in _FIELDS
            )
        )
    for step, date in enumerate(step_dates_exit):
        for name in _FIELDS:
            atol, rtol = tolerances[name]
            test_utils.assert_dallclose(
                computed[step][name],
                references[step][name],
                atol=atol,
                rtol=rtol,
                err_msg=f"{name} at {date}",
            )
