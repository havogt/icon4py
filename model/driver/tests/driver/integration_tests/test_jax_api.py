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

import numpy as np
import pytest

from icon4py.model.common import constants, model_backends
from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.common.states import tracer_states
from icon4py.model.driver import config as driver_config, driver, driver_utils
from icon4py.model.testing import datatest_utils as dt_utils, definitions as test_defs, grid_utils


_FIELDS = ("vn", "w", "rho", "exner", "theta_v")


def _grouped_time_step(static, state):  # type: ignore[no-untyped-def]
    from icon4py.model.driver import jax_api  # noqa: PLC0415 [import-outside-top-level]

    state = jax_api.dynamics_step(static, state)
    if static.setup.granules.physics is None:
        return state
    col = jax_api.dyn2phy(static, state)
    applied = []
    tend, _ = jax_api.muphys(static, col)
    col = jax_api.apply_column_tendencies(col, tend, static.dtime)
    applied.append(tend)
    fluxes = jax_api.surface(static, jax_api.surface_input(static, col))
    tend, _ = jax_api.tmx(static, col, fluxes)
    col = jax_api.apply_column_tendencies(col, tend, static.dtime)
    applied.append(tend)
    return jax_api.phy2dyn(static, state, col, jax_api.sum_column_tendencies(*applied))


@pytest.mark.datatest
@pytest.mark.embedded_only
@pytest.mark.parametrize(
    "experiment, num_steps, tracer_config",
    [
        (test_defs.Experiments.JW, 2, None),
        (test_defs.Experiments.JW, 2, tracer_states.TracerConfig(qv=True)),
        (test_defs.Experiments.EXCLAIM_APE_AES, 1, None),
    ],
)
@pytest.mark.parametrize("composition", ["time_step", "groups"])
def test_jax_api_reproduces_jax_driver(
    experiment: test_defs.ExperimentDescription,
    num_steps: int,
    tracer_config: tracer_states.TracerConfig | None,
    composition: str,
    tmp_path: pathlib.Path,
) -> None:
    """`jax_api` on one rank, bitwise equal to the JAX driver with the jitted time step."""
    jax = pytest.importorskip("jax")
    from icon4py.model.driver import jax_api  # noqa: PLC0415 [import-outside-top-level]

    config = driver_config.read_experiment_config_from_yaml(
        dt_utils.get_path_for_experiment(experiment, decomp_defs.SingleNodeProcessProperties())
        / "config.yml"
    )
    config = config.with_overrides(
        driver={
            "output_path": tmp_path / "output",
            "start_of_timestepping": datetime.datetime(2008, 9, 1, tzinfo=datetime.UTC),
            "end_of_simulation": num_steps,
        }
    )
    if tracer_config is not None:
        config = config.with_overrides(tracer_config=tracer_config)
    grid_file = grid_utils._download_grid_file(experiment.grid)
    process_props = decomp_defs.SingleNodeProcessProperties()
    ds, _ = driver.run_driver(
        config=config,
        grid_manager=driver_utils.create_grid_manager(
            grid_file_path=grid_file,
            vertical_grid_config=config.vertical_grid,
            allocator=model_backends.get_allocator(None),
            process_props=process_props,
        ),
        process_props=process_props,
        backend=None,
        jax=True,
        jax_jit_time_step=True,
    )
    expected = {name: getattr(ds.prognostics.current, name).asnumpy() for name in _FIELDS}
    expected |= {t.name: t.field.asnumpy() for t in ds.tracers.current.active_fields()}
    del ds
    jax.clear_caches()
    gc.collect()

    static, state = jax_api.init_from_grid_file(grid_file, config)
    step = jax.jit(jax_api.time_step if composition == "time_step" else _grouped_time_step)
    for _ in range(num_steps):
        state = step(static, state)
    computed = {name: state.prognostic[name].asnumpy() for name in _FIELDS}
    computed |= {name: field.asnumpy() for name, field in state.tracers.items()}

    assert computed.keys() == expected.keys()
    for name in expected:
        np.testing.assert_array_equal(computed[name], expected[name], err_msg=name)


@pytest.mark.datatest
@pytest.mark.embedded_only
def test_jax_api_external_physics_without_tendencies(tmp_path: pathlib.Path) -> None:
    """
    With `external_physics` and no ICON process, `dyn2phy` and `phy2dyn` with no tendencies after
    `dynamics_step` give the dynamics alone, except that `phy2dyn` diagnoses exner and theta_v from
    rho and the virtual temperature with the equation of state, which the dynamics' exner meets
    only to its own accuracy: the witness for those two is that diagnosis, here in NumPy.
    """
    jax = pytest.importorskip("jax")
    from icon4py.model.driver import jax_api  # noqa: PLC0415 [import-outside-top-level]

    experiment = test_defs.Experiments.JW
    config = driver_config.read_experiment_config_from_yaml(
        dt_utils.get_path_for_experiment(experiment, decomp_defs.SingleNodeProcessProperties())
        / "config.yml"
    )
    config = config.with_overrides(
        driver={
            "output_path": tmp_path / "output",
            "start_of_timestepping": datetime.datetime(2008, 9, 1, tzinfo=datetime.UTC),
            "end_of_simulation": 1,
        },
        tracer_config=tracer_states.TracerConfig(
            qv=True, qc=True, qi=True, qr=True, qs=True, qg=True
        ),
        external_physics=True,
    )
    assert config.muphys is None and config.surface is None and config.tmx is None
    static, initial = jax_api.init_from_grid_file(
        grid_utils._download_grid_file(experiment.grid), config
    )
    physics = static.setup.granules.physics
    assert physics is not None and not physics.has_processes

    def external(static, state):  # type: ignore[no-untyped-def]
        state = jax_api.dynamics_step(static, state)
        col = jax_api.dyn2phy(static, state)
        return jax_api.phy2dyn(static, state, col, jax_api.ColumnTendencies())

    results = {}
    for name, step in (("dynamics", jax_api.time_step), ("external", external)):
        jitted, state = jax.jit(step), initial
        state = jitted(static, state)
        results[name] = {k: state.prognostic[k].asnumpy() for k in _FIELDS} | {
            k: v.asnumpy() for k, v in state.tracers.items()
        }

    dynamics, external_ = results["dynamics"], results["external"]
    virtual_temperature = dynamics["theta_v"] * dynamics["exner"]
    exner = np.exp(
        constants.RD_O_CPD
        * np.log(constants.RD / constants.P0REF * dynamics["rho"] * virtual_temperature)
    )
    expected_fields = dynamics | {"exner": exner, "theta_v": virtual_temperature / exner}
    for name, expected in expected_fields.items():
        actual = external_[name]
        print(
            f"{name}: max |external - expected| {np.max(np.abs(actual - expected)):.3e}, "
            f"max {np.max(np.abs(expected)):.3e}, against dynamics {np.max(np.abs(actual - dynamics[name])):.3e}"
        )
        if name in ("exner", "theta_v"):
            np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0, err_msg=name)
        else:
            np.testing.assert_array_equal(actual, expected, err_msg=name)
