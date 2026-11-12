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
from icon4py.model.testing import (
    datatest_utils as dt_utils,
    definitions as test_defs,
    grid_utils,
    serialbox as sb,
    test_utils,
)
from icon4py.model.testing.fixtures.datatest import (
    backend,
    data_provider,
    download_ser_data,
    experiment,
    experiment_description,
    metrics_savepoint,
    process_props,
)


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


@pytest.mark.datatest
@pytest.mark.embedded_only
@pytest.mark.parametrize("experiment_description", [test_defs.Experiments.JW])
def test_jax_api_heights_are_icons(
    experiment_description: test_defs.ExperimentDescription,
    metrics_savepoint: sb.MetricSavepoint,
    tmp_path: pathlib.Path,
) -> None:
    """`IconStatic.z_mc` and `z_ifc` against ICON's serialized heights, at the metrics' tolerance."""
    pytest.importorskip("jax")
    from icon4py.model.driver import jax_api  # noqa: PLC0415 [import-outside-top-level]

    config = driver_config.read_experiment_config_from_yaml(
        dt_utils.get_path_for_experiment(
            experiment_description, decomp_defs.SingleNodeProcessProperties()
        )
        / "config.yml"
    )
    config = config.with_overrides(driver={"output_path": tmp_path / "output"})
    static, _ = jax_api.init_from_grid_file(
        grid_utils._download_grid_file(experiment_description.grid), config
    )

    for name, computed, reference in (
        ("z_mc", static.z_mc, metrics_savepoint.z_mc()),
        ("z_ifc", static.z_ifc, metrics_savepoint.z_ifc()),
    ):
        assert computed.domain.dims == reference.domain.dims, name
        actual, expected = computed.asnumpy(), reference.asnumpy()
        print(f"{name}: max |jax_api - ICON| {np.max(np.abs(actual - expected)):.3e}")
        assert test_utils.dallclose(actual, expected, rtol=1e-9), name


@pytest.mark.datatest
@pytest.mark.embedded_only
@pytest.mark.parametrize("experiment_description", [test_defs.Experiments.JW])
def test_jax_api_topography_array(
    experiment_description: test_defs.ExperimentDescription,
    metrics_savepoint: sb.MetricSavepoint,
    tmp_path: pathlib.Path,
) -> None:
    """
    The JW topography as an array, with a flat topography configured, gives ICON's serialized
    heights, at the metrics' tolerance.
    """
    pytest.importorskip("jax")
    from icon4py.model.common.topography.analytical import (  # noqa: PLC0415 [import-outside-top-level]
        flat_topography,
        jablonowski_williamson as jw_topography,
    )
    from icon4py.model.driver import jax_api  # noqa: PLC0415 [import-outside-top-level]

    config = driver_config.read_experiment_config_from_yaml(
        dt_utils.get_path_for_experiment(
            experiment_description, decomp_defs.SingleNodeProcessProperties()
        )
        / "config.yml"
    )
    assert isinstance(config.topography, jw_topography.JablonowskiWilliamsonConfig)
    grid_file = grid_utils._download_grid_file(experiment_description.grid)
    topography = jw_topography.jablonowski_williamson(
        config=config.topography,
        grid_manager=driver_utils.create_grid_manager(
            grid_file_path=grid_file,
            vertical_grid_config=config.vertical_grid,
            allocator=model_backends.get_allocator(None),
            process_props=decomp_defs.SingleNodeProcessProperties(),
        ),
    )
    config = config.with_overrides(
        driver={"output_path": tmp_path / "output"},
        topography=flat_topography.FlatTopographyConfig(),
    )
    static, _ = jax_api.init_from_grid_file(grid_file, config, topography=np.asarray(topography))

    for name, computed, reference in (
        ("z_mc", static.z_mc, metrics_savepoint.z_mc()),
        ("z_ifc", static.z_ifc, metrics_savepoint.z_ifc()),
    ):
        actual, expected = computed.asnumpy(), reference.asnumpy()
        print(f"{name}: max |jax_api - ICON| {np.max(np.abs(actual - expected)):.3e}")
        assert test_utils.dallclose(actual, expected, rtol=1e-9), name


@pytest.mark.datatest
@pytest.mark.embedded_only
def test_jax_api_initial_state_from_arrays(tmp_path: pathlib.Path) -> None:
    """
    The JW initial state, as the column arrays `dyn2phy` gives (cell wind, temperature, surface
    pressure) and its moisture, balanced by `initial_condition.from_arrays`, is the JW initial
    state again: its thermodynamics to rounding, vn and w to the interpolation of the cell wind to
    the edges, and icon4py diagnoses the given surface pressure from it.
    """
    pytest.importorskip("jax")
    from icon4py.model.common.initial_condition import (  # noqa: PLC0415 [import-outside-top-level]
        from_arrays,
    )
    from icon4py.model.driver import jax_api  # noqa: PLC0415 [import-outside-top-level]

    experiment = test_defs.Experiments.JW
    config = driver_config.read_experiment_config_from_yaml(
        dt_utils.get_path_for_experiment(experiment, decomp_defs.SingleNodeProcessProperties())
        / "config.yml"
    )
    config = config.with_overrides(
        driver={"output_path": tmp_path / "output"},
        tracer_config=tracer_states.TracerConfig(**dict.fromkeys(from_arrays.MOISTURE, True)),
        external_physics=True,
    )
    grid_file = grid_utils._download_grid_file(experiment.grid)

    def fields_and_column(static, state):  # type: ignore[no-untyped-def]
        col = jax_api.dyn2phy(static, state)
        fields = {k: state.prognostic[k].asnumpy() for k in _FIELDS} | {
            k: v.asnumpy() for k, v in state.tracers.items()
        }
        surface = col.pressure_ifc.asnumpy()[:, -1]
        return fields, col.u.asnumpy(), col.v.asnumpy(), col.temperature.asnumpy(), surface

    reference, u, v, temperature, surface_pressure = fields_and_column(
        *jax_api.init_from_grid_file(grid_file, config)
    )
    gc.collect()
    initial_state = from_arrays.ColumnInitialState(
        u=u,
        v=v,
        temperature=temperature,
        q={name: reference[name] for name in from_arrays.MOISTURE},
        surface_pressure=surface_pressure,
    )
    computed, *_, diagnosed_surface_pressure = fields_and_column(
        *jax_api.init_from_grid_file(grid_file, config, initial_state=initial_state)
    )

    for name, expected in reference.items():
        scale = np.max(np.abs(expected))
        print(
            f"{name}: max |arrays - JW| {np.max(np.abs(computed[name] - expected)):.3e}, max {scale:.3e}"
        )
    for name in ("rho", "exner", "theta_v"):
        np.testing.assert_allclose(computed[name], reference[name], rtol=1e-13, err_msg=name)
    for name in from_arrays.MOISTURE:
        np.testing.assert_array_equal(computed[name], reference[name], err_msg=name)
    np.testing.assert_allclose(
        computed["vn"], reference["vn"], rtol=0, atol=5e-3 * np.max(np.abs(reference["vn"]))
    )
    np.testing.assert_allclose(
        computed["w"], reference["w"], rtol=0, atol=1e-2 * np.max(np.abs(reference["w"]))
    )
    np.testing.assert_allclose(diagnosed_surface_pressure, surface_pressure, rtol=1e-13)
