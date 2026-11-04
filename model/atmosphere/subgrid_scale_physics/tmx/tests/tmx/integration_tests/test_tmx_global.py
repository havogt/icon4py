# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import dataclasses
import types
from typing import TYPE_CHECKING, Any

import gt4py.next as gtx
import numpy as np
import pytest
from gt4py.next import common as gtx_common

from icon4py.model.atmosphere.subgrid_scale_physics.tmx import tmx, tmx_global, tmx_states
from icon4py.model.common import dimension as dims, model_backends
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.states.surface_flux_state import SurfaceFluxState
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import definitions, test_utils

from ..fixtures import *  # noqa: F403
from .utils import (
    assert_tmx_exit_fields,
    construct_input_state,
    construct_interpolation_state,
    construct_metric_state,
    construct_surface_flux_state,
)


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import icon as icon_grid_
    from icon4py.model.testing import serialbox as sb


_DATES = [
    (definitions.Experiments.EXCLAIM_APE_AES, date)
    for date in definitions.Experiments.EXCLAIM_APE_AES.dates[1:]
]


def _setup(
    *,
    data_provider: sb.IconSerialDataProvider,
    grid_savepoint: sb.IconGridSavepoint,
    metrics_savepoint: sb.MetricSavepoint,
    interpolation_savepoint: sb.InterpolationSavepoint,
    allocator: gtx_typing.Allocator | None,
) -> dict[str, Any]:
    return {
        "metric_state": construct_metric_state(
            metrics_savepoint=metrics_savepoint,
            init_savepoint=data_provider.from_savepoint_tmx_init(),
            allocator=allocator,
        ),
        "interpolation_state": construct_interpolation_state(interpolation_savepoint),
        "edge_params": grid_savepoint.construct_edge_geometry(),
        "cell_params": grid_savepoint.construct_cell_geometry(),
    }


def _inputs(
    data_provider: sb.IconSerialDataProvider, date: str, allocator: gtx_typing.Allocator | None
) -> tuple[tmx_states.TmxInputState, SurfaceFluxState]:
    entry = data_provider.from_savepoint_tmx_entry(date=date)
    return construct_input_state(entry), construct_surface_flux_state(
        data_provider.from_savepoint_tmx_surface_fluxes(date=date), entry, allocator=allocator
    )


@pytest.mark.datatest
@pytest.mark.parametrize("experiment_description, date", _DATES)
def test_tmx_global_single_step(
    *,
    data_provider: sb.IconSerialDataProvider,
    grid_savepoint: sb.IconGridSavepoint,
    metrics_savepoint: sb.MetricSavepoint,
    interpolation_savepoint: sb.InterpolationSavepoint,
    icon_grid: icon_grid_.IconGrid,
    decomposition_info: decomposition.DecompositionInfo,
    backend: gtx_typing.Backend | None,
    date: str,
    experiment: definitions.Experiment,
) -> None:
    config = experiment.config.tmx
    assert config is not None
    allocator = model_backends.get_allocator(backend)
    step = tmx_global.TmxGlobal(
        grid=icon_grid,
        config=config,
        **_setup(
            data_provider=data_provider,
            grid_savepoint=grid_savepoint,
            metrics_savepoint=metrics_savepoint,
            interpolation_savepoint=interpolation_savepoint,
            allocator=allocator,
        ),
        allocator=allocator,
        backend=backend,
    )
    input_state, surface_flux_state = _inputs(data_provider, date, allocator)
    temperature_in = input_state.temperature.asnumpy().copy()

    _, tendencies, diagnostics = step.run(
        input_state=input_state,
        surface_flux_state=surface_flux_state,
        dtime=experiment.config.driver.dtime.total_seconds(),
    )

    np.testing.assert_array_equal(input_state.temperature.asnumpy(), temperature_in)
    assert_tmx_exit_fields(
        tendency_state=tendencies,
        diagnostic_state=types.SimpleNamespace(**diagnostics),
        exit_savepoint=data_provider.from_savepoint_tmx_exit(date=date),
        use_km_const=config.use_km_const,
        owner_mask=data_alloc.as_numpy(decomposition_info.owner_mask(dims.CellDim)),
    )


def _synthetic_surface_fluxes(
    fluxes: SurfaceFluxState, allocator: gtx_typing.Allocator | None
) -> SurfaceFluxState:
    """Nonzero wind stress, evaporation and snow melt, which the savepoints have as zeros."""
    rng = np.random.default_rng(29)
    domain = fluxes.evapotranspiration.domain
    num_cells = domain.shape[0]

    def field(scale: float) -> gtx.Field:
        return gtx.as_field(domain, scale * rng.standard_normal(num_cells), allocator=allocator)

    return dataclasses.replace(
        fluxes,
        u_stress=field(0.1),
        v_stress=field(0.1),
        evapotranspiration=field(1.0e-5),
        q_snocpymlt=field(1.0),
        vapor_energy_flux=field(10.0),
    )


_CONFIGS = {
    "experiment": {},
    "classic_km_const_scaled": {
        "use_louis": False,
        "use_km_const": True,
        "km_const": 2.0,
        "use_scale_turb_energy_flux": True,
        "scale_turb_energy_flux": 0.5,
    },
}


@pytest.mark.datatest
@pytest.mark.parametrize("experiment_description, date", _DATES[:1])
@pytest.mark.parametrize(
    "config_name, synthetic_fluxes",
    [("experiment", False), ("experiment", True), ("classic_km_const_scaled", True)],
)
def test_tmx_global_matches_granule(
    *,
    data_provider: sb.IconSerialDataProvider,
    grid_savepoint: sb.IconGridSavepoint,
    metrics_savepoint: sb.MetricSavepoint,
    interpolation_savepoint: sb.InterpolationSavepoint,
    icon_grid: icon_grid_.IconGrid,
    backend: gtx_typing.Backend | None,
    date: str,
    experiment: definitions.Experiment,
    config_name: str,
    synthetic_fluxes: bool,
) -> None:
    if backend is None:
        pytest.skip("the granule is the witness on a compiled backend.")
    assert experiment.config.tmx is not None
    config = dataclasses.replace(experiment.config.tmx, **_CONFIGS[config_name])
    allocator = model_backends.get_allocator(backend)
    setup = _setup(
        data_provider=data_provider,
        grid_savepoint=grid_savepoint,
        metrics_savepoint=metrics_savepoint,
        interpolation_savepoint=interpolation_savepoint,
        allocator=allocator,
    )
    dtime = experiment.config.driver.dtime.total_seconds()
    input_state, surface_flux_state = _inputs(data_provider, date, allocator)
    if synthetic_fluxes:
        surface_flux_state = _synthetic_surface_fluxes(surface_flux_state, allocator)

    granule = tmx.Tmx(
        grid=icon_grid,
        config=config,
        **setup,
        backend=backend,
        exchange=decomposition.SingleNodeExchange(),
    )
    diagnostic_state = tmx_states.TmxDiagnosticState.allocate(icon_grid, allocator=allocator)
    tendency_state = tmx_states.TmxTendencyState.allocate(icon_grid, allocator=allocator)
    new_state = tmx_states.TmxNewState.allocate(icon_grid, allocator=allocator)
    granule.run(
        input_state=input_state,
        surface_flux_state=surface_flux_state,
        diagnostic_state=diagnostic_state,
        tendency_state=tendency_state,
        new_state=new_state,
        dtime=dtime,
    )

    step = tmx_global.TmxGlobal(grid=icon_grid, config=config, **setup, allocator=None)
    global_new, global_tendencies, global_diagnostics = step.run(
        input_state=input_state, surface_flux_state=surface_flux_state, dtime=dtime
    )

    num_levels = icon_grid.num_levels
    # the granule writes neither w on the top and surface half levels nor km, kh on the lowest
    # level without `use_km_const`
    interior_half = slice(1, num_levels)
    km_levels = slice(None) if config.use_km_const else slice(0, num_levels - 1)
    compared = [
        *(
            (f"new_{f.name}", getattr(global_new, f.name), getattr(new_state, f.name))
            for f in dataclasses.fields(tmx_states.TmxNewState)
        ),
        *(
            (f.name, getattr(global_tendencies, f.name), getattr(tendency_state, f.name))
            for f in dataclasses.fields(tmx_states.TmxTendencyState)
        ),
        *(
            (name, global_diagnostics[name], getattr(diagnostic_state, name))
            for name in (
                "km",
                "kh",
                "dissip_ke",
                "heating",
                "cptgz_vi",
                "dissip_ke_vi",
                "int_energy_vi",
                "tend_int_energy_vi",
            )
        ),
    ]
    for name, actual, expected in compared:
        a, e = actual.asnumpy(), expected.asnumpy()
        if name in ("new_w", "tend_w"):
            a, e = a[:, interior_half], e[:, interior_half]
        if name in ("km", "kh"):
            a, e = a[:, km_levels], e[:, km_levels]
        diff = np.max(np.abs(a - e))
        print(f"{name}: max abs diff {diff:.3e}, max abs {np.max(np.abs(e)):.3e}")
        assert test_utils.dallclose(a, e, rtol=1e-12, atol=1e-30), name
    np.testing.assert_array_equal(global_tendencies.tend_w.asnumpy()[:, [0, num_levels]], 0.0)


def _to_jax(obj: Any, jnp: Any) -> Any:
    if isinstance(obj, gtx_common.Connectivity):
        return gtx.as_connectivity(
            obj.domain,
            obj.codomain,
            jnp.asarray(obj.asnumpy()),
            skip_value=obj.skip_value,
            allocator=jnp,
        )
    if isinstance(obj, gtx.Field):
        return gtx.as_field(obj.domain, jnp.asarray(obj.asnumpy()), allocator=jnp)
    if isinstance(obj, tuple):
        return tuple(_to_jax(x, jnp) for x in obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.replace(
            obj,
            **{
                f.name: _to_jax(getattr(obj, f.name), jnp)
                for f in dataclasses.fields(obj)
                if f.init
            },
        )
    return obj


def _fields_of(state: Any) -> dict[str, gtx.Field]:
    return {f.name: getattr(state, f.name) for f in dataclasses.fields(state)}


@pytest.mark.datatest
@pytest.mark.parametrize("experiment_description, date", _DATES[:1])
def test_tmx_global_single_step_jax_jit(
    *,
    data_provider: sb.IconSerialDataProvider,
    grid_savepoint: sb.IconGridSavepoint,
    metrics_savepoint: sb.MetricSavepoint,
    interpolation_savepoint: sb.InterpolationSavepoint,
    icon_grid: icon_grid_.IconGrid,
    decomposition_info: decomposition.DecompositionInfo,
    backend: gtx_typing.Backend | None,
    date: str,
    experiment: definitions.Experiment,
) -> None:
    if backend is not None:
        pytest.skip("jax fields run on the embedded backend only.")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    jnp = jax.numpy

    config = experiment.config.tmx
    assert config is not None
    setup = _setup(
        data_provider=data_provider,
        grid_savepoint=grid_savepoint,
        metrics_savepoint=metrics_savepoint,
        interpolation_savepoint=interpolation_savepoint,
        allocator=None,
    )
    step = tmx_global.TmxGlobal(
        grid=dataclasses.replace(
            icon_grid,
            connectivities={k: _to_jax(v, jnp) for k, v in icon_grid.connectivities.items()},
        ),
        config=config,
        **{name: _to_jax(value, jnp) for name, value in setup.items()},
        allocator=jnp,
    )
    dtime = experiment.config.driver.dtime.total_seconds()
    input_state, surface_flux_state = _inputs(data_provider, date, None)

    @jax.jit
    def run(inputs: dict[str, gtx.Field], fluxes: dict[str, gtx.Field]) -> Any:
        _, tendencies, diagnostics = step.run(
            input_state=tmx_states.TmxInputState(**inputs),
            surface_flux_state=SurfaceFluxState(**fluxes),
            dtime=dtime,
        )
        return _fields_of(tendencies), diagnostics

    tendencies, diagnostics = run(
        _fields_of(_to_jax(input_state, jnp)), _fields_of(_to_jax(surface_flux_state, jnp))
    )

    assert isinstance(tendencies["tend_temperature"].ndarray, jax.Array)
    assert_tmx_exit_fields(
        tendency_state=tmx_states.TmxTendencyState(**tendencies),
        diagnostic_state=types.SimpleNamespace(**diagnostics),
        exit_savepoint=data_provider.from_savepoint_tmx_exit(date=date),
        use_km_const=config.use_km_const,
        owner_mask=data_alloc.as_numpy(decomposition_info.owner_mask(dims.CellDim)),
    )
