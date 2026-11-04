# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.surface import (
    surface,
    surface_global,
    surface_states,
)
from icon4py.model.common import dimension as dims, model_backends
from icon4py.model.common.states.surface_flux_state import SurfaceFluxState
from icon4py.model.testing import definitions, test_utils

from ..fixtures import *  # noqa: F403


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import icon as icon_grid_
    from icon4py.model.testing import serialbox as sb


_DATES = [
    (definitions.Experiments.EXCLAIM_APE_AES, date)
    for date in definitions.Experiments.EXCLAIM_APE_AES.dates
]


def _input_state(
    data_provider: sb.IconSerialDataProvider, icon_grid: icon_grid_.IconGrid, date: str
) -> surface_states.SurfaceInputState:
    entry_savepoint = data_provider.from_savepoint_tmx_entry(date=date)
    return surface_states.SurfaceInputState(
        surface_pressure=entry_savepoint.pres_ifc()[dims.KHalfDim(icon_grid.num_levels)],
        temperature=entry_savepoint.ta()[dims.KDim(icon_grid.num_levels - 1)],
    )


def _assert_matches_savepoint(
    fluxes: SurfaceFluxState, reference: sb.TmxSurfaceFluxesSavepoint
) -> None:
    for name, computed, desired in (
        ("hfss", fluxes.sensible_heat_flux, reference.hfss()),
        ("evspsbl", fluxes.evapotranspiration, reference.evspsbl()),
        ("tauu", fluxes.u_stress, reference.tauu()),
        ("tauv", fluxes.v_stress, reference.tauv()),
        ("q_snocpymlt", fluxes.q_snocpymlt, reference.q_snocpymlt()),
        ("ufts", fluxes.thermal_energy_flux, reference.hfss()),
    ):
        test_utils.assert_dallclose(
            computed.asnumpy(), desired.asnumpy(), rtol=1.0e-14, err_msg=name
        )
    assert np.all(np.abs(fluxes.sensible_heat_flux.asnumpy()) > 0.0)
    assert np.all(fluxes.vapor_energy_flux.asnumpy() == 0.0)


@pytest.mark.datatest
@pytest.mark.parametrize("experiment_description, date", _DATES)
def test_surface_global_single_step(
    *,
    data_provider: sb.IconSerialDataProvider,
    icon_grid: icon_grid_.IconGrid,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend | None,
    date: str,
) -> None:
    config = experiment.config.surface
    assert config is not None
    step = surface_global.SurfaceGlobal(
        grid=icon_grid,
        config=config,
        allocator=model_backends.get_allocator(backend),
        backend=backend,
    )
    fluxes = step.run(input_state=_input_state(data_provider, icon_grid, date))
    _assert_matches_savepoint(fluxes, data_provider.from_savepoint_tmx_surface_fluxes(date=date))


@pytest.mark.datatest
@pytest.mark.parametrize("experiment_description, date", _DATES[1:2])
@pytest.mark.parametrize("latent_heat_flux", [None, 2.0e-5])
def test_surface_global_matches_granule(
    *,
    data_provider: sb.IconSerialDataProvider,
    icon_grid: icon_grid_.IconGrid,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend | None,
    date: str,
    latent_heat_flux: float | None,
) -> None:
    if backend is None:
        pytest.skip("the granule is the witness on a compiled backend.")
    config = experiment.config.surface
    assert config is not None
    if latent_heat_flux is not None:
        config = dataclasses.replace(config, kinematic_latent_heat_flux=latent_heat_flux)
    allocator = model_backends.get_allocator(backend)
    input_state = _input_state(data_provider, icon_grid, date)

    expected = SurfaceFluxState.allocate(icon_grid, allocator=allocator)
    surface.Surface(grid=icon_grid, config=config, backend=backend).run(
        input_state=input_state, out=expected
    )
    actual = surface_global.SurfaceGlobal(grid=icon_grid, config=config, allocator=None).run(
        input_state=input_state
    )

    for field in dataclasses.fields(SurfaceFluxState):
        a = getattr(actual, field.name).asnumpy()
        e = getattr(expected, field.name).asnumpy()
        print(
            f"{field.name}: max abs diff {np.max(np.abs(a - e)):.3e}, max abs {np.max(np.abs(e)):.3e}"
        )
        assert test_utils.dallclose(a, e, rtol=1e-14, atol=0.0), field.name
    if latent_heat_flux is not None:
        assert np.all(np.abs(actual.vapor_energy_flux.asnumpy()) > 0.0)


@pytest.mark.datatest
@pytest.mark.parametrize("experiment_description, date", _DATES[1:2])
def test_surface_global_single_step_jax_jit(
    *,
    data_provider: sb.IconSerialDataProvider,
    icon_grid: icon_grid_.IconGrid,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend | None,
    date: str,
) -> None:
    if backend is not None:
        pytest.skip("jax fields run on the embedded backend only.")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    jnp = jax.numpy
    config = experiment.config.surface
    assert config is not None
    step = surface_global.SurfaceGlobal(grid=icon_grid, config=config, allocator=jnp)
    input_state = _input_state(data_provider, icon_grid, date)

    @jax.jit
    def run(surface_pressure: gtx.Field, temperature: gtx.Field) -> dict[str, gtx.Field]:
        fluxes = step.run(
            input_state=surface_states.SurfaceInputState(
                surface_pressure=surface_pressure, temperature=temperature
            )
        )
        return {f.name: getattr(fluxes, f.name) for f in dataclasses.fields(fluxes)}

    def to_jax(field: gtx.Field) -> gtx.Field:
        return gtx.as_field(field.domain, jnp.asarray(field.asnumpy()), allocator=jnp)

    fluxes = run(to_jax(input_state.surface_pressure), to_jax(input_state.temperature))

    assert isinstance(fluxes["sensible_heat_flux"].ndarray, jax.Array)
    _assert_matches_savepoint(
        SurfaceFluxState(**fluxes), data_provider.from_savepoint_tmx_surface_fluxes(date=date)
    )
