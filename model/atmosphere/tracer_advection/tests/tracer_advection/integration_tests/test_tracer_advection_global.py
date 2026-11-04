# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import dataclasses
from typing import Any

import gt4py.next as gtx
import numpy as np
import pytest
from gt4py.next import common as gtx_common

from icon4py.model.atmosphere.tracer_advection import (
    tracer_advection,
    tracer_advection_global,
    tracer_advection_states,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.states import tracer_prep_adv_states as prep_adv_states
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import definitions as test_defs, test_utils
from icon4py.model.testing.fixtures.datatest import (
    backend,
    backend_like,
    data_provider,
    download_ser_data,
    experiment,
    experiment_description,
    grid_savepoint,
    icon_grid,
    interpolation_savepoint,
    metrics_savepoint,
    process_props,
)

from ..fixtures import advection_exit_savepoint, advection_init_savepoint
from ..utils import construct_interpolation_state, construct_metric_state, construct_prep_adv


# Every tracer listed is advected with 2nd order Miura, the positive definite horizontal
# limiter, PPM and the semi-monotonic vertical limiter (ihadv_tracer=2, itype_hlimit=4,
# ivadv_tracer=3, itype_vlimit=1 in the experiments' transport_nml). The splitting order is
# the one ICON used for the step.
_CONFIG = tracer_advection.AdvectionConfig(
    horizontal_advection_type=tracer_advection.HorizontalAdvectionType.SECOND_ORDER_LINEAR_MIURA,
    horizontal_advection_limiter=tracer_advection.HorizontalAdvectionLimiter.POSITIVE_DEFINITE,
    vertical_advection_type=tracer_advection.VerticalAdvectionType.THIRD_ORDER_PPM,
    vertical_advection_limiter=tracer_advection.VerticalAdvectionLimiter.SEMI_MONOTONIC,
)

_STEPS = [
    (test_defs.Experiments.WEISMAN_KLEMP_TORUS, "2008-09-01T01:59:48.000", False, 0),
    (test_defs.Experiments.WEISMAN_KLEMP_TORUS, "2008-09-01T01:59:52.000", True, 1),
    (test_defs.Experiments.WEISMAN_KLEMP_TORUS, "2008-09-01T01:59:56.000", False, 3),
    (test_defs.Experiments.WEISMAN_KLEMP_TORUS, "2008-09-01T02:00:00.000", True, 5),
    (test_defs.Experiments.EXCLAIM_APE, "2000-01-01T00:00:04.000", True, 2),
]


def _setup(icon_grid, grid_savepoint, interpolation_savepoint, metrics_savepoint, backend):
    least_squares_state = tracer_advection_states.AdvectionLeastSquaresState(
        lsq_pseudoinv_1=interpolation_savepoint.lsq_pseudoinv_1(),
        lsq_pseudoinv_2=interpolation_savepoint.lsq_pseudoinv_2(),
    )
    return {
        "grid": icon_grid,
        "config": _CONFIG,
        "interpolation_state": construct_interpolation_state(
            interpolation_savepoint, backend=backend
        ),
        "least_squares_state": least_squares_state,
        "metric_state": construct_metric_state(icon_grid, metrics_savepoint, backend=backend),
        "edge_params": grid_savepoint.construct_edge_geometry(),
    }


@pytest.mark.datatest
@pytest.mark.parametrize("experiment_description, date, even_timestep, ntracer", _STEPS)
def test_tracer_advection_global_single_step(
    date,
    even_timestep,
    ntracer,
    *,
    grid_savepoint,
    icon_grid,
    interpolation_savepoint,
    metrics_savepoint,
    backend,
    advection_init_savepoint,
    advection_exit_savepoint,
):
    granule = tracer_advection_global.TracerAdvectionGlobal(
        **_setup(icon_grid, grid_savepoint, interpolation_savepoint, metrics_savepoint, backend),
        allocator=backend,
        backend=backend,
    )
    p_tracer_now = advection_init_savepoint.tracer(ntracer)
    dtime = advection_init_savepoint.get_metadata("dtime").get("dtime")

    p_tracer_new, hfl_tracer, vfl_tracer = granule.run(
        p_tracer_now=p_tracer_now,
        airmass_now=advection_init_savepoint.airmass_now(),
        airmass_new=advection_init_savepoint.airmass_new(),
        prep_adv=construct_prep_adv(advection_init_savepoint),
        dtime=dtime,
        even_timestep=even_timestep,
    )

    assert test_utils.dallclose(
        p_tracer_now.asnumpy(), advection_init_savepoint.tracer(ntracer).asnumpy()
    )
    tracer_ref = advection_exit_savepoint.tracer(ntracer).asnumpy()
    assert not np.allclose(p_tracer_now.asnumpy(), tracer_ref, rtol=0.0, atol=0.0)
    assert test_utils.dallclose(
        hfl_tracer.asnumpy(),
        advection_exit_savepoint.hfl_tracer(ntracer).asnumpy(),
        rtol=1e-10,
        atol=1e-11,
    )
    assert test_utils.dallclose(
        vfl_tracer.asnumpy(),
        advection_exit_savepoint.vfl_tracer(ntracer).asnumpy(),
        rtol=1e-10,
        atol=1e-16,
    )
    assert test_utils.dallclose(p_tracer_new.asnumpy(), tracer_ref, atol=1e-16)


@pytest.mark.datatest
@pytest.mark.parametrize("experiment_description, date, even_timestep, ntracer", _STEPS)
@pytest.mark.parametrize("granule_even_timestep", [False, True])
@pytest.mark.parametrize("max_vertical_courant_number", [None, 3.5])
def test_tracer_advection_global_matches_granule(
    date,
    even_timestep,
    ntracer,
    granule_even_timestep,
    max_vertical_courant_number,
    *,
    grid_savepoint,
    icon_grid,
    interpolation_savepoint,
    metrics_savepoint,
    backend,
    advection_init_savepoint,
):
    if backend is None:
        pytest.skip("the granule is the witness on a compiled backend.")
    setup = _setup(icon_grid, grid_savepoint, interpolation_savepoint, metrics_savepoint, backend)
    granule = tracer_advection.convert_config_to_advection(
        **setup,
        cell_params=grid_savepoint.construct_cell_geometry(),
        backend=backend,
        exchange=decomposition.SingleNodeExchange(),
        even_timestep=granule_even_timestep,
    )
    diagnostic_state = tracer_advection_states.AdvectionDiagnosticState(
        airmass_now=advection_init_savepoint.airmass_now(),
        airmass_new=advection_init_savepoint.airmass_new(),
        grf_tend_tracer=advection_init_savepoint.grf_tend_tracer(ntracer),
        hfl_tracer=data_alloc.zero_field(icon_grid, dims.EdgeDim, dims.KDim, allocator=backend),
        vfl_tracer=data_alloc.zero_field(icon_grid, dims.CellDim, dims.KHalfDim, allocator=backend),
    )
    p_tracer_now = advection_init_savepoint.tracer(ntracer)
    p_tracer_new = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, allocator=backend)
    dtime = advection_init_savepoint.get_metadata("dtime").get("dtime")
    prep_adv = construct_prep_adv(advection_init_savepoint)
    if max_vertical_courant_number is not None:
        # Scale the vertical mass flux so that the PPM flux takes whole cells (|CFL| > 1).
        mass_flx_ic = prep_adv.mass_flx_ic.asnumpy()
        airmass = diagnostic_state.airmass_now.asnumpy()
        courant = dtime * np.abs(mass_flx_ic[:, 1:-1]) / np.minimum(airmass[:, 1:], airmass[:, :-1])
        factor = max_vertical_courant_number / np.max(courant)
        prep_adv = dataclasses.replace(
            prep_adv,
            mass_flx_ic=gtx.as_field(
                prep_adv.mass_flx_ic.domain, factor * mass_flx_ic, allocator=backend
            ),
        )
    granule.run(
        diagnostic_state=diagnostic_state,
        prep_adv=prep_adv,
        p_tracer_now=p_tracer_now,
        p_tracer_new=p_tracer_new,
        dtime=dtime,
    )

    global_step = tracer_advection_global.TracerAdvectionGlobal(**setup, allocator=None)
    tracer, hfl, vfl = global_step.run(
        p_tracer_now=p_tracer_now,
        airmass_now=diagnostic_state.airmass_now,
        airmass_new=diagnostic_state.airmass_new,
        prep_adv=prep_adv,
        dtime=dtime,
        even_timestep=granule_even_timestep,
    )

    for name, actual, expected in (
        ("tracer", tracer, p_tracer_new),
        ("hfl", hfl, diagnostic_state.hfl_tracer),
        ("vfl", vfl, diagnostic_state.vfl_tracer),
    ):
        diff = np.max(np.abs(actual.asnumpy() - expected.asnumpy()))
        print(f"{name}: max abs diff {diff:.3e}, max abs {np.max(np.abs(expected.asnumpy())):.3e}")
        assert test_utils.dallclose(actual.asnumpy(), expected.asnumpy(), rtol=1e-12, atol=1e-16)


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


@pytest.mark.datatest
@pytest.mark.parametrize("experiment_description, date, even_timestep, ntracer", _STEPS[:2])
def test_tracer_advection_global_single_step_jax_jit(
    date,
    even_timestep,
    ntracer,
    *,
    grid_savepoint,
    icon_grid,
    interpolation_savepoint,
    metrics_savepoint,
    backend,
    advection_init_savepoint,
    advection_exit_savepoint,
):
    if backend is not None:
        pytest.skip("jax fields run on the embedded backend only.")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    jnp = jax.numpy

    setup = _setup(icon_grid, grid_savepoint, interpolation_savepoint, metrics_savepoint, None)
    granule = tracer_advection_global.TracerAdvectionGlobal(
        **{name: _to_jax(value, jnp) for name, value in setup.items() if name != "grid"},
        grid=dataclasses.replace(
            icon_grid,
            connectivities={k: _to_jax(v, jnp) for k, v in icon_grid.connectivities.items()},
        ),
        allocator=jnp,
    )
    dtime = advection_init_savepoint.get_metadata("dtime").get("dtime")

    @jax.jit
    def step(state: dict[str, gtx.Field]) -> tuple[gtx.Field, gtx.Field, gtx.Field]:
        return granule.run(
            p_tracer_now=state["tracer"],
            airmass_now=state["airmass_now"],
            airmass_new=state["airmass_new"],
            prep_adv=prep_adv_states.TracerPrepAdvState(
                vn_traj=state["vn_traj"],
                mass_flx_me=state["mass_flx_me"],
                mass_flx_ic=state["mass_flx_ic"],
            ),
            dtime=dtime,
            even_timestep=even_timestep,
        )

    sp = advection_init_savepoint
    p_tracer_new, hfl_tracer, vfl_tracer = step(
        {
            "tracer": _to_jax(sp.tracer(ntracer), jnp),
            "airmass_now": _to_jax(sp.airmass_now(), jnp),
            "airmass_new": _to_jax(sp.airmass_new(), jnp),
            "vn_traj": _to_jax(sp.vn_traj(), jnp),
            "mass_flx_me": _to_jax(sp.mass_flx_me(), jnp),
            "mass_flx_ic": _to_jax(sp.mass_flx_ic(), jnp),
        }
    )

    assert isinstance(p_tracer_new.ndarray, jax.Array)
    assert test_utils.dallclose(
        hfl_tracer.asnumpy(),
        advection_exit_savepoint.hfl_tracer(ntracer).asnumpy(),
        rtol=1e-10,
        atol=1e-11,
    )
    assert test_utils.dallclose(
        vfl_tracer.asnumpy(),
        advection_exit_savepoint.vfl_tracer(ntracer).asnumpy(),
        rtol=1e-10,
        atol=1e-16,
    )
    assert test_utils.dallclose(
        p_tracer_new.asnumpy(), advection_exit_savepoint.tracer(ntracer).asnumpy(), atol=1e-16
    )
