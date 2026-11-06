# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import datetime

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys import (
    component as muphys_component,
    config as muphys_config,
    muphys_global,
)
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.definitions import SPECIES, Q
from icon4py.model.common.states.data import QC, QG, QI, QR, QS, QV
from icon4py.model.testing import definitions, test_utils

from ..fixtures import *  # noqa: F403


_DATES = definitions.Experiments.EXCLAIM_APE_AES.dates
_TRACER_INDEX = {"v": QV, "c": QC, "r": QR, "s": QS, "i": QI, "g": QG}
# the tolerances of test_muphys_datatest.test_muphys_granule in double precision
_TRACER_ATOL = 1e-13
_TEMPERATURE_ATOL = 1e-10


def _inputs(sp) -> dict[str, gtx.Field]:
    return {
        "dz": sp.dz(),
        "te": sp.temperature(),
        "p": sp.pressure(),
        "rho": sp.rho(),
        **{f"q{s}": getattr(sp, f"q{s}")() for s in SPECIES},
    }


def _run(step, fields, dtime):
    return step.run(
        dz=fields["dz"],
        te=fields["te"],
        p=fields["p"],
        rho=fields["rho"],
        q=Q(**{s: fields[f"q{s}"] for s in SPECIES}),
        dtime=dtime,
    )


def _check_against_savepoints(outputs, inputs, dtime, init_savepoint, exit_savepoint):
    te, q, _, pr, ps, pi, pg, pre = outputs
    # ICON runs the scheme from jks_cloudy (1-based) down only
    jks = init_savepoint.jks_cloudy() - 1
    for s in SPECIES:
        new = getattr(q, s).asnumpy()
        test_utils.assert_dallclose(
            new[:, jks:],
            exit_savepoint.tracer(_TRACER_INDEX[s]).asnumpy()[:, jks:],
            atol=_TRACER_ATOL,
            err_msg=f"q{s} in cloud",
        )
        test_utils.assert_dallclose(
            (new[:, :jks] - inputs[f"q{s}"].asnumpy()[:, :jks]) / dtime,
            0.0,
            atol=_TRACER_ATOL / dtime,
            err_msg=f"tend_q{s} above cloud",
        )
    test_utils.assert_dallclose(
        te.asnumpy()[:, jks:],
        exit_savepoint.temperature().asnumpy()[:, jks:],
        atol=_TEMPERATURE_ATOL,
        err_msg="temperature in cloud",
    )
    test_utils.assert_dallclose(
        (te.asnumpy()[:, :jks] - inputs["te"].asnumpy()[:, :jks]) / dtime,
        0.0,
        atol=_TEMPERATURE_ATOL / dtime,
        err_msg="tend_temperature above cloud",
    )

    rain, ice, snow, graupel = (x.asnumpy()[:, -1] for x in (pr, pi, ps, pg))
    test_utils.assert_dallclose(rain, exit_savepoint.rsfl().asnumpy(), atol=1e-10)
    test_utils.assert_dallclose(ice + snow + graupel, exit_savepoint.ssfl().asnumpy(), atol=1e-10)
    test_utils.assert_dallclose(
        rain + ice + snow + graupel, exit_savepoint.pr().asnumpy(), atol=1e-10
    )
    test_utils.assert_dallclose(pre.asnumpy()[:, -1], exit_savepoint.ufcs().asnumpy(), rtol=1e-12)


@pytest.mark.datatest
@pytest.mark.parametrize("experiment_description", [definitions.Experiments.EXCLAIM_APE_AES])
@pytest.mark.parametrize("date", _DATES)
def test_muphys_global(date, *, experiment, data_provider, icon_grid, backend):
    if backend is not None:
        pytest.skip("the global step is verified on the embedded backend.")
    init_savepoint = data_provider.from_savepoint_muphys_init(date=date)
    exit_savepoint = data_provider.from_savepoint_muphys_exit(date=date)
    inputs = _inputs(init_savepoint)
    dtime = experiment.config.driver.dtime.total_seconds()

    step = muphys_global.MuphysGlobal(grid=icon_grid, qnc=muphys_config.MuphysConfig().qnc)
    outputs = _run(step, inputs, dtime)

    for name, value in _inputs(init_savepoint).items():
        np.testing.assert_array_equal(inputs[name].asnumpy(), value.asnumpy())
    _check_against_savepoints(outputs, inputs, dtime, init_savepoint, exit_savepoint)


@pytest.mark.datatest
@pytest.mark.parametrize("experiment_description", [definitions.Experiments.EXCLAIM_APE_AES])
@pytest.mark.parametrize("date", _DATES)
def test_muphys_global_matches_granule(date, *, experiment, data_provider, icon_grid, backend):
    if backend is None:
        pytest.skip("the granule is the witness on a compiled backend.")
    init_savepoint = data_provider.from_savepoint_muphys_init(date=date)
    dtime = experiment.config.driver.dtime.total_seconds()
    qnc = muphys_config.MuphysConfig().qnc

    granule = muphys_component.MuphysComponent(
        grid=icon_grid, dtime=datetime.timedelta(seconds=dtime), qnc=qnc, backend=backend
    )
    granule_out = granule(_inputs(init_savepoint), datetime.datetime.fromisoformat(date))

    te, q, pflx, *surface = _run(
        muphys_global.MuphysGlobal(grid=icon_grid, qnc=qnc), _inputs(init_savepoint), dtime
    )

    def compare(name, actual, expected):
        diff = np.max(np.abs(actual - expected))
        print(f"{name}: max abs diff {diff:.3e}, max abs {np.max(np.abs(expected)):.3e}")
        test_utils.assert_dallclose(actual, expected, rtol=1e-12, atol=1e-16, err_msg=name)

    # the granule's in-place buffers hold its new state
    compare("te", te.asnumpy(), granule._te_in.asnumpy())
    for s in SPECIES:
        compare(f"q{s}", getattr(q, s).asnumpy(), getattr(granule._q_in, s).asnumpy())
    compare("pflx", pflx.asnumpy(), granule_out["pflx"].asnumpy())
    for name, value in zip(("pr", "ps", "pi", "pg", "pre"), surface, strict=True):
        compare(name, value.asnumpy()[:, -1], granule_out[name].asnumpy()[:, -1])


@pytest.mark.datatest
@pytest.mark.parametrize("experiment_description", [definitions.Experiments.EXCLAIM_APE_AES])
@pytest.mark.parametrize("date", _DATES[1:2])
def test_muphys_global_jax_jit(date, *, experiment, data_provider, icon_grid, backend):
    if backend is not None:
        pytest.skip("jax fields run on the embedded backend only.")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    jnp = jax.numpy

    init_savepoint = data_provider.from_savepoint_muphys_init(date=date)
    exit_savepoint = data_provider.from_savepoint_muphys_exit(date=date)
    inputs = _inputs(init_savepoint)
    dtime = experiment.config.driver.dtime.total_seconds()
    step = muphys_global.MuphysGlobal(grid=icon_grid, qnc=muphys_config.MuphysConfig().qnc)

    outputs = jax.jit(lambda fields: _run(step, fields, dtime))(
        {
            name: gtx.as_field(f.domain, jnp.asarray(f.asnumpy()), allocator=jnp)
            for name, f in inputs.items()
        }
    )

    assert isinstance(outputs[0].ndarray, jax.Array)
    _check_against_savepoints(outputs, inputs, dtime, init_savepoint, exit_savepoint)


@pytest.mark.datatest
@pytest.mark.parametrize("experiment_description", [definitions.Experiments.EXCLAIM_APE_AES])
@pytest.mark.parametrize("date", _DATES[1:2])
@pytest.mark.parametrize("compiled", [False, True])
def test_muphys_global_torch(date, compiled, *, experiment, data_provider, icon_grid, backend):
    if backend is not None:
        pytest.skip("torch fields run on the embedded backend only.")
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    jnp = jax.numpy

    init_savepoint = data_provider.from_savepoint_muphys_init(date=date)
    exit_savepoint = data_provider.from_savepoint_muphys_exit(date=date)
    inputs = _inputs(init_savepoint)
    dtime = experiment.config.driver.dtime.total_seconds()
    step = muphys_global.MuphysGlobal(grid=icon_grid, qnc=muphys_config.MuphysConfig().qnc)

    def run(fields):
        return _run(step, fields, dtime)

    torch_inputs = {
        name: gtx.as_field(f.domain, torch.asarray(f.asnumpy()), allocator=torch)
        for name, f in inputs.items()
    }
    eager = run(torch_inputs)
    if compiled:
        run = torch.compile(run, fullgraph=True)
    outputs = run(torch_inputs)
    jax_outputs = jax.jit(lambda fields: _run(step, fields, dtime))(
        {
            name: gtx.as_field(f.domain, jnp.asarray(f.asnumpy()), allocator=jnp)
            for name, f in inputs.items()
        }
    )

    assert isinstance(outputs[0].ndarray, torch.Tensor)
    _check_against_savepoints(outputs, inputs, dtime, init_savepoint, exit_savepoint)
    flat, eager_flat, jax_flat = (
        [out[0], *out[1], *out[2:]] for out in (outputs, eager, jax_outputs)
    )
    for label, references in (("eager", eager_flat), ("jax", jax_flat)):
        for torch_field, reference in zip(flat, references, strict=True):
            actual, expected = torch_field.asnumpy(), reference.asnumpy()
            scale = np.max(np.abs(expected))
            diff = np.max(np.abs(actual - expected))
            print(f"max |torch - {label}| {diff:.3e}, max abs {scale:.3e}")
            test_utils.assert_dallclose(actual, expected, rtol=0.0, atol=1e-10 * scale)
