# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""Reverse-mode AD through two `DiffusionGlobal` steps, one run on JAX arrays and one on torch tensors."""

from __future__ import annotations

import contextlib
import functools
import dataclasses
from typing import Any

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.diffusion import diffusion_global
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.common.states import prognostic_state as prognostics
from icon4py.model.testing.fixtures.benchmark import (
    geometry_field_source,
    interpolation_field_source,
    metrics_field_source,
)
from icon4py.model.testing.fixtures.datatest import backend_like
from icon4py.model.testing.fixtures.stencil_tests import grid_manager

from .test_benchmark_diffusion import _setup, _to_jax
from .test_diffusion_global import _to_torch


jax = pytest.importorskip("jax")
torch = pytest.importorskip("torch")
jax.config.update("jax_enable_x64", True)
jnp = jax.numpy
from jax.experimental import buffer_callback  # noqa: E402 [module-import-not-at-top-of-file]


_FIELDS = ("vn", "w", "theta_v", "exner", "rho")


def _torch_in_jax(fn: Any, out_like: Any) -> Any:
    """
    `fn` on torch tensors as a function of JAX arrays that `jax.grad` and `jax.jit` see through.

    The tensors are the XLA buffers themselves (DLPack, in a `buffer_callback`). The backward
    pass runs `fn` again: JAX residuals must be arrays, so the torch graph cannot be kept.
    """

    def stream(ctx: Any, buf: Any) -> Any:
        on_cuda = buf.__dlpack_device__()[0] == 2  # kDLCUDA
        return (
            torch.cuda.stream(torch.cuda.ExternalStream(ctx.stream))
            if on_cuda
            else contextlib.nullcontext()
        )

    def fwd_cb(ctx: Any, out: Any, *xs: Any) -> None:
        with stream(ctx, out):
            torch.from_dlpack(out).copy_(fn(*(torch.from_dlpack(x) for x in xs)))

    def bwd_cb(ctx: Any, outs: Any, ct: Any, *xs: Any) -> None:
        with stream(ctx, ct), torch.enable_grad():
            ts = [torch.from_dlpack(x).detach().requires_grad_(True) for x in xs]
            grads = torch.autograd.grad(fn(*ts), ts, torch.from_dlpack(ct), allow_unused=True)
            for o, g in zip(outs, grads):
                if g is None:
                    torch.from_dlpack(o).zero_()
                else:
                    torch.from_dlpack(o).copy_(g)

    spec = lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype)  # noqa: E731
    call = buffer_callback.buffer_callback(fwd_cb, spec(out_like))

    @jax.custom_vjp
    def f(*xs: Any) -> Any:
        return call(*xs)

    def bwd(xs: tuple, ct: Any) -> tuple:
        return buffer_callback.buffer_callback(bwd_cb, tuple(spec(x) for x in xs))(ct, *xs)

    f.defvjp(lambda *xs: (call(*xs), xs), bwd)
    return f


def _jax_in_torch(fn: Any) -> Any:
    """`fn` on JAX arrays as a torch autograd function; the tensors cross by DLPack."""
    fwd = jax.jit(fn)
    vjp = jax.jit(lambda xs, ct: jax.vjp(fn, *xs)[1](ct))
    to_jax = lambda t: jnp.from_dlpack(t.detach().contiguous())  # noqa: E731

    class Fn(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, *ts: Any) -> Any:
            ctx.xs = tuple(to_jax(t) for t in ts)
            return torch.from_dlpack(fwd(*ctx.xs))

        @staticmethod
        def backward(ctx: Any, ct: Any) -> tuple:
            return tuple(torch.from_dlpack(g) for g in vjp(ctx.xs, to_jax(ct)))

    return Fn.apply


def _granule(
    setup: dict[str, Any], convert: Any, allocator: Any
) -> diffusion_global.DiffusionGlobal:
    mesh = setup["mesh"]
    vertical_grid = setup["vertical_grid"]
    return diffusion_global.DiffusionGlobal(
        grid=dataclasses.replace(
            mesh, connectivities={k: convert(v) for k, v in mesh.connectivities.items()}
        ),
        config=setup["config"],
        params=setup["diffusion_parameters"],
        vertical_grid=v_grid.VerticalGrid(
            config=vertical_grid.config,
            vct_a=convert(vertical_grid.vct_a),
            vct_b=convert(vertical_grid.vct_b),
        ),
        metric_state=convert(setup["metric_state"]),
        interpolation_state=convert(setup["interpolation_state"]),
        edge_params=convert(setup["edge_geometry"]),
        cell_params=convert(setup["cell_geometry"]),
        allocator=allocator,
        ndyn_substeps=5,
    )


def _step(
    granule: Any, domains: dict[str, Any], dtime: float, allocator: Any, wrap: Any = None
) -> Any:
    """One diffusion step as a function of the raw prognostic arrays, returning all but rho."""

    def run(*fields: Any) -> tuple:
        new = granule.run(
            prognostic_state=prognostics.PrognosticState(**dict(zip(_FIELDS, fields))), dtime=dtime
        )
        return tuple(getattr(new, k).ndarray for k in _FIELDS[:-1])

    run = wrap(run) if wrap else run

    def step(*arrays: Any) -> tuple:
        return run(
            *(gtx.as_field(domains[k], a, allocator=allocator) for k, a in zip(_FIELDS, arrays))
        )

    return step


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_diffusion_global_jax_torch_ad(  # noqa: PLR0917 [too-many-positional-arguments]
    geometry_field_source: Any,
    grid_manager: Any,
    interpolation_field_source: Any,
    metrics_field_source: Any,
    backend_like: Any,
    device: str,
) -> None:
    if device == "cuda" and not (torch.cuda.is_available() and jax.devices()[0].platform == "gpu"):
        pytest.skip("no CUDA device for both JAX and torch.")
    setup = _setup(
        geometry_field_source,
        grid_manager,
        interpolation_field_source,
        metrics_field_source,
        backend_like,
    )
    if setup["mesh"].limited_area:
        pytest.skip("'DiffusionGlobal' does not support limited area grids.")
    with (
        jax.default_device(jax.devices("cpu" if device == "cpu" else "gpu")[0]),
        torch.device(device),
    ):
        _check_gradients(setup)


def _check_gradients(setup: dict[str, Any]) -> None:
    dtime = setup["dtime"]
    domains = {k: getattr(setup["prognostic_state"], k).domain for k in _FIELDS}
    x0 = tuple(jnp.asarray(getattr(setup["prognostic_state"], k).asnumpy()) for k in _FIELDS)
    ct_vn = jnp.asarray(np.random.default_rng(0).standard_normal(x0[0].shape))
    print(f"JAX arrays on {ct_vn.device}, torch tensors on {torch.from_dlpack(ct_vn).device}")

    step_jax = _step(_granule(setup, lambda v: _to_jax(v, jnp), jnp), domains, dtime, jnp)
    granule_torch = _granule(setup, lambda v: _to_torch(v, torch), torch)
    step_torch = _step(granule_torch, domains, dtime, torch)

    cost_jax = lambda vn: jnp.sum(vn * ct_vn)  # noqa: E731

    def all_jax(vn: Any) -> Any:
        return cost_jax(step_jax(*step_jax(vn, *x0[1:]), x0[-1])[0])

    grad_ref = np.asarray(jax.jit(jax.grad(all_jax))(x0[0]))
    assert np.max(np.abs(grad_ref)) > 0
    direction = jnp.asarray(np.random.default_rng(1).standard_normal(x0[0].shape))
    eps = 1e-4
    fd = (jax.jit(all_jax)(x0[0] + eps * direction) - jax.jit(all_jax)(x0[0] - eps * direction)) / (
        2 * eps
    )
    fd_err = abs(float(fd) - float(np.sum(grad_ref * np.asarray(direction)))) / abs(float(fd))
    print(f"all-JAX gradient vs finite differences: rel. err {fd_err:.1e}")
    assert fd_err < 1e-6
    # diffusion is close to the identity: the gradient must differ from the cost's own
    diffusion_part = np.max(np.abs(grad_ref - np.asarray(ct_vn))) / np.max(np.abs(grad_ref))
    print(f"diffusion part of the gradient: {diffusion_part:.1e}")
    assert diffusion_part > 1e-6

    def check(name: str, grad: Any) -> None:
        err = np.max(np.abs(np.asarray(grad) - grad_ref)) / np.max(np.abs(grad_ref))
        print(f"{name}: max rel. err vs all-JAX gradient {err:.1e}")
        assert err < 1e-12, name

    # JAX drives: step 1 in JAX, step 2 in torch inside jax.jit
    torch_vn = lambda *xs: step_torch(*xs)[0]  # noqa: E731
    step2 = _torch_in_jax(torch_vn, x0[0])

    def jax_then_torch(vn: Any) -> Any:
        out1 = step_jax(vn, *x0[1:])
        return cost_jax(step2(*out1, x0[-1]))

    check("jax.grad(JAX step -> torch step)", jax.grad(jax_then_torch)(x0[0]))
    check("jax.jit(jax.grad(JAX step -> torch step))", jax.jit(jax.grad(jax_then_torch))(x0[0]))

    # torch drives: step 1 in torch, step 2 in JAX inside torch autograd
    jax_vn = _jax_in_torch(lambda *xs: step_jax(*xs)[0])
    rest = tuple(torch.from_dlpack(x).clone() for x in x0[1:])
    ct_t = torch.from_dlpack(ct_vn).clone()
    for name, step1 in (
        ("", step_torch),
        (
            ", torch.compile",
            _step(
                granule_torch,
                domains,
                dtime,
                torch,
                functools.partial(torch.compile, fullgraph=True),
            ),
        ),
    ):
        vn = torch.from_dlpack(x0[0]).clone().requires_grad_(True)
        out1 = step1(vn, *rest)
        (jax_vn(*out1, rest[-1]) * ct_t).sum().backward()
        assert vn.grad.device == ct_t.device
        check(f"torch backward(torch step -> JAX step{name})", vn.grad.cpu().numpy())
