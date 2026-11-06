# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""
The stateless global steps under `jax.jit` against their granules on `--backend`.

The inputs are seeded synthetic profiles, not a model state, so that two processes (one with the
JAX branch, one with icon4py main) see the same inputs: `ICON4PY_BENCH_DUMP_DIR` holds the JAX
outputs written by `test_global_jax_matches_granule`, which `test_granule_matches_dump` reads.
The component configuration is the `config.yml` at `ICON4PY_BENCH_CONFIG`.

This module imports nothing that only exists on the JAX branch at module level.
"""

# ruff: noqa: PLC0415  # the JAX-branch modules must not be imported on icon4py main
# one `step` variable per component branch; benchmark-only code
# mypy: ignore-errors

from __future__ import annotations

import dataclasses
import datetime
import math
import os
import pathlib
import time
from typing import Any

import gt4py.next as gtx
import numpy as np
import pytest
from gt4py.next import common as gtx_common

import icon4py.model.common.dimension as dims
import icon4py.model.common.grid.states as grid_states
from icon4py.model.common import model_backends
from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.common.grid import geometry_attributes as geometry_meta
from icon4py.model.common.interpolation import interpolation_attributes
from icon4py.model.common.metrics import metrics_attributes
from icon4py.model.common.utils import data_allocation as data_alloc, device_utils
from icon4py.model.driver import config as driver_config
from icon4py.model.testing import torch_steps
from icon4py.model.testing.fixtures.benchmark import (
    geometry_field_source,
    interpolation_field_source,
    metrics_field_source,
)
from icon4py.model.testing.fixtures.datatest import backend_like
from icon4py.model.testing.fixtures.stencil_tests import grid_manager


_SEED = 20261104
_SPECIES = ("v", "c", "i", "r", "s", "g")
COMPONENTS = ("tracer", "tmx", "surface", "muphys")
TORCH_COMPONENTS = ("tmx", "muphys")


def _config() -> driver_config.ExperimentConfig:
    path = os.environ.get("ICON4PY_BENCH_CONFIG")
    if path is None:
        pytest.skip("needs ICON4PY_BENCH_CONFIG (an APE_AES config.yml)")
    return driver_config.read_experiment_config_from_yaml(pathlib.Path(path))


def _dtime(config: driver_config.ExperimentConfig, num_cells: int) -> float:
    """The experiment's R2B4 time step, halved per bisection."""
    bisections = round(math.log(num_cells / 80, 4))
    return config.driver.dtime.total_seconds() / 2 ** (bisections - 4)


def _edge_geometry(geometry: Any) -> grid_states.EdgeParams:
    g = geometry.get
    return grid_states.EdgeParams(
        tangent_orientation=g(geometry_meta.TANGENT_ORIENTATION),
        inverse_primal_edge_lengths=g(f"inverse_of_{geometry_meta.EDGE_LENGTH}"),
        inverse_dual_edge_lengths=g(f"inverse_of_{geometry_meta.DUAL_EDGE_LENGTH}"),
        inverse_vertex_vertex_lengths=g(f"inverse_of_{geometry_meta.VERTEX_VERTEX_LENGTH}"),
        primal_normal_vert=(
            g(geometry_meta.EDGE_NORMAL_VERTEX_U),
            g(geometry_meta.EDGE_NORMAL_VERTEX_V),
        ),
        dual_normal_vert=(
            g(geometry_meta.EDGE_TANGENT_VERTEX_U),
            g(geometry_meta.EDGE_TANGENT_VERTEX_V),
        ),
        primal_normal_cell=(
            g(geometry_meta.EDGE_NORMAL_CELL_U),
            g(geometry_meta.EDGE_NORMAL_CELL_V),
        ),
        dual_normal_cell=(
            g(geometry_meta.EDGE_TANGENT_CELL_U),
            g(geometry_meta.EDGE_TANGENT_CELL_V),
        ),
        edge_areas=g(geometry_meta.EDGE_AREA),
        coriolis_frequency=g(geometry_meta.CORIOLIS_PARAMETER),
        edge_center=(g(geometry_meta.EDGE_LAT), g(geometry_meta.EDGE_LON)),
        primal_normal=(g(geometry_meta.EDGE_NORMAL_U), g(geometry_meta.EDGE_NORMAL_V)),
        primal_edge_lengths=g(geometry_meta.EDGE_LENGTH),
        dual_edge_lengths=g(geometry_meta.DUAL_EDGE_LENGTH),
        edge_cell_distances=g(geometry_meta.EDGE_CELL_DISTANCE),
    )


def _cell_geometry(geometry: Any) -> grid_states.CellParams:
    return grid_states.CellParams(
        cell_center_lat=geometry.get(geometry_meta.CELL_LAT),
        cell_center_lon=geometry.get(geometry_meta.CELL_LON),
        area=geometry.get(geometry_meta.CELL_AREA),
    )


def _synthetic_inputs(mesh: Any, metrics: Any, dtime: float) -> dict[str, np.ndarray]:
    """
    Seeded profiles of a moist standard atmosphere plus noise, every field within its physical
    range; the vertical and horizontal Courant numbers stay below 0.5.
    """
    rng = np.random.default_rng(_SEED)
    z = data_alloc.as_numpy(metrics.get(metrics_attributes.Z_MC).ndarray)
    dz = data_alloc.as_numpy(metrics.get(metrics_attributes.DDQZ_Z_FULL).ndarray)
    num_cells, num_levels = z.shape
    num_edges = mesh.num_edges
    noise = lambda *shape: rng.standard_normal(shape)  # noqa: E731
    temperature = np.maximum(288.0 - 0.0065 * np.minimum(z, 11000.0) + 1.0 * noise(*z.shape), 200.0)
    pressure = 101325.0 * np.exp(-z / 8000.0) * (1.0 + 1.0e-3 * noise(*z.shape))
    qv = 0.015 * np.exp(-z / 2500.0) * rng.uniform(0.5, 1.0, z.shape)
    condensate = {
        s: 1.0e-4 * rng.uniform(0.0, 1.0, z.shape) * (rng.uniform(size=z.shape) < 0.3)
        for s in _SPECIES[1:]
    }
    for s in ("i", "s", "g"):
        condensate[s] = np.where(temperature < 273.15, condensate[s], 0.0)
    virtual_temperature = temperature * (1.0 + 0.608 * qv)
    rho = pressure / (287.04 * virtual_temperature)
    w = 0.1 * noise(num_cells, num_levels + 1)
    w[:, 0] = w[:, -1] = 0.0
    airmass = rho * dz
    mass_flx_ic = rho[:, np.r_[0, 0:num_levels]] * w
    courant = dtime * np.abs(mass_flx_ic[:, 1:-1]) / np.minimum(airmass[:, 1:], airmass[:, :-1])
    mass_flx_ic *= min(1.0, 0.5 / courant.max())
    vn = 10.0 * noise(num_edges, num_levels)
    return {
        "temperature": temperature,
        "virtual_temperature": virtual_temperature,
        "pressure": pressure,
        "rho": rho,
        "dz": dz,
        "u": 10.0 + 3.0 * noise(*z.shape),
        "v": 3.0 * noise(*z.shape),
        "w": w,
        "qv": qv,
        **{f"q{s}": condensate[s] for s in _SPECIES[1:]},
        "air_mass": airmass,
        "airmass_new": airmass * (1.0 + 1.0e-4 * noise(*z.shape)),
        "vn_traj": vn,
        "mass_flx_me": vn * rng.uniform(0.9, 1.2, vn.shape),
        "mass_flx_ic": mass_flx_ic,
        "surface_pressure": pressure[:, -1] * 1.005,
        "evapotranspiration": 1.0e-5 * noise(num_cells),
        "u_stress": 0.1 * noise(num_cells),
        "v_stress": 0.1 * noise(num_cells),
        "q_snocpymlt": np.zeros(num_cells),
        "thermal_energy_flux": 50.0 + 10.0 * noise(num_cells),
        "vapor_energy_flux": 10.0 * noise(num_cells),
    }


_DIMS = {
    "w": (dims.CellDim, dims.KHalfDim),
    "mass_flx_ic": (dims.CellDim, dims.KHalfDim),
    "vn_traj": (dims.EdgeDim, dims.KDim),
    "mass_flx_me": (dims.EdgeDim, dims.KDim),
}


def _as_fields(arrays: dict[str, np.ndarray], allocator: Any) -> dict[str, gtx.Field]:
    def dims_of(name: str, array: np.ndarray) -> tuple[gtx.Dimension, ...]:
        if name in _DIMS:
            return _DIMS[name]
        return (dims.CellDim, dims.KDim) if array.ndim == 2 else (dims.CellDim,)

    return {
        name: gtx.as_field(dims_of(name, a), a, allocator=allocator) for name, a in arrays.items()
    }


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


@dataclasses.dataclass
class _Setup:
    mesh: Any
    config: driver_config.ExperimentConfig
    dtime: float
    constants: dict[str, Any]
    inputs: dict[str, np.ndarray]


def _host_factories(grid_manager: Any, backend_like: Any) -> tuple[Any, Any, Any]:
    """The factories of `icon4py.model.testing.fixtures.benchmark`, not cached by pytest."""
    from icon4py.model.common import model_options, topography
    from icon4py.model.common.grid import (
        geometry as grid_geometry,
        geometry_config,
        vertical as v_grid,
    )
    from icon4py.model.common.interpolation import interpolation_factory
    from icon4py.model.common.metrics import metrics_factory
    from icon4py.model.common.topography.analytical import jablonowski_williamson as jw_topo

    mesh = grid_manager.grid
    backend = model_options.customize_backend(None, backend_like)
    allocator = model_backends.get_allocator(backend_like)
    props = decomp_defs.SingleNodeProcessProperties()
    geometry = grid_geometry.GridGeometry(
        grid=mesh,
        decomposition_info=grid_manager.decomposition_info,
        backend=backend,
        coordinates=grid_manager.coordinates,
        extra_fields=grid_manager.geometry_fields,
        metadata=geometry_meta.attrs,
        config=geometry_config.GeometryConfig(),
        process_props=props,
    )
    interpolation = interpolation_factory.InterpolationFieldsFactory(
        config=interpolation_factory.InterpolationConfig(),
        grid=mesh,
        decomposition_info=grid_manager.decomposition_info,
        geometry_source=geometry,
        backend=backend,
        metadata=interpolation_attributes.attrs,
        process_props=props,
    )
    vertical_config = v_grid.VerticalGridConfig(
        mesh.num_levels,
        lowest_layer_thickness=50,
        model_top_height=23500.0,
        stretch_factor=1.0,
        rayleigh_damping_height=1.0,
    )
    vct_a, vct_b = v_grid.get_vct_a_and_vct_b(vertical_config, allocator)
    topo_c = topography.create(
        config=jw_topo.JablonowskiWilliamsonConfig(),
        grid_manager=grid_manager,
        backend=backend,
        exchange=decomp_defs.SingleNodeExchange(),
    )
    metrics = metrics_factory.MetricsFieldsFactory(
        grid=mesh,
        vertical_grid=v_grid.VerticalGrid(config=vertical_config, vct_a=vct_a, vct_b=vct_b),
        decomposition_info=grid_manager.decomposition_info,
        geometry_source=geometry,
        topography=gtx.as_field((dims.CellDim,), data=topo_c),
        interpolation_source=interpolation,
        config=metrics_factory.MetricsConfig(),
        backend=backend,
        metadata=metrics_attributes.attrs,
        process_props=props,
    )
    return geometry, interpolation, metrics


@pytest.fixture
def setup(request: pytest.FixtureRequest, grid_manager: Any, backend_like: Any) -> _Setup:
    """
    With `ICON4PY_BENCH_CONSTANTS_ON_HOST=1` the constants are computed the same way, copied to
    host memory, and the factories that hold their device copies are dropped: only the JAX tests
    run in that mode, which otherwise hold every constant twice on the device.
    """
    on_host = os.environ.get("ICON4PY_BENCH_CONSTANTS_ON_HOST", "0") == "1"
    if on_host:
        geometry, interp, metrics = _host_factories(grid_manager, backend_like)
    else:
        geometry = request.getfixturevalue("geometry_field_source")
        interp = request.getfixturevalue("interpolation_field_source")
        metrics = request.getfixturevalue("metrics_field_source")
    config = _config()
    mesh = grid_manager.grid
    dtime = _dtime(config, mesh.num_cells)
    lsq_pseudoinv = interp.get(interpolation_attributes.LSQ_PSEUDOINV)
    from icon4py.model.atmosphere.subgrid_scale_physics.tmx import static_fields

    constants = {
        "edge_params": _edge_geometry(geometry),
        "cell_params": _cell_geometry(geometry),
        "geofac_div": interp.get(interpolation_attributes.GEOFAC_DIV),
        "rbf_vec_coeff_e": interp.get(interpolation_attributes.RBF_VEC_COEFF_E),
        "pos_on_tplane_e_1": interp.get(interpolation_attributes.POS_ON_TPLANE_E_X),
        "pos_on_tplane_e_2": interp.get(interpolation_attributes.POS_ON_TPLANE_E_Y),
        "lsq_pseudoinv_1": lsq_pseudoinv[:, 0, :],
        "lsq_pseudoinv_2": lsq_pseudoinv[:, 1, :],
        "ddqz_z_full": metrics.get(metrics_attributes.DDQZ_Z_FULL),
        "tmx_metric_state": static_fields.build_metric_state(metrics),
        "tmx_interpolation_state": static_fields.build_interpolation_state(interp),
    }
    inputs = _synthetic_inputs(mesh, metrics, dtime)
    if on_host:
        import gc

        constants = {k: _to_jax(v, np) for k, v in constants.items()}
        del geometry, interp, metrics, lsq_pseudoinv
        gc.collect()
        try:
            import cupy

            cupy.get_default_memory_pool().free_all_blocks()
            free, total = cupy.cuda.runtime.memGetInfo()
            print(f"CONSTANTS_ON_HOST device used after freeing {(total - free) / 2**30:.2f} GiB")
        except ImportError:
            pass
    return _Setup(mesh, config, dtime, constants, inputs)


# --- the stateless global steps ------------------------------------------------------------------


def _tracer_states(setup: _Setup, allocator: Any, convert: Any) -> dict[str, Any]:
    from icon4py.model.atmosphere.tracer_advection import tracer_advection_states

    c = setup.constants
    shallow = data_alloc.constant_field(setup.mesh, 1.0, dims.KDim, allocator=allocator)
    assert setup.config.tracer_advection is not None
    return {
        "config": setup.config.tracer_advection,
        "interpolation_state": convert(
            tracer_advection_states.AdvectionInterpolationState(
                geofac_div=c["geofac_div"],
                rbf_vec_coeff_e=c["rbf_vec_coeff_e"],
                pos_on_tplane_e_1=c["pos_on_tplane_e_1"],
                pos_on_tplane_e_2=c["pos_on_tplane_e_2"],
            )
        ),
        "least_squares_state": convert(
            tracer_advection_states.AdvectionLeastSquaresState(
                lsq_pseudoinv_1=c["lsq_pseudoinv_1"], lsq_pseudoinv_2=c["lsq_pseudoinv_2"]
            )
        ),
        "metric_state": convert(
            tracer_advection_states.AdvectionMetricState(
                deepatmo_divh=shallow,
                deepatmo_divzl=shallow,
                deepatmo_divzu=shallow,
                ddqz_z_full=c["ddqz_z_full"],
            )
        ),
        "edge_params": convert(c["edge_params"]),
    }


def _tmx_states(setup: _Setup, convert: Any) -> dict[str, Any]:
    c = setup.constants
    return {
        "metric_state": convert(c["tmx_metric_state"]),
        "interpolation_state": convert(c["tmx_interpolation_state"]),
        "edge_params": convert(c["edge_params"]),
        "cell_params": convert(c["cell_params"]),
    }


def _global_step(component: str, setup: _Setup, jnp: Any, execution: str = "jax") -> Any:
    """
    A function from the input fields to a dict of output fields: under `jax.jit` on JAX arrays,
    or with `execution` `"torch"` / `"torch_compile"` on torch tensors on the torch device, eager
    and under `torch.compile`; `jnp` is the array namespace of the inputs.
    """
    if execution != "jax":
        torch = jnp
        with torch_steps.device(torch):
            run = _component_run(component, setup, torch)
        return torch_steps.step(run, torch, compiled=execution == "torch_compile")
    import jax

    jitted = jax.jit(_component_run(component, setup, jnp))
    return lambda fields: jax.block_until_ready(jitted(fields))


def _component_run(component: str, setup: _Setup, jnp: Any) -> Any:
    from icon4py.model.common.states import tracer_prep_adv_states as prep_adv_states
    from icon4py.model.common.states.surface_flux_state import SurfaceFluxState

    convert = lambda obj: torch_steps.converted(obj, jnp, type(obj).__name__)  # noqa: E731
    mesh = dataclasses.replace(
        setup.mesh, connectivities={k: convert(v) for k, v in setup.mesh.connectivities.items()}
    )
    dtime = setup.dtime

    if component == "tracer":
        from icon4py.model.atmosphere.tracer_advection import tracer_advection_global

        step = tracer_advection_global.TracerAdvectionGlobal(
            grid=mesh, **_tracer_states(setup, jnp, convert), allocator=jnp
        )

        def run(f: dict[str, gtx.Field]) -> dict[str, gtx.Field]:
            tracer, hfl, vfl = step.run(
                p_tracer_now=f["qv"],
                airmass_now=f["air_mass"],
                airmass_new=f["airmass_new"],
                prep_adv=prep_adv_states.TracerPrepAdvState(
                    vn_traj=f["vn_traj"], mass_flx_me=f["mass_flx_me"], mass_flx_ic=f["mass_flx_ic"]
                ),
                dtime=dtime,
                even_timestep=False,
            )
            return {"tracer": tracer, "hfl": hfl, "vfl": vfl}

    elif component == "tmx":
        from icon4py.model.atmosphere.subgrid_scale_physics.tmx import tmx_global, tmx_states

        step = tmx_global.TmxGlobal(
            grid=mesh, config=setup.config.tmx, **_tmx_states(setup, convert), allocator=jnp
        )
        input_names = [f.name for f in dataclasses.fields(tmx_states.TmxInputState)]
        flux_names = [f.name for f in dataclasses.fields(SurfaceFluxState)]

        def run(f: dict[str, gtx.Field]) -> dict[str, gtx.Field]:
            fluxes = {n: f.get(n) for n in flux_names}
            for n in ("sensible_heat_flux", "latent_heat_flux"):
                fluxes[n] = f["evapotranspiration"]
            new, tendencies, diagnostics = step.run(
                input_state=tmx_states.TmxInputState(**{n: f[n] for n in input_names}),
                surface_flux_state=SurfaceFluxState(**fluxes),
                dtime=dtime,
            )
            return {
                **{f"new_{k.name}": getattr(new, k.name) for k in dataclasses.fields(new)},
                **{k.name: getattr(tendencies, k.name) for k in dataclasses.fields(tendencies)},
                **diagnostics,
            }

    elif component == "surface":
        from icon4py.model.atmosphere.subgrid_scale_physics.surface import (
            surface_global,
            surface_states,
        )

        step = surface_global.SurfaceGlobal(grid=mesh, config=setup.config.surface, allocator=jnp)
        num_levels = setup.mesh.num_levels

        def run(f: dict[str, gtx.Field]) -> dict[str, gtx.Field]:
            out = step.run(
                input_state=surface_states.SurfaceInputState(
                    surface_pressure=f["surface_pressure"],
                    temperature=f["temperature"][dims.KDim(num_levels - 1)],
                )
            )
            return {
                n: getattr(out, n)
                for n in (
                    "evapotranspiration",
                    "sensible_heat_flux",
                    "latent_heat_flux",
                    "thermal_energy_flux",
                    "vapor_energy_flux",
                )
            }

    elif component == "muphys":
        from icon4py.model.atmosphere.subgrid_scale_physics.muphys import muphys_global
        from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.definitions import Q

        step = muphys_global.MuphysGlobal(grid=mesh, qnc=setup.config.muphys.qnc)

        def run(f: dict[str, gtx.Field]) -> dict[str, gtx.Field]:
            te, q, pflx, pr, ps, pi, pg, pre = step.run(
                dz=f["dz"],
                te=f["temperature"],
                p=f["pressure"],
                rho=f["rho"],
                q=Q(**{s: f[f"q{s}"] for s in _SPECIES}),
                dtime=dtime,
            )
            return {
                "te": te,
                **{f"q{s}": getattr(q, s) for s in _SPECIES},
                "pflx": pflx,
                **{
                    n: x[dims.KDim(setup.mesh.num_levels - 1)]
                    for n, x in zip(("pr", "ps", "pi", "pg", "pre"), (pr, ps, pi, pg, pre))
                },
            }

    else:
        raise ValueError(component)

    return run


# --- the granules --------------------------------------------------------------------------------


def _granule_step(component: str, setup: _Setup, backend_like: Any) -> Any:
    """A synced call of the granule on `--backend` on fields it owns, and its output dict."""
    allocator = model_backends.get_allocator(backend_like)
    mesh = setup.mesh
    f = _as_fields(setup.inputs, allocator)
    zero = lambda *d: data_alloc.zero_field(mesh, *d, allocator=allocator)  # noqa: E731
    exchange = decomp_defs.SingleNodeExchange()

    if component == "tracer":
        from icon4py.model.atmosphere.tracer_advection import (
            tracer_advection,
            tracer_advection_states,
        )
        from icon4py.model.common.states import tracer_prep_adv_states as prep_adv_states

        granule = tracer_advection.convert_config_to_advection(
            grid=mesh,
            **_tracer_states(setup, allocator, lambda x: x),
            cell_params=setup.constants["cell_params"],
            backend=backend_like,
            exchange=exchange,
            even_timestep=False,
        )
        diagnostic = tracer_advection_states.AdvectionDiagnosticState(
            airmass_now=f["air_mass"],
            airmass_new=f["airmass_new"],
            grf_tend_tracer=zero(dims.CellDim, dims.KDim),
            hfl_tracer=zero(dims.EdgeDim, dims.KDim),
            vfl_tracer=zero(dims.CellDim, dims.KHalfDim),
        )
        prep_adv = prep_adv_states.TracerPrepAdvState(
            vn_traj=f["vn_traj"], mass_flx_me=f["mass_flx_me"], mass_flx_ic=f["mass_flx_ic"]
        )
        tracer_new = zero(dims.CellDim, dims.KDim)

        def run() -> None:
            granule.run(
                diagnostic_state=diagnostic,
                prep_adv=prep_adv,
                p_tracer_now=f["qv"],
                p_tracer_new=tracer_new,
                dtime=setup.dtime,
            )

        outputs = lambda: {  # noqa: E731
            "tracer": tracer_new,
            "hfl": diagnostic.hfl_tracer,
            "vfl": diagnostic.vfl_tracer,
        }

    elif component == "tmx":
        from icon4py.model.atmosphere.subgrid_scale_physics.tmx import tmx, tmx_states
        from icon4py.model.common.states.surface_flux_state import SurfaceFluxState

        granule = tmx.Tmx(
            grid=mesh,
            config=setup.config.tmx,
            **_tmx_states(setup, lambda x: x),
            backend=backend_like,
            exchange=exchange,
        )
        input_state = tmx_states.TmxInputState(
            **{k.name: f[k.name] for k in dataclasses.fields(tmx_states.TmxInputState)}
        )
        fluxes = {
            k.name: f.get(k.name, f["evapotranspiration"])
            for k in dataclasses.fields(SurfaceFluxState)
        }
        surface_flux_state = SurfaceFluxState(**fluxes)
        diagnostic = tmx_states.TmxDiagnosticState.allocate(mesh, allocator=allocator)
        tendencies = tmx_states.TmxTendencyState.allocate(mesh, allocator=allocator)
        new = tmx_states.TmxNewState.allocate(mesh, allocator=allocator)

        def run() -> None:
            granule.run(
                input_state=input_state,
                surface_flux_state=surface_flux_state,
                diagnostic_state=diagnostic,
                tendency_state=tendencies,
                new_state=new,
                dtime=setup.dtime,
            )

        def outputs() -> dict[str, gtx.Field]:
            return {
                **{f"new_{k.name}": getattr(new, k.name) for k in dataclasses.fields(new)},
                **{k.name: getattr(tendencies, k.name) for k in dataclasses.fields(tendencies)},
                **{
                    n: getattr(diagnostic, n)
                    for n in (
                        "km",
                        "kh",
                        "dissip_ke",
                        "heating",
                        "cptgz_vi",
                        "dissip_ke_vi",
                        "int_energy_vi",
                        "tend_int_energy_vi",
                    )
                },
            }

    elif component == "surface":
        from icon4py.model.atmosphere.subgrid_scale_physics.surface import surface, surface_states
        from icon4py.model.common.states.surface_flux_state import SurfaceFluxState

        granule = surface.Surface(grid=mesh, config=setup.config.surface, backend=backend_like)
        input_state = surface_states.SurfaceInputState(
            surface_pressure=f["surface_pressure"],
            temperature=gtx.as_field(
                (dims.CellDim,), setup.inputs["temperature"][:, -1], allocator=allocator
            ),
        )
        out = SurfaceFluxState(
            **{k.name: zero(dims.CellDim) for k in dataclasses.fields(SurfaceFluxState)}
        )

        def run() -> None:
            granule.run(input_state=input_state, out=out)

        outputs = lambda: {  # noqa: E731
            n: getattr(out, n)
            for n in (
                "evapotranspiration",
                "sensible_heat_flux",
                "latent_heat_flux",
                "thermal_energy_flux",
                "vapor_energy_flux",
            )
        }

    elif component == "muphys":
        from icon4py.model.atmosphere.subgrid_scale_physics.muphys import (
            component as muphys_component,
        )

        granule = muphys_component.MuphysComponent(
            grid=mesh,
            dtime=datetime.timedelta(seconds=setup.dtime),
            qnc=setup.config.muphys.qnc,
            backend=backend_like,
        )
        state = {"dz": f["dz"], "te": f["temperature"], "p": f["pressure"], "rho": f["rho"]}
        state.update({f"q{s}": f[f"q{s}"] for s in _SPECIES})
        result: dict[str, Any] = {}
        when = datetime.datetime(2008, 9, 1)

        def run() -> None:
            result.update(granule(state, when))

        def outputs() -> dict[str, gtx.Field]:
            return {
                "te": granule._te_in,
                **{f"q{s}": getattr(granule._q_in, s) for s in _SPECIES},
                "pflx": result["pflx"],
                **{
                    n: gtx.as_field((dims.CellDim,), result[n].asnumpy()[:, -1])
                    for n in ("pr", "ps", "pi", "pg", "pre")
                },
            }

    else:
        raise ValueError(component)

    def synced() -> None:
        run()
        device_utils.sync(allocator)

    return synced, outputs


# --- comparison ----------------------------------------------------------------------------------


def _compared(component: str, name: str, a: np.ndarray, num_levels: int, config: Any) -> np.ndarray:
    """The part of an output both implementations define."""
    if component == "tmx" and name in ("new_w", "tend_w"):
        return a[:, 1:num_levels]
    if component == "tmx" and name in ("km", "kh") and not config.tmx.use_km_const:
        return a[:, : num_levels - 1]
    return a


def _compare(
    component: str,
    setup: _Setup,
    actual: dict[str, np.ndarray],
    expected: dict[str, np.ndarray],
    label: str,
) -> None:
    num_levels = setup.mesh.num_levels
    assert set(actual) == set(expected), (sorted(actual), sorted(expected))
    failures = []
    for name in sorted(expected):
        a = _compared(component, name, actual[name], num_levels, setup.config)
        e = _compared(component, name, expected[name], num_levels, setup.config)
        diff = np.abs(a - e)
        print(
            f"CHECK {label} {component} {name}: max abs {np.nanmax(diff):.3e}, "
            f"max |ref| {np.nanmax(np.abs(e)):.3e}, nan {int(np.isnan(a).sum())}/{int(np.isnan(e).sum())}"
        )
        if not np.allclose(a, e, rtol=1e-12, atol=1e-16, equal_nan=False):
            failures.append(name)
    assert not failures, f"{label} {component}: {failures}"


def _dump_path(component: str, mesh: Any) -> pathlib.Path:
    directory = os.environ.get("ICON4PY_BENCH_DUMP_DIR")
    if directory is None:
        pytest.skip("needs ICON4PY_BENCH_DUMP_DIR")
    return pathlib.Path(directory) / f"{component}_{mesh.num_cells}x{mesh.num_levels}.npz"


def _perturbed(inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Negative control: `ICON4PY_BENCH_PERTURB` scales the temperature, pressure and qv."""
    factor = float(os.environ.get("ICON4PY_BENCH_PERTURB", "0"))
    if not factor:
        return inputs
    scaled = ("temperature", "pressure", "surface_pressure", "qv")
    return {**inputs, **{name: inputs[name] * (1.0 + factor) for name in scaled}}


def _jax_numpy() -> Any:
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    return jax.numpy


def _wait_for_compilation() -> None:
    if (wait := getattr(gtx, "wait_for_compilation", None)) is not None:
        wait()


def _peak_bytes() -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        import jax

        stats = jax.devices()[0].memory_stats() or {}
        out["jax_peak_bytes"] = int(stats.get("peak_bytes_in_use", -1))
    except Exception:
        pass
    try:
        import cupy

        out["cupy_pool_total_bytes"] = int(cupy.get_default_memory_pool().total_bytes())
        free, total = cupy.cuda.runtime.memGetInfo()
        out["device_used_bytes"] = int(total - free)
    except Exception:
        pass
    return out


# --- tests ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("component", COMPONENTS)
def test_global_jax_matches_granule(component: str, setup: _Setup, backend_like: Any) -> None:
    """
    Writes the JAX jit outputs, then checks them against the granule on `--backend` in the same
    process unless `ICON4PY_BENCH_JAX_ONLY=1`.
    """
    jnp = _jax_numpy()
    step = _global_step(component, setup, jnp)
    out = step(_as_fields(_perturbed(setup.inputs), jnp))
    jax_out = {k: np.asarray(v.asnumpy()) for k, v in out.items()}
    np.savez(_dump_path(component, setup.mesh), **jax_out)
    if os.environ.get("ICON4PY_BENCH_JAX_ONLY", "0") == "1":
        return
    run, outputs = _granule_step(component, setup, backend_like)
    run()
    _compare(
        component, setup, jax_out, {k: v.asnumpy() for k, v in outputs().items()}, "jax-vs-granule"
    )


@pytest.mark.parametrize("component", COMPONENTS)
def test_granule_matches_dump(component: str, setup: _Setup, backend_like: Any) -> None:
    """The granule on `--backend` against the JAX outputs another process wrote."""
    path = _dump_path(component, setup.mesh)
    if not path.exists():
        pytest.skip(f"no dump at {path}")
    run, outputs = _granule_step(component, setup, backend_like)
    run()
    with np.load(path) as dump:
        expected = dict(dump)
    _compare(
        component,
        setup,
        {k: v.asnumpy() for k, v in outputs().items()},
        expected,
        "granule-vs-jaxdump",
    )


@pytest.mark.benchmark
@pytest.mark.benchmark_only
@pytest.mark.parametrize("component", COMPONENTS)
def test_global_jax_benchmark(component: str, setup: _Setup, benchmark: Any) -> None:
    jnp = _jax_numpy()
    step = _global_step(component, setup, jnp)
    fields = _as_fields(setup.inputs, jnp)
    start = time.perf_counter()
    step(fields)
    benchmark.extra_info["first_call_s"] = time.perf_counter() - start
    benchmark(step, fields)
    benchmark.extra_info.update(_peak_bytes())


@pytest.mark.benchmark
@pytest.mark.benchmark_only
@pytest.mark.parametrize("component", COMPONENTS)
def test_granule_benchmark(
    component: str, setup: _Setup, backend_like: Any, benchmark: Any
) -> None:
    run, _ = _granule_step(component, setup, backend_like)
    start = time.perf_counter()
    run()
    _wait_for_compilation()
    benchmark.extra_info["first_call_s"] = time.perf_counter() - start
    benchmark(run)
    benchmark.extra_info.update(_peak_bytes())


def _torch_fields(inputs: dict[str, np.ndarray], torch: Any) -> dict[str, gtx.Field]:
    with torch_steps.device(torch):
        return torch_steps.converted(_as_fields(inputs, None), torch, "inputs")


@pytest.mark.benchmark
@pytest.mark.benchmark_only
@pytest.mark.parametrize("execution", ["torch_compile", "torch"])
@pytest.mark.parametrize("component", TORCH_COMPONENTS)
def test_global_torch_benchmark(
    component: str, execution: str, setup: _Setup, benchmark: Any
) -> None:
    """
    The global step on torch, under `torch.compile` or, with `ICON4PY_BENCH_TORCH_EAGER=1`,
    eager; the first call, with an empty inductor cache in a fresh `TORCHINDUCTOR_CACHE_DIR`, is
    timed in `extra_info["first_call_s"]`.
    """
    torch = pytest.importorskip("torch")
    if execution == "torch_compile" and (reason := torch_steps.compile_blocked(component)):
        pytest.skip(reason)
    if execution == "torch" and not torch_steps.eager_timed():
        pytest.skip("eager torch is timed only with ICON4PY_BENCH_TORCH_EAGER=1")
    step = _global_step(component, setup, torch, execution)
    fields = _torch_fields(setup.inputs, torch)
    start = time.perf_counter()
    step(fields)
    benchmark.extra_info["first_call_s"] = time.perf_counter() - start
    benchmark(step, fields)
    if torch.cuda.is_available():
        benchmark.extra_info["torch_peak_bytes"] = int(torch.cuda.max_memory_allocated())


@pytest.mark.parametrize("component", TORCH_COMPONENTS)
def test_global_torch_matches_jax(component: str, setup: _Setup) -> None:
    """
    The global step under `torch.compile` and eager on torch against `jax.jit`, on the same
    inputs; `ICON4PY_BENCH_PERTURB` perturbs the torch inputs as a negative control.
    """
    torch = pytest.importorskip("torch")
    jnp = _jax_numpy()
    num_levels = setup.mesh.num_levels

    def outputs(step: Any, fields: dict[str, gtx.Field]) -> dict[str, np.ndarray]:
        return {
            k: _compared(component, k, v.asnumpy(), num_levels, setup.config)
            for k, v in step(fields).items()
        }

    jax_out = outputs(_global_step(component, setup, jnp), _as_fields(setup.inputs, jnp))
    torch_inputs = _torch_fields(_perturbed(setup.inputs), torch)
    eager = outputs(_global_step(component, setup, torch, "torch"), torch_inputs)
    failures = torch_steps.compare(eager, jax_out, f"eager-vs-jax {component}")
    if not torch_steps.compile_blocked(component):
        compiled = outputs(_global_step(component, setup, torch, "torch_compile"), torch_inputs)
        failures += torch_steps.compare(compiled, eager, f"compile-vs-eager {component}")
        failures += torch_steps.compare(compiled, jax_out, f"compile-vs-jax {component}")
    assert not failures, failures
