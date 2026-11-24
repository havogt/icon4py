# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""
The Jablonowski-Williamson step of `jax_api` (diffusion and solve_nonhydro, no physics, no
tracers) under jax.jit, for `jw_tpu.ipynb` and for the CPU reference it compares against.

ICON4PY_FLOAT_PRECISION must be set before this module is imported.
"""

from __future__ import annotations

import datetime
import hashlib
import pathlib
import statistics
import tarfile
import time
import urllib.request
from typing import Any

import numpy as np

from icon4py.model.common import type_alias
from icon4py.model.driver import config as driver_config, jax_api, jax_utils


jax = jax_utils.import_jax()

HERE = pathlib.Path(__file__).parent
CONFIG_FILE = HERE / "jw_config.yml"
PROGNOSTICS = ("vn", "w", "rho", "exner", "theta_v")
GRIDS = {
    "R02B04": (
        "icon_grid_0013_R02B04_R",
        "https://rgw.cscs.ch/c2sm:testdata/grids/icon_grid_0013_R02B04_R.tar.gz",
    ),
    "R02B05": (
        "icon_grid_0019_R02B05_G",
        "http://icon-downloads.mpimet.mpg.de/grids/public/mpim/0019/icon_grid_0019_R02B05_G.nc",
    ),
}
SAMPLE_SIZE = 256
SAMPLE_SEED = 38


def config_sha256() -> str:
    return hashlib.sha256(CONFIG_FILE.read_bytes()).hexdigest()


def download_grid(name: str, directory: pathlib.Path) -> pathlib.Path:
    stem, url = GRIDS[name]
    grid_file = directory / stem / f"{stem}.nc"
    if grid_file.exists():
        return grid_file
    grid_file.parent.mkdir(parents=True, exist_ok=True)
    target = grid_file.parent / url.rsplit("/", 1)[1]
    urllib.request.urlretrieve(url, target)
    if target.suffix == ".gz":
        with tarfile.open(target) as archive:
            archive.extractall(grid_file.parent, filter="data")
        target.unlink()
    if not grid_file.exists():
        raise FileNotFoundError(f"{url} did not give {grid_file}")
    return grid_file


def init(grid_file: pathlib.Path, output_path: pathlib.Path) -> tuple[Any, Any]:
    config = driver_config.read_experiment_config_from_yaml(CONFIG_FILE).with_overrides(
        driver={
            "output_path": output_path,
            "start_of_timestepping": datetime.datetime(2008, 9, 1, tzinfo=datetime.UTC),
            "end_of_simulation": 1000,
        }
    )
    return jax_api.init_from_grid_file(grid_file, config)


def leaf_dtypes(tree: Any) -> dict[str, list[str]]:
    by_dtype: dict[str, list[str]] = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        if hasattr(leaf, "dtype"):
            by_dtype.setdefault(str(leaf.dtype), []).append(jax.tree_util.keystr(path))
    return by_dtype


def precision_report(static: Any, state: Any, step: Any) -> dict[str, Any]:
    """The float dtypes of every leaf and of the lowered programs of both traces of a step."""
    expected = np.dtype(type_alias.wpfloat).name
    other = "float64" if expected == "float32" else "float32"
    hlo_type = {"float32": "f32", "float64": "f64"}
    first = step.lower(static, state).as_text()
    later = step.lower(static, jax.block_until_ready(step(static, state))).as_text()
    leaves = leaf_dtypes((static, state))
    return {
        "precision": type_alias.precision,
        "wpfloat": expected,
        "leaf_dtypes": {k: len(v) for k, v in leaves.items()},
        "leaves_not_wpfloat": leaves.get(other, []),
        f"hlo_{hlo_type[other]}_first_step": first.count(hlo_type[other]),
        f"hlo_{hlo_type[other]}_later_steps": later.count(hlo_type[other]),
    }


def sample_indices(static: Any, state: Any) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SAMPLE_SEED)
    sizes = {name: state.prognostic[name].shape[0] for name in PROGNOSTICS}
    return {
        name: np.sort(rng.choice(size, size=min(SAMPLE_SIZE, size), replace=False))
        for name, size in sorted(sizes.items())
    }


def prognostics(state: Any) -> dict[str, np.ndarray]:
    return {name: state.prognostic[name].asnumpy() for name in PROGNOSTICS}


def run(static: Any, state: Any, num_steps: int, step: Any = None) -> Any:
    step = step or jax.jit(jax_api.time_step)
    for _ in range(num_steps):
        state = step(static, state)
    return jax.block_until_ready(state)


def time_steps(static: Any, state: Any, warmup: int, rounds: int) -> tuple[dict[str, float], Any]:
    """
    Trace and compile time of the first step on its own (ahead of time), then the jitted
    step: its first call, which compiles again unless the persistent cache serves it, the
    second (a new trace: `at_initial_timestep` is metadata), and the median of `rounds`
    synchronised steps after `warmup` steps.
    """
    step = jax.jit(jax_api.time_step)
    t0 = time.perf_counter()
    lowered = step.lower(static, state)
    t1 = time.perf_counter()
    lowered.compile()
    t2 = time.perf_counter()
    timings = {"trace_and_lower_s": t1 - t0, "compile_s": t2 - t1}
    calls = []
    for _ in range(max(warmup, 2)):
        t0 = time.perf_counter()
        state = jax.block_until_ready(step(static, state))
        calls.append(time.perf_counter() - t0)
    timings["first_call_s"], timings["second_call_s"] = calls[0], calls[1]
    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        state = jax.block_until_ready(step(static, state))
        times.append(time.perf_counter() - t0)
    timings |= {
        "step_median_ms": 1e3 * statistics.median(times),
        "step_min_ms": 1e3 * min(times),
        "step_max_ms": 1e3 * max(times),
        "rounds": rounds,
    }
    return timings, state


def compare(
    computed: dict[str, np.ndarray], reference: Any, indices: dict[str, np.ndarray], tag: str
) -> dict[str, dict[str, float]]:
    """Max abs and max rel difference of the sampled prognostics against `reference[f"{tag}_{name}"]`."""
    result = {}
    for name in PROGNOSTICS:
        a = computed[name][indices[name]].astype(np.float64)
        b = reference[f"{tag}_{name}"].astype(np.float64)
        abs_diff = np.abs(a - b)
        result[name] = {
            "max_abs": float(abs_diff.max()),
            "max_rel": float(abs_diff.max() / max(np.abs(b).max(), np.finfo(np.float64).tiny)),
            "finite": bool(np.isfinite(a).all()),
        }
    return result
