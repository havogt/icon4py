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
import re
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


def placement(tree: Any) -> dict[str, int]:
    """Number of array leaves per device platform."""
    counts: dict[str, int] = {}
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "devices"):
            for device in leaf.devices():
                counts[device.platform] = counts.get(device.platform, 0) + 1
    return counts


_COMPUTATION = re.compile(r"^(?:ENTRY )?%?([\w.\-]+) .*\{$")
_INSTRUCTION = re.compile(r"^\s*(?:ROOT )?%?([\w.\-]+) = .*?\s([a-z][a-z0-9\-]*)\((.*)$")
_CALLED = re.compile(r"(?:calls|to_apply|body|condition|branch_computations)=\{?%?([\w.\-, %]+)\}?")


def _hlo_ops(hlo: str) -> dict[str, dict[str, Any]]:
    """Each instruction of an optimized HLO module: opcode, fusion kind, gather inside, in a while body."""
    instructions: dict[str, dict[str, Any]] = {}
    calls: dict[str, set[str]] = {}
    bodies: set[str] = set()
    computation = None
    for line in hlo.splitlines():
        if (m := _COMPUTATION.match(line)) and "=" not in line.split("{")[0]:
            computation = m.group(1)
            calls.setdefault(computation, set())
            continue
        if computation is None or not (m := _INSTRUCTION.match(line)):
            continue
        name, opcode, rest = m.groups()
        called = {
            c.strip().lstrip("%") for g in _CALLED.findall(rest) for c in g.split(",") if c.strip()
        }
        calls[computation] |= called
        if opcode == "while" and (b := re.search(r"body=%?([\w.\-]+)", rest)):
            bodies.add(b.group(1))
        kind = re.search(r"kind=(k\w+)", rest)
        instructions[name] = {
            "computation": computation,
            "opcode": opcode,
            "kind": kind.group(1) if kind else "",
            "called": called,
        }

    def reachable(roots: set[str]) -> set[str]:
        seen, todo = set(), list(roots)
        while todo:
            c = todo.pop()
            if c not in seen:
                seen.add(c)
                todo.extend(calls.get(c, ()))
        return seen

    gather_computations = {
        i["computation"] for i in instructions.values() if i["opcode"] == "gather"
    }
    in_while = reachable(bodies)
    for op in instructions.values():
        op["gather"] = op["opcode"] == "gather" or bool(
            reachable(op["called"]) & gather_computations
        )
        op["in_while"] = op["computation"] in in_while
    return instructions


def profile_steps(static: Any, state: Any, num_steps: int, log_dir: pathlib.Path) -> dict[str, Any]:
    """
    Device time per HLO op of `num_steps` jitted steps (after a warm-up step), from a
    jax.profiler trace, keyed to the optimized HLO of the step.

    `busy_ms_per_step` sums the ops on the device (control ops whose body is traced op by op,
    while/conditional/call, are left out so nothing is counted twice); `wall_ms_per_step` is the
    synchronised wall time of the profiled steps. Their difference is time the device waits.
    """
    import gzip  # noqa: PLC0415 [import-outside-top-level]
    import json  # noqa: PLC0415 [import-outside-top-level]

    step = jax.jit(jax_api.time_step)
    state = jax.block_until_ready(step(static, state))
    state = jax.block_until_ready(step(static, state))
    ops = _hlo_ops(step.lower(static, state).compile().as_text())
    with jax.profiler.trace(str(log_dir), create_perfetto_trace=True):
        t0 = time.perf_counter()
        for _ in range(num_steps):
            state = jax.block_until_ready(step(static, state))
        wall = time.perf_counter() - t0
    trace_file = max(log_dir.rglob("perfetto_trace.json.gz"), key=lambda p: p.stat().st_mtime)
    with gzip.open(trace_file) as f:
        events = json.load(f)
    events = events["traceEvents"] if isinstance(events, dict) else events
    processes = {
        e["pid"]: e["args"]["name"]
        for e in events
        if e.get("ph") == "M" and e.get("name") == "process_name"
    }
    threads = {
        (e["pid"], e["tid"]): e["args"]["name"]
        for e in events
        if e.get("ph") == "M" and e.get("name") == "thread_name"
    }
    lines: dict[tuple[Any, Any], int] = {}
    for e in events:
        if e.get("ph") == "X":
            lines[e["pid"], e["tid"]] = lines.get((e["pid"], e["tid"]), 0) + 1
    per_op: dict[str, list[float]] = {}
    unmatched = 0.0
    for e in events:
        if e.get("ph") != "X":
            continue
        name = (e.get("args") or {}).get("hlo_op") or (
            e["name"]
            if "XLA Ops" in threads.get((e["pid"], e["tid"]), "")
            and "TPU" in processes.get(e["pid"], "")
            else None
        )
        if name is None:
            continue
        if name not in ops:
            unmatched += e["dur"]
            continue
        per_op.setdefault(name, []).append(e["dur"])
    control = {"while", "conditional", "call"}
    rows = []
    for name, durations in per_op.items():
        op = ops[name]
        if op["opcode"] in control:
            continue
        rows.append(
            {
                "op": name,
                "opcode": op["opcode"],
                "kind": op["kind"],
                "gather": op["gather"],
                "in_while": op["in_while"],
                "ms_per_step": sum(durations) / 1e3 / num_steps,
                "calls_per_step": len(durations) / num_steps,
            }
        )
    rows.sort(key=lambda r: -r["ms_per_step"])
    busy = sum(r["ms_per_step"] for r in rows)

    def share(predicate: Any) -> float:
        return sum(r["ms_per_step"] for r in rows if predicate(r)) / busy if busy else float("nan")

    return {
        "steps": num_steps,
        "trace_file": str(trace_file),
        "wall_ms_per_step": 1e3 * wall / num_steps,
        "busy_ms_per_step": busy,
        "unmatched_ms_per_step": unmatched / 1e3 / num_steps,
        "ops_traced": len(rows),
        "launches_per_step": sum(r["calls_per_step"] for r in rows),
        "share_gather": share(lambda r: r["gather"]),
        "share_in_while": share(lambda r: r["in_while"]),
        "share_gather_in_while": share(lambda r: r["gather"] and r["in_while"]),
        "trace_lines": sorted(
            (
                {
                    "process": processes.get(pid, str(pid)),
                    "thread": threads.get((pid, tid), str(tid)),
                    "events": n,
                }
                for (pid, tid), n in lines.items()
            ),
            key=lambda line: -line["events"],
        )[:12],
        "top": rows[:25],
    }
