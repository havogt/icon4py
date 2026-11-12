# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
import os
import pathlib
import socket
import time
from collections.abc import Callable
from typing import Any

import pytest

from icon4py.model.common import dimension as dims, model_backends
from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.common.grid import grid_manager as gm
from icon4py.model.common.utils import device_utils
from icon4py.model.driver import config as driver_config, driver, driver_states
from icon4py.model.testing.fixtures.datatest import backend, backend_like, process_props

from ..fixtures import (
    BENCHMARK_ROUNDS,
    BENCHMARK_WARMUP_ROUNDS,
    driver_benchmark_config,
    driver_benchmark_experiment,
    driver_benchmark_grid,
    driver_benchmark_grid_manager,
)


def _barrier(process_props: decomp_defs.ProcessProperties) -> None:
    if process_props.comm is not None:
        process_props.comm.Barrier()


def _with_barriers[T](
    process_props: decomp_defs.ProcessProperties, fn: Callable[..., T]
) -> Callable[..., T]:
    def _wrapped(*args: Any, **kwargs: Any) -> T:
        _barrier(process_props)
        result = fn(*args, **kwargs)
        _barrier(process_props)
        return result

    return _wrapped


def _record_time_steps(
    icon4py_driver: driver.Icon4pyDriver, exchange_calls: list[tuple[str, int]]
) -> dict[str, list[Any]]:
    """Time every time step up to a device synchronization and collect its halo exchange calls."""
    record: dict[str, list[Any]] = {"step_wall": [], "exchanges": []}
    integrate = icon4py_driver._integrate_one_time_step

    def _integrate_one_time_step(**kwargs: Any) -> None:
        exchange_calls.clear()
        start = time.perf_counter()
        integrate(**kwargs)
        device_utils.sync(icon4py_driver.backend)
        record["step_wall"].append(time.perf_counter() - start)
        record["exchanges"].append(list(exchange_calls))

    icon4py_driver._integrate_one_time_step = _integrate_one_time_step  # type: ignore[method-assign]
    return record


def _dump_round(
    dump_dir: str,
    round_index: int,
    icon4py_driver: driver.Icon4pyDriver,
    record: dict[str, list[Any]],
    process_props: decomp_defs.ProcessProperties,
) -> None:
    decomposition_info = icon4py_driver.decomposition_info
    grid = icon4py_driver.grid
    content = {
        "rank": process_props.rank,
        "comm_size": process_props.comm_size,
        "host": socket.gethostname(),
        "round": round_index,
        "warmup_rounds": BENCHMARK_WARMUP_ROUNDS,
        "synchronize_timers": icon4py_driver.config.driver.synchronize_timers,
        "extra_halo_rings": icon4py_driver.config.driver.extra_halo_rings,
        "replace_skip_values": icon4py_driver.config.driver.replace_skip_values,
        "halo_exchange_mode": os.environ.get("ICON4PY_BENCH_HALO_EXCHANGE_MODE", "inner"),
        "dtime": icon4py_driver.model_time_variables.dtime_in_seconds,
        "ndyn_substeps": icon4py_driver.config.driver.ndyn_substeps,
        "ndyn_substeps_at_end": icon4py_driver.model_time_variables.ndyn_substeps_var,
        "local_points": {
            "cells": grid.num_cells,
            "edges": grid.num_edges,
            "vertices": grid.num_vertices,
        },
        "owned_points": {
            name: int(decomposition_info.owner_mask(dim).sum())
            for name, dim in (
                ("cells", dims.CellDim),
                ("edges", dims.EdgeDim),
                ("vertices", dims.VertexDim),
            )
        },
        **record,
        "timers": {
            name: [result.elapsed() for result in timer.results]
            for name, timer in icon4py_driver.timer_collection.timers.items()
        },
    }
    path = pathlib.Path(dump_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / f"timeloop-rank{process_props.rank}-round{round_index}.json").write_text(
        json.dumps(content)
    )


@pytest.mark.mpi
@pytest.mark.benchmark
@pytest.mark.continuous_benchmarking
@pytest.mark.benchmark_only
@pytest.mark.parametrize("process_props", [True], indirect=True, ids=["distributed"])
def test_benchmark_driver_init(
    driver_benchmark_config: driver_config.ExperimentConfig,
    driver_benchmark_grid_manager: gm.GridManager,
    process_props: decomp_defs.ProcessProperties,
    backend_like: model_backends.BackendLike,
    benchmark: Any,
) -> None:
    assert driver_benchmark_config.io is None

    def _setup() -> tuple[tuple[Any, ...], dict[str, Any]]:
        return (
            driver_benchmark_config,
            driver_benchmark_grid_manager,
            process_props,
            backend_like,
        ), {}

    def _timed(
        config: driver_config.ExperimentConfig,
        grid_manager: gm.GridManager,
        props: decomp_defs.ProcessProperties,
        bench_backend: model_backends.BackendLike,
    ) -> driver_states.DriverStates:
        icon4py_driver = driver.initialize_driver(
            config=config,
            grid_manager=grid_manager,
            process_props=props,
            backend=bench_backend,
        )
        allocator = model_backends.get_allocator(icon4py_driver.backend)
        return driver.initialize_driver_states(
            icon4py_driver=icon4py_driver,
            allocator=allocator,
        )

    benchmark.pedantic(
        _with_barriers(process_props, _timed),
        setup=_setup,
        rounds=BENCHMARK_ROUNDS,
        iterations=1,
        warmup_rounds=BENCHMARK_WARMUP_ROUNDS,
    )


@pytest.mark.mpi
@pytest.mark.benchmark
@pytest.mark.continuous_benchmarking
@pytest.mark.benchmark_only
@pytest.mark.parametrize("process_props", [True], indirect=True, ids=["distributed"])
def test_benchmark_driver_timeloop(
    driver_benchmark_config: driver_config.ExperimentConfig,
    driver_benchmark_grid_manager: gm.GridManager,
    process_props: decomp_defs.ProcessProperties,
    backend_like: model_backends.BackendLike,
    benchmark: Any,
) -> None:
    assert driver_benchmark_config.io is None
    dump_dir = os.environ.get("ICON4PY_BENCH_DUMP_DIR")
    exchange_calls: list[tuple[str, int]] = []
    recorded: list[tuple[int, driver.Icon4pyDriver, dict[str, list[Any]]]] = []
    if dump_dir:
        from icon4py.model.common.decomposition import mpi_decomposition  # noqa: PLC0415

        start_exchange = mpi_decomposition.GHexMultiNodeExchange.start

        def _counted_start(self: Any, dim: Any, *fields: Any, **kwargs: Any) -> Any:
            exchange_calls.append((dim.value, len(fields)))
            return start_exchange(self, dim, *fields, **kwargs)

        mpi_decomposition.GHexMultiNodeExchange.start = _counted_start  # type: ignore[method-assign]

    def _setup() -> tuple[tuple[Any, ...], dict[str, Any]]:
        icon4py_driver = driver.initialize_driver(
            config=driver_benchmark_config,
            grid_manager=driver_benchmark_grid_manager,
            process_props=process_props,
            backend=backend_like,
        )
        allocator = model_backends.get_allocator(icon4py_driver.backend)
        ds = driver.initialize_driver_states(icon4py_driver=icon4py_driver, allocator=allocator)
        if dump_dir:
            round_index = 0
            if recorded:
                round_index = recorded[0][0] + 1
                _dump_round(dump_dir, *recorded.pop(), process_props)
            recorded.append(
                (round_index, icon4py_driver, _record_time_steps(icon4py_driver, exchange_calls))
            )
        return (icon4py_driver, ds), {}

    def _timed(fresh_driver: driver.Icon4pyDriver, ds: driver_states.DriverStates) -> None:
        fresh_driver.time_integration(ds)

    benchmark.pedantic(
        _with_barriers(process_props, _timed),
        setup=_setup,
        rounds=BENCHMARK_ROUNDS,
        iterations=1,
        warmup_rounds=BENCHMARK_WARMUP_ROUNDS,
    )
    if dump_dir:
        _dump_round(dump_dir, *recorded.pop(), process_props)
        mpi_decomposition.GHexMultiNodeExchange.start = start_exchange  # type: ignore[method-assign]


@pytest.mark.mpi
@pytest.mark.benchmark
@pytest.mark.continuous_benchmarking
@pytest.mark.benchmark_only
@pytest.mark.parametrize("process_props", [True], indirect=True, ids=["distributed"])
def test_benchmark_driver_total(
    driver_benchmark_config: driver_config.ExperimentConfig,
    driver_benchmark_grid_manager: gm.GridManager,
    process_props: decomp_defs.ProcessProperties,
    backend_like: model_backends.BackendLike,
    benchmark: Any,
) -> None:
    assert driver_benchmark_config.io is None

    def _timed() -> tuple[driver_states.DriverStates, driver.Icon4pyDriver]:
        return driver.run_driver(
            config=driver_benchmark_config,
            grid_manager=driver_benchmark_grid_manager,
            process_props=process_props,
            backend=backend_like,
        )

    benchmark.pedantic(
        _with_barriers(process_props, _timed),
        rounds=BENCHMARK_ROUNDS,
        iterations=1,
        warmup_rounds=BENCHMARK_WARMUP_ROUNDS,
    )
