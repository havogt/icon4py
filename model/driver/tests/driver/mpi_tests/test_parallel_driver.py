# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import logging
import pathlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing
import numpy as np
import pytest

from icon4py.model.atmosphere.dycore import solve_nonhydro as solve_nh
from icon4py.model.common import model_backends, time
from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.common.states import nonhydro_states
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.driver import config as driver_config, driver, driver_utils
from icon4py.model.testing import (
    datatest_utils as dt_utils,
    definitions as test_defs,
    grid_utils,
    parallel_helpers,
    test_utils,
)
from icon4py.model.testing.fixtures.datatest import (
    backend,
    backend_like,
    download_ser_data,
    experiment_description,
    process_props,
)


_log = logging.getLogger(__file__)


@pytest.mark.datatest
@pytest.mark.embedded_remap_error
@pytest.mark.mpi
@pytest.mark.parametrize("process_props", [True], indirect=True)
@pytest.mark.parametrize(
    "experiment_description, end_of_simulation",
    [
        pytest.param(
            test_defs.Experiments.JW,
            time.NumTimeSteps(1),
            marks=[pytest.mark.level("integration")],
            id="integration-jw",
        ),
        pytest.param(
            test_defs.Experiments.GAUSS3D,
            time.NumTimeSteps(1),
            marks=[pytest.mark.level("integration")],
            id="integration-gauss3d",
        ),
        pytest.param(
            test_defs.Experiments.MCH_CH_R04B09,
            time.NumTimeSteps(1),
            marks=[pytest.mark.level("integration")],
            id="integration-mch-ch-r04b09",
        ),
        pytest.param(
            test_defs.Experiments.EXCLAIM_APE_AES,
            time.NumTimeSteps(1),
            marks=[pytest.mark.level("integration")],
            id="integration-exclaim-ape-aes",
        ),
        pytest.param(
            test_defs.Experiments.JW,
            time.RelativeTime(days=7),
            marks=[pytest.mark.level("validation")],
            id="validation-jw",
        ),
    ],
)
def test_driver_compare_single_multi_rank(  # noqa: PLR0917 [too-many-positional-arguments]
    download_ser_data: None,
    experiment_description: test_defs.ExperimentDescription,
    end_of_simulation: time.EndOfSimulation,
    tmp_path: pathlib.Path,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend,
    backend_like: model_backends.BackendLike,
) -> None:
    _run_driver_compare_single_multi_rank(
        experiment_description=experiment_description,
        end_of_simulation=end_of_simulation,
        tmp_path=tmp_path,
        process_props=process_props,
        backend=backend,
        backend_like=backend_like,
    )


@pytest.mark.datatest
@pytest.mark.embedded_remap_error
@pytest.mark.mpi
@pytest.mark.level("integration")
@pytest.mark.parametrize("process_props", [True], indirect=True)
@pytest.mark.parametrize("experiment_description", [test_defs.Experiments.JW], ids=["jw"])
@pytest.mark.parametrize(
    "extra_halo_rings, halo_exchange_mode, num_time_steps, expect_mismatch",
    [
        pytest.param(2, decomp_defs.HaloExchangeMode.INNER, 1, False, id="rings2-inner"),
        pytest.param(2, decomp_defs.HaloExchangeMode.INNER_WIDE, 1, False, id="rings2-inner_wide"),
        pytest.param(2, decomp_defs.HaloExchangeMode.ENTRY, 1, False, id="rings2-entry"),
        pytest.param(2, decomp_defs.HaloExchangeMode.ENTRY, 2, False, id="rings2-entry-2steps"),
        # the halo is too shallow for a component without inner exchanges
        pytest.param(1, decomp_defs.HaloExchangeMode.ENTRY, 1, True, id="rings1-entry"),
    ],
)
def test_driver_compare_single_multi_rank_halo_exchange_mode(  # noqa: PLR0917 [too-many-positional-arguments]
    download_ser_data: None,
    experiment_description: test_defs.ExperimentDescription,
    extra_halo_rings: int,
    halo_exchange_mode: decomp_defs.HaloExchangeMode,
    num_time_steps: int,
    expect_mismatch: bool,
    tmp_path: pathlib.Path,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend,
    backend_like: model_backends.BackendLike,
) -> None:
    _run_driver_compare_single_multi_rank(
        experiment_description=experiment_description,
        end_of_simulation=time.NumTimeSteps(num_time_steps),
        tmp_path=tmp_path,
        process_props=process_props,
        backend=backend,
        backend_like=backend_like,
        extra_halo_rings=extra_halo_rings,
        halo_exchange_mode=halo_exchange_mode,
        expect_mismatch=expect_mismatch,
    )


# A field carried from one substep to the next is exchanged in the slot the reading substep
# finds it in, or in the slot the writing substep leaves it in.
_SOLVE_NONHYDRO_CARRIED_FIELDS: dict[
    str, Callable[[nonhydro_states.DiagnosticStateNonHydro], tuple[Any, ...]]
] = {
    "tangential_wind": lambda diagnostic_state: (diagnostic_state.tangential_wind,),
    "normal_wind_advective_tendency": (
        lambda diagnostic_state: (
            diagnostic_state.normal_wind_advective_tendency.predictor,
            diagnostic_state.normal_wind_advective_tendency.corrector,
        )
    ),
    "vertical_wind_advective_tendency": (
        lambda diagnostic_state: (
            diagnostic_state.vertical_wind_advective_tendency.predictor,
            diagnostic_state.vertical_wind_advective_tendency.corrector,
        )
    ),
}


class _FieldDroppingExchange:
    """Forwards exchanges without the dropped fields and counts how often it dropped one."""

    def __init__(
        self, exchange: decomp_defs.ExchangeRuntime, dropped_fields: tuple[Any, ...]
    ) -> None:
        self._exchange = exchange
        self._dropped_fields = dropped_fields
        self.num_drops = 0

    def _kept(self, fields: Sequence[gtx.Field]) -> list[gtx.Field]:
        kept_fields = [
            field
            for field in fields
            if not any(field is dropped_field for dropped_field in self._dropped_fields)
        ]
        self.num_drops += len(fields) - len(kept_fields)
        return kept_fields

    def start(
        self, dim: gtx.Dimension, *fields: gtx.Field, **kwargs: Any
    ) -> decomp_defs.ExchangeResult:
        return self._exchange.start(dim, *self._kept(fields), **kwargs)

    def exchange(self, dim: gtx.Dimension, *fields: gtx.Field, **kwargs: Any) -> None:
        self._exchange.exchange(dim, *self._kept(fields), **kwargs)

    def start_merged(
        self, fields: Mapping[Any, Sequence[gtx.Field]], **kwargs: Any
    ) -> decomp_defs.ExchangeResult:
        return self._exchange.start_merged(
            {key: self._kept(key_fields) for key, key_fields in fields.items()}, **kwargs
        )

    def exchange_merged(self, fields: Mapping[Any, Sequence[gtx.Field]], **kwargs: Any) -> None:
        self._exchange.exchange_merged(
            {key: self._kept(key_fields) for key, key_fields in fields.items()}, **kwargs
        )


@pytest.mark.datatest
@pytest.mark.embedded_remap_error
@pytest.mark.mpi
@pytest.mark.level("integration")
@pytest.mark.parametrize("process_props", [True], indirect=True)
@pytest.mark.parametrize("experiment_description", [test_defs.Experiments.JW], ids=["jw"])
@pytest.mark.parametrize(
    "dropped_field_name",
    [
        pytest.param(field_name, id=f"drop-{field_name}")
        for field_name in _SOLVE_NONHYDRO_CARRIED_FIELDS
    ],
)
def test_driver_compare_single_multi_rank_without_exchange_of_carried_field(  # noqa: PLR0917 [too-many-positional-arguments]
    download_ser_data: None,
    experiment_description: test_defs.ExperimentDescription,
    dropped_field_name: str,
    tmp_path: pathlib.Path,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend,
    backend_like: model_backends.BackendLike,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    time_step = solve_nh.SolveNonhydro.time_step
    num_drops = 0

    def time_step_without_field(
        self: solve_nh.SolveNonhydro,
        *,
        diagnostic_state_nh: nonhydro_states.DiagnosticStateNonHydro,
        **kwargs: Any,
    ) -> None:
        nonlocal num_drops
        exchange = self._exchange
        dropping_exchange = _FieldDroppingExchange(
            exchange, _SOLVE_NONHYDRO_CARRIED_FIELDS[dropped_field_name](diagnostic_state_nh)
        )
        self._exchange = dropping_exchange  # type: ignore[assignment]
        try:
            time_step(self, diagnostic_state_nh=diagnostic_state_nh, **kwargs)
        finally:
            self._exchange = exchange
        num_drops += dropping_exchange.num_drops

    monkeypatch.setattr(solve_nh.SolveNonhydro, "time_step", time_step_without_field)

    # the dropped fields are carried from one substep to the next
    config = driver_config.read_experiment_config_from_yaml(
        dt_utils.get_path_for_experiment(experiment_description, process_props) / "config.yml"
    )
    assert config.driver.ndyn_substeps >= 2

    _run_driver_compare_single_multi_rank(
        experiment_description=experiment_description,
        end_of_simulation=time.NumTimeSteps(1),
        tmp_path=tmp_path,
        process_props=process_props,
        backend=backend,
        backend_like=backend_like,
        extra_halo_rings=2,
        halo_exchange_mode=decomp_defs.HaloExchangeMode.ENTRY,
        expect_mismatch=True,
    )
    assert num_drops > 0, f"'{dropped_field_name}' was never dropped from an exchange."


def _with_halo_exchange_mode(
    config: driver_config.ExperimentConfig, halo_exchange_mode: decomp_defs.HaloExchangeMode
) -> driver_config.ExperimentConfig:
    return config.with_overrides(
        **{
            component: {"halo_exchange_mode": halo_exchange_mode}
            for component in ("diffusion", "nonhydrostatic")
            if getattr(config, component) is not None
        }
    )


def _run_driver_compare_single_multi_rank(  # noqa: PLR0917 [too-many-positional-arguments]
    experiment_description: test_defs.ExperimentDescription,
    end_of_simulation: time.EndOfSimulation,
    tmp_path: pathlib.Path,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend,
    backend_like: model_backends.BackendLike,
    extra_halo_rings: int = 0,
    halo_exchange_mode: decomp_defs.HaloExchangeMode = decomp_defs.HaloExchangeMode.INNER,
    expect_mismatch: bool = False,
) -> None:
    if experiment_description.grid.limited_area:
        pytest.xfail("Limited-area grids not yet supported")

    if model_backends.is_cpu_backend(backend) and test_utils.is_gtfn_backend(backend):
        atol = 1e-12
        rtol = 1e-14
    else:
        atol = 1e-9
        rtol = 0.0
    atol, rtol = test_utils.get_mpi_comparison_tolerance(backend, atol=atol, rtol=rtol)
    if experiment_description is test_defs.Experiments.EXCLAIM_APE_AES:
        # TODO (Yilu): drop this tolerance once #1535 is merged. The multi-rank global sum is
        # order-dependent, so the moisture rescale of the APE initial condition makes qv differ
        # by ~1e-14 relative from step 0, and muphys amplifies that to ~2e-11 in qc, qi and qr.
        # #1535 ports ICON's order-insensitive sum, after which this experiment is bitwise
        # reproducible (verified on #1534). The tolerance still catches a missing halo exchange
        # (1e-5 and more).
        atol, rtol = 1e-16, 1e-9

    _log.info(
        f"running on {process_props.comm} with {process_props.comm_size} ranks and atol = {atol}, rtol = {rtol}"
    )

    allocator = model_backends.get_allocator(backend)

    grid_file_path = grid_utils._download_grid_file(experiment_description.grid)
    config_file_path = dt_utils.get_path_for_experiment(experiment_description, process_props)

    config = driver_config.read_experiment_config_from_yaml(config_file_path / "config.yml")

    single_rank_process_props = decomp_defs.SingleNodeProcessProperties()
    single_rank_config = config.with_overrides(driver={"end_of_simulation": end_of_simulation})
    single_rank_grid_manager = driver_utils.create_grid_manager(
        grid_file_path=grid_file_path,
        vertical_grid_config=single_rank_config.vertical_grid,
        allocator=allocator,
        process_props=single_rank_process_props,
    )
    single_rank_ds, _ = driver.run_driver(
        config=single_rank_config,
        grid_manager=single_rank_grid_manager,
        process_props=single_rank_process_props,
        backend=backend_like,
    )

    multi_rank_config = _with_halo_exchange_mode(config, halo_exchange_mode).with_overrides(
        driver={"end_of_simulation": end_of_simulation, "extra_halo_rings": extra_halo_rings}
    )
    multi_rank_grid_manager = driver_utils.create_grid_manager(
        grid_file_path=grid_file_path,
        vertical_grid_config=multi_rank_config.vertical_grid,
        allocator=allocator,
        process_props=process_props,
        extra_halo_rings=multi_rank_config.driver.extra_halo_rings,
    )
    multi_rank_ds, multi_rank_driver = driver.run_driver(
        config=multi_rank_config,
        grid_manager=multi_rank_grid_manager,
        process_props=process_props,
        backend=backend_like,
    )

    fields = ["vn", "w", "exner", "theta_v", "rho"]
    compared = {
        field_name: getattr(single_rank_ds.prognostics.current, field_name) for field_name in fields
    }
    multi_rank_compared = {
        field_name: getattr(multi_rank_ds.prognostics.current, field_name) for field_name in fields
    }
    # the physics experiments validate the tracers too: stale tracer halos are exactly
    # what the physics driver's end-of-step exchange fixes
    for tracer in single_rank_ds.tracers.current.active_fields():
        compared[tracer.name] = tracer.field
    for tracer in multi_rank_ds.tracers.current.active_fields():
        multi_rank_compared[tracer.name] = tracer.field
    single_rank_reference_fields: dict[str, object] = {
        field_name: field.asnumpy() for field_name, field in compared.items()
    }

    decomposition_info = multi_rank_driver.decomposition_info
    # The extra halo rings are valid only where a component exchanged the field, and a
    # component without inner exchanges leaves the whole halo stale.
    check_halos = extra_halo_rings == 0 and halo_exchange_mode is decomp_defs.HaloExchangeMode.INNER
    mismatched_fields = []
    for field_name in compared:
        print(f"\nverifying field {field_name}")
        global_reference_field = process_props.comm.bcast(
            single_rank_reference_fields.get(field_name),
            root=0,
        )
        local_field = multi_rank_compared[field_name]
        dim = local_field.domain.dims[0]
        try:
            parallel_helpers.check_local_global_field(
                decomposition_info=decomposition_info,
                process_props=process_props,
                dim=dim,
                global_reference_field=global_reference_field,
                local_field=local_field.asnumpy(),
                check_halos=check_halos,
                atol=atol,
                rtol=rtol,
            )
        except AssertionError as error:
            if not expect_mismatch:
                raise
            print(error)
            mismatched_fields.append(field_name)
        if expect_mismatch:
            owned = data_alloc.as_numpy(
                decomposition_info.local_index(dim, decomp_defs.DecompositionInfo.EntryType.OWNED)
            )
            assert np.isfinite(local_field.asnumpy()[owned]).all(), (
                f"'{field_name}' is not finite on the owned points."
            )

    # the owned points are compared on the first rank only
    if expect_mismatch and process_props.rank == 0:
        assert mismatched_fields, "Expected the multi-rank run to differ from the single-rank run."
