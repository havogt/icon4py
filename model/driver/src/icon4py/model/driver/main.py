# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import logging
import pathlib
import sys
from typing import Annotated

import typer

from icon4py.model.common import model_backends, model_options, time
from icon4py.model.common.decomposition import (
    definitions as decomposition_defs,
    mpi_decomposition as mpi_decomp,
)
from icon4py.model.common.io import io as common_io
from icon4py.model.driver import config as driver_config, driver, driver_utils


log = logging.getLogger(__name__)

app = typer.Typer(no_args_is_help=True)


@app.command()
def main(
    *,
    grid_file_path: Annotated[pathlib.Path, typer.Option(help="Grid file path.")],
    config_file_path: Annotated[pathlib.Path, typer.Option(help="Configuration file path.")],
    output_path: Annotated[
        pathlib.Path | None,
        typer.Option(help="Optional override output path. Normally read from config."),
    ] = None,
    # it may be better to split device from backend,
    # or only asking for cpu or gpu and the best backend for perfornamce is handled inside icon4py,
    # whether to automatically use gpu if cupy is installed can be discussed further
    icon4py_backend: Annotated[
        str,
        typer.Option(
            help=f"GT4Py backend for running the entire driver. Possible options are: {' / '.join([*model_backends.BACKENDS.keys()])}",
        ),
    ],
    log_level: Annotated[
        str,
        typer.Option(
            help=f"Logging level of the model. Possible options are {' / '.join([*driver_utils._LOGGING_LEVELS.keys()])}",
        ),
    ] = next(iter(driver_utils._LOGGING_LEVELS.keys())),
    print_distributed_debug_msg: Annotated[
        bool,
        typer.Option(
            help="Print out debug logging message for all ranks (only works when log_level is set to debug).",
        ),
    ] = False,
    enable_output: Annotated[
        bool,
        typer.Option(
            "--enable-output/--no-enable-output",
            help="Write the prognostic and diagnostic fields to output.",
        ),
    ] = False,
    output_backend: Annotated[
        common_io.OutputBackend,
        typer.Option(help="Output file format."),
    ] = common_io.OutputBackend.ZARR,
    output_mode: Annotated[
        common_io.OutputMode,
        typer.Option(
            help=(
                "How ranks write output in distributed runs ('distributed' netCDF "
                "needs an MPI-parallel netCDF4 installation)."
            )
        ),
    ] = common_io.OutputMode.DISTRIBUTED,
    jax: Annotated[
        bool,
        typer.Option(
            "--jax/--no-jax",
            help=(
                "Run the time loop with the single-field-operator global steps on JAX arrays under "
                "jax.jit (global grids, no tracer transport, no output; needs --icon4py-backend "
                "embedded)."
            ),
        ),
    ] = False,
    jax_halo_exchange: Annotated[
        driver_utils.HaloExchange,
        typer.Option(
            help=(
                "With --jax on several ranks, where to exchange the halos: through host memory, "
                "or on the GPU from CuPy buffers (needs GHEX with GPU support)."
            ),
        ),
    ] = driver_utils.HaloExchange.HOST,
    jax_exchange_read_fields_only: Annotated[
        bool,
        typer.Option(
            "--jax-exchange-read-fields-only/--no-jax-exchange-read-fields-only",
            help=(
                "With --jax on several ranks, exchange only the fields the next step reads outside "
                "the owned points."
            ),
        ),
    ] = False,
    jax_constants_as_arguments: Annotated[
        bool,
        typer.Option(
            "--jax-constants-as-arguments/--no-jax-constants-as-arguments",
            help=(
                "With --jax, pass the static fields and connectivities to the jitted steps as "
                "arguments instead of compiling them in as constants."
            ),
        ),
    ] = True,
    jax_jit_time_step: Annotated[
        bool,
        typer.Option(
            "--jax-jit-time-step/--no-jax-jit-time-step",
            help=(
                "With --jax, jit the whole time step including its halo exchanges instead of each "
                "substep on its own; the exchanges then run on the device of the JAX backend, "
                "whatever --jax-halo-exchange says."
            ),
        ),
    ] = False,
    jax_spmd: Annotated[
        bool,
        typer.Option(
            "--jax-spmd/--no-jax-spmd",
            help=(
                "With --jax on several ranks, run the whole time step as one SPMD program over all "
                "ranks (jax.distributed), with the halo exchanges as JAX collectives."
            ),
        ),
    ] = False,
    jax_spmd_transport: Annotated[
        driver_utils.SpmdTransport,
        typer.Option(
            help=(
                "With --jax-spmd, how to move the halos: one all-to-all, or rounds of pairwise "
                "ppermute swaps."
            ),
        ),
    ] = driver_utils.SpmdTransport.ALL_TO_ALL,
    n_time_steps: Annotated[
        int | None,
        typer.Option(
            help="Number of time steps to run, instead of the configured end of simulation."
        ),
    ] = None,
) -> None:
    """
    CLI entry point that runs the icon4py driver.

    The configuration is read from ``config_file_path``, the driver is
    initialized, an initial condition is generated, and the time integration is
    run.
    """

    process_props = decomposition_defs.get_process_properties(
        decomposition_defs.get_runtype(with_mpi=mpi_decomp.mpi4py is not None)
    )
    driver_utils.configure_logging(
        logging_level=log_level,
        print_distributed_debug_msg=print_distributed_debug_msg,
        process_props=process_props,
    )

    config = driver_config.read_experiment_config_from_fortran(config_file_path)
    driver_overrides: dict[str, object] = {
        "enable_output": enable_output,
        "output_backend": output_backend,
        "output_mode": output_mode,
    }
    if output_path is not None:
        driver_overrides["output_path"] = output_path
    if n_time_steps is not None:
        driver_overrides["end_of_simulation"] = time.NumTimeSteps(n_time_steps)
    config = config.with_overrides(driver=driver_overrides)

    backend = model_options.customize_backend(
        program=None,
        backend=driver_utils.get_backend_from_name(icon4py_backend),
        backend_config=config.driver.backend_config,
    )
    allocator = model_backends.get_allocator(backend)

    grid_manager = driver_utils.create_grid_manager(
        grid_file_path=grid_file_path,
        vertical_grid_config=config.vertical_grid,
        allocator=allocator,
        process_props=process_props,
        extra_halo_rings=(
            driver_utils.JAX_EXTRA_HALO_RINGS if jax and not process_props.is_single_rank() else 0
        ),
    )

    driver.run_driver(
        config=config,
        grid_manager=grid_manager,
        process_props=process_props,
        backend=backend,
        jax=jax,
        jax_halo_exchange=jax_halo_exchange,
        jax_exchange_read_fields_only=jax_exchange_read_fields_only,
        jax_constants_as_arguments=jax_constants_as_arguments,
        jax_jit_time_step=jax_jit_time_step,
        jax_spmd=jax_spmd,
        jax_spmd_transport=jax_spmd_transport,
    )


if __name__ == "__main__":
    sys.exit(app())
