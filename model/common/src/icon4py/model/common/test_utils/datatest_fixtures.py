# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest

from ..decomposition.definitions import SingleNodeRun
from .data_handling import download_and_extract
from .datatest_utils import (
    DATA_URIS,
    SER_DATA_BASEPATH,
    create_icon_serial_data_provider,
    get_datapath_for_ranked_data,
    get_processor_properties_for_run,
    get_ranked_data_path,
)


# TODO: a run that contains all the fields needed for dycore, diffusion, interpolation fields needs to be consolidated


@pytest.fixture(params=[False], scope="session")
def processor_props(request):
    return get_processor_properties_for_run(SingleNodeRun())


@pytest.fixture(scope="session")
def ranked_data_path(processor_props):
    return get_ranked_data_path(SER_DATA_BASEPATH, processor_props)


@pytest.fixture(scope="session")
def datapath(ranked_data_path):
    return get_datapath_for_ranked_data(ranked_data_path)


@pytest.fixture(scope="session")
def download_ser_data(request, processor_props, ranked_data_path, pytestconfig):
    """
    Get the binary ICON data from a remote server.

    Session scoped fixture which is a prerequisite of all the other fixtures in this file.
    """
    try:
        has_data_marker = any(map(lambda i: i.iter_markers(name="datatest"), request.node.items))
        if not has_data_marker or not request.config.getoption("datatest"):
            pytest.skip("not running datatest marked tests")
    except ValueError:
        pass

    try:
        uri = DATA_URIS[processor_props.comm_size]

        data_file = ranked_data_path.joinpath(
            f"mch_ch_r04b09_dsl_mpitask{processor_props.comm_size}.tar.gz"
        ).name
        if processor_props.rank == 0:
            download_and_extract(uri, ranked_data_path, data_file)
        if processor_props.comm:
            processor_props.comm.barrier()
    except KeyError:
        raise AssertionError(
            f"no data for communicator of size {processor_props.comm_size} exists, use 1, 2 or 4"
        )


@pytest.fixture(scope="session")
def data_provider(download_ser_data, datapath, processor_props):
    return create_icon_serial_data_provider(datapath, processor_props)


@pytest.fixture
def grid_savepoint(data_provider):
    return data_provider.from_savepoint_grid()


@pytest.fixture
def icon_grid(grid_savepoint):
    """
    Load the icon grid from an ICON savepoint.

    Uses the special grid_savepoint that contains data from p_patch
    """
    return grid_savepoint.construct_icon_grid()


@pytest.fixture
def decomposition_info(data_provider):
    return data_provider.from_savepoint_grid().construct_decomposition_info()


@pytest.fixture
def damping_height():
    return 12500


@pytest.fixture
def ndyn_substeps():
    """
    Return number of dynamical substeps.

    Serialized data uses a reduced number (2 instead of the default 5) in order to reduce the amount
    of data generated.
    """
    return 2


@pytest.fixture
def linit():
    """
    Set the 'linit' flag for the ICON diffusion data savepoint.

    Defaults to False
    """
    return False


@pytest.fixture
def step_date_init():
    """
    Set the step date for the loaded ICON time stamp at start of module.

    Defaults to 2021-06-20T12:00:10.000'
    """
    return "2021-06-20T12:00:10.000"


@pytest.fixture
def step_date_exit():
    """
    Set the step date for the loaded ICON time stamp at the end of module.

    Defaults to 2021-06-20T12:00:10.000'
    """
    return "2021-06-20T12:00:10.000"


@pytest.fixture
def interpolation_savepoint(data_provider):  # F811
    """Load data from ICON interplation state savepoint."""
    return data_provider.from_interpolation_savepoint()


@pytest.fixture
def metrics_savepoint(data_provider):  # F811
    """Load data from ICON mestric state savepoint."""
    return data_provider.from_metrics_savepoint()


@pytest.fixture
def metrics_nonhydro_savepoint(data_provider):  # F811
    """Load data from ICON metric state nonhydro savepoint."""
    return data_provider.from_metrics_nonhydro_savepoint()


@pytest.fixture
def savepoint_velocity_init(data_provider, step_date_init, istep_init, vn_only, jstep_init):  # F811
    """
    Load data from ICON savepoint at start of velocity_advection module.

    date of the timestamp to be selected can be set seperately by overriding the 'step_data'
    fixture, passing 'step_data=<iso_string>'
    """
    return data_provider.from_savepoint_velocity_init(
        istep=istep_init, vn_only=vn_only, date=step_date_init, jstep=jstep_init
    )


@pytest.fixture
def savepoint_nonhydro_init(data_provider, step_date_init, istep_init, jstep_init):  # noqa F811
    """
    Load data from ICON savepoint at exist of solve_nonhydro module.

    date of the timestamp to be selected can be set seperately by overriding the 'step_data'
    fixture, passing 'step_data=<iso_string>'
    """
    return data_provider.from_savepoint_nonhydro_init(
        istep=istep_init, date=step_date_init, jstep=jstep_init
    )


@pytest.fixture
def savepoint_velocity_exit(data_provider, step_date_exit, istep_exit, vn_only, jstep_exit):  # F811
    """
    Load data from ICON savepoint at exist of solve_nonhydro module.

    date of the timestamp to be selected can be set seperately by overriding the 'step_data'
    fixture, passing 'step_data=<iso_string>'
    """
    return data_provider.from_savepoint_velocity_exit(
        istep=istep_exit, vn_only=vn_only, date=step_date_exit, jstep=jstep_exit
    )


@pytest.fixture
def savepoint_nonhydro_exit(data_provider, step_date_exit, istep_exit, jstep_exit):  # noqa F811
    """
    Load data from ICON savepoint at exist of solve_nonhydro module.

    date of the timestamp to be selected can be set seperately by overriding the 'step_data'
    fixture, passing 'step_data=<iso_string>'
    """
    return data_provider.from_savepoint_nonhydro_exit(
        istep=istep_exit, date=step_date_exit, jstep=jstep_exit
    )


@pytest.fixture
def savepoint_nonhydro_step_exit(data_provider, step_date_exit, jstep_exit):  # noqa F811
    """
    Load data from ICON savepoint at final exit (after predictor and corrector, and 3 final stencils) of solve_nonhydro module.

    date of the timestamp to be selected can be set seperately by overriding the 'step_data'
    fixture, passing 'step_data=<iso_string>'
    """
    return data_provider.from_savepoint_nonhydro_step_exit(date=step_date_exit, jstep=jstep_exit)


@pytest.fixture
def istep_init():
    return 1


@pytest.fixture
def istep_exit():
    return 1


@pytest.fixture
def jstep_init():
    return 0


@pytest.fixture
def jstep_exit():
    return 0


@pytest.fixture
def ntnd(savepoint_velocity_init):
    return savepoint_velocity_init.get_metadata("ntnd").get("ntnd")


@pytest.fixture
def vn_only():
    return False
