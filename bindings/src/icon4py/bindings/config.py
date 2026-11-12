# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import os

from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.utils import env


WAIT_FOR_COMPILATION: bool = env.flag_to_bool("ICON4PY_WAIT_FOR_COMPILATION", False)
"""Wait in granule initialization until jit compilation is complete."""

HALO_EXCHANGE_MODE: decomposition.HaloExchangeMode = decomposition.HaloExchangeMode(
    os.environ.get("ICON4PY_HALO_EXCHANGE_MODE", decomposition.HaloExchangeMode.INNER.value).lower()
)
"""Halo exchange mode of the diffusion and solve_nonhydro granules: 'inner', 'inner_wide' or 'entry'."""

EXCHANGE_CARRIED_INPUTS_AT_EVERY_SUBSTEP: bool = env.flag_to_bool(
    "ICON4PY_EXCHANGE_CARRIED_INPUTS_AT_EVERY_SUBSTEP", False
)
"""In halo exchange mode 'entry', exchange the fields of the first substep's entry exchange at every substep."""

SKIP_REDUNDANT_ENTRY_EXCHANGE: bool = env.flag_to_bool("ICON4PY_HALO_SKIP_REDUNDANT", False)
"""In halo exchange mode 'entry', diffusion does not exchange at entry the fields whose halo solve_nonhydro has just exchanged."""

HALO_DEPTH_READ_BY_CALLER: int | None = 1 if decomposition.LIMIT_EXCHANGE_DEPTH else None
"""Halo rings on which the caller reads the results of a granule, `None` for all rings."""

HALO_DEPTH_OF_SOLVE_NH_RESULTS: dict[str, int | None] | None = (
    {
        # With its entry exchange skipped, diffusion reads `vn` on all rings as solve_nonhydro leaves it.
        "vn": None if SKIP_REDUNDANT_ENTRY_EXCHANGE else HALO_DEPTH_READ_BY_CALLER,
        **dict.fromkeys(
            ("vn_traj", "mass_flx_me", "w", "rho", "exner", "theta_v"), HALO_DEPTH_READ_BY_CALLER
        ),
    }
    if decomposition.LIMIT_EXCHANGE_DEPTH
    else None
)
"""Halo rings on which the results of the last substep of solve_nonhydro are read before its next call, `None` for all rings."""
