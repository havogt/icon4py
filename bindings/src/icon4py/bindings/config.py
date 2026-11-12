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
