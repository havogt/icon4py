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
from typing import Final

from packaging import version as pkg_version


__all__ = [
    "__author__",
    "__copyright__",
    "__license__",
    "__version__",
    "__version_info__",
]


__author__: Final = "ETH Zurich and individual contributors"
__copyright__: Final = "Copyright (c) 2014-2022 ETH Zurich"
__license__: Final = "GPL-3.0-or-later"


__version__: Final = "0.0.6"
__version_info__: Final = pkg_version.parse(__version__)
