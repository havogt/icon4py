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

from gt4py.next.common import Dimension
from gt4py.next.type_system.type_specifications import FieldType, ScalarKind, ScalarType
from gt4py.next.type_system.type_translation import from_type_hint


def parse_annotation(annotation) -> tuple[list[Dimension], ScalarKind]:
    type_spec = from_type_hint(annotation)

    if isinstance(type_spec, ScalarType):
        return [], type_spec.kind
    elif isinstance(type_spec, FieldType):
        return type_spec.dims, type_spec.dtype.kind
    else:
        raise ValueError(f"Unsupported type specification: {type_spec}")
