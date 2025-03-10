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
from copy import deepcopy

import pytest

import icon4pytools.liskov.parsing.parse as ts
from icon4pytools.liskov.parsing.utils import (
    extract_directive,
    print_parsed_directive,
    remove_directive_types,
    string_to_bool,
)


def test_extract_directive():
    directives = [
        ts.Imports("IMPORTS()", 1, 1),
        ts.StartCreate("START CREATE()", 3, 4),
    ]

    # Test that only the expected directive is extracted.
    assert extract_directive(directives, ts.Imports) == [directives[0]]
    assert extract_directive(directives, ts.StartCreate) == [directives[1]]


def test_remove_directive():
    directives = [
        ts.Imports("IMPORTS()", 1, 1),
        ts.StartCreate("START CREATE()", 3, 4),
    ]
    new_directives = deepcopy(directives)
    assert remove_directive_types(new_directives, [ts.Imports]) == [directives[1]]


@pytest.mark.parametrize(
    "string, expected",
    [
        ("True", True),
        ("TRUE", True),
        ("false", False),
        ("FALSE", False),
        ("not a boolean", ValueError("Cannot convert 'not a boolean' to a boolean.")),
    ],
)
def test_string_to_bool(string, expected):
    if isinstance(expected, bool):
        assert string_to_bool(string) == expected
    else:
        with pytest.raises(ValueError) as exc_info:
            string_to_bool(string)
        assert str(exc_info.value) == str(expected)


def test_print_parsed_directive():
    directive = ts.Imports("IMPORTS()", 1, 1)
    expected_output = "Directive: IMPORTS(), start line: 1, end line: 1\n"
    assert print_parsed_directive(directive) == expected_output
