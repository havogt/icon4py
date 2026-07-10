# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import contextlib

import gt4py.next as gtx
import numpy as np
import pytest
from _pytest.outcomes import Failed

from icon4py.model.common import dimension as dims
from icon4py.model.testing import verification


def _field(array) -> gtx.Field:
    return gtx.as_field((dims.CellDim, dims.KDim), np.asarray(array, dtype=float))


class _SubtestsRecorder:
    """Mimics the `pytest-subtests` fixture: records each subtest's outcome."""

    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str]] = []

    @contextlib.contextmanager
    def test(self, msg=None, **kwargs):
        field = kwargs.get("field")
        try:
            yield
        except (Failed, AssertionError) as exc:
            self.results.append((field, False, str(exc)))
        else:
            self.results.append((field, True, ""))

    @property
    def failed(self) -> list[str]:
        return [name for name, ok, _ in self.results if not ok]

    @property
    def passed(self) -> list[str]:
        return [name for name, ok, _ in self.results if ok]

    def message(self, field: str) -> str:
        return next(msg for name, _, msg in self.results if name == field)


# --- compare_field -----------------------------------------------------------


def test_compare_field_identical():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = verification.compare_field(a, a.copy(), atol=0.0, rtol=1e-12, equal_nan=False)
    assert result.ok
    assert result.max_abs_diff == 0.0
    assert result.n_mismatch == 0


def test_compare_field_atol():
    a = np.array([[1.0, 2.0]])
    b = a + 1e-9
    assert verification.compare_field(a, b, atol=1e-8, rtol=0.0, equal_nan=False).ok
    assert not verification.compare_field(a, b, atol=1e-12, rtol=0.0, equal_nan=False).ok


def test_compare_field_rtol_scales_with_magnitude():
    a = np.array([[1.0, 1000.0]])
    b = np.array([[1.0 + 1e-6, 1000.0 + 1e-3]])  # both off by rtol=1e-6 relative
    assert verification.compare_field(a, b, atol=0.0, rtol=2e-6, equal_nan=False).ok
    assert not verification.compare_field(a, b, atol=0.0, rtol=1e-9, equal_nan=False).ok


def test_compare_field_reports_diff_and_count():
    a = np.array([[1.0, 1.0, 1.0, 1.0]])
    b = np.array([[1.0, 1.0, 1.0, 3.0]])
    result = verification.compare_field(a, b, atol=0.0, rtol=1e-12, equal_nan=False)
    assert not result.ok
    assert result.max_abs_diff == pytest.approx(2.0)
    assert result.n_mismatch == 1
    assert result.size == 4
    assert "25.00%" in result.message
    assert "index (0, 3)" in result.message


def test_compare_field_shape_mismatch():
    result = verification.compare_field(
        np.zeros((2, 2)), np.zeros((2, 3)), atol=0.0, rtol=0.0, equal_nan=False
    )
    assert not result.ok
    assert "shape mismatch" in result.message


def test_compare_field_equal_nan():
    a = np.array([[1.0, np.nan]])
    b = np.array([[1.0, np.nan]])
    assert verification.compare_field(a, b, atol=0.0, rtol=0.0, equal_nan=True).ok
    assert not verification.compare_field(a, b, atol=0.0, rtol=0.0, equal_nan=False).ok


def test_compare_field_empty():
    result = verification.compare_field(
        np.zeros((0, 3)), np.zeros((0, 3)), atol=0.0, rtol=0.0, equal_nan=False
    )
    assert result.ok


def test_compare_field_bool():
    a = np.array([[True, False, True]])
    assert verification.compare_field(a, a.copy(), atol=0.0, rtol=0.0, equal_nan=False).ok
    b = np.array([[True, True, True]])
    result = verification.compare_field(a, b, atol=0.0, rtol=0.0, equal_nan=False)
    assert not result.ok
    assert result.n_mismatch == 1


# --- Check.tolerance ---------------------------------------------------------


def test_check_tolerance_falls_back_to_defaults():
    assert verification.Check("x").tolerance(verification.DATA_DEFAULT) == verification.DATA_DEFAULT


def test_check_tolerance_per_field_override():
    tol = verification.Check("x", atol=1e-3).tolerance(verification.DATA_DEFAULT)
    assert tol.atol == 1e-3
    assert tol.rtol == verification.DATA_DEFAULT.rtol  # untouched fields keep the default


# --- check_fields ------------------------------------------------------------


def test_check_fields_all_pass():
    subtests = _SubtestsRecorder()
    fields = {"out": _field([[1.0, 2.0]]), "other": _field([[3.0, 4.0]])}
    verification.check_fields(subtests, fields, dict(fields), (verification.Check("out"), "other"))
    assert subtests.passed == ["out", "other"]
    assert subtests.failed == []


def test_check_fields_collects_all_failures():
    subtests = _SubtestsRecorder()
    actual = {"a": _field([[1.0]]), "b": _field([[9.0]]), "c": _field([[9.0]])}
    expected = {"a": _field([[1.0]]), "b": _field([[2.0]]), "c": _field([[3.0]])}
    verification.check_fields(subtests, actual, expected, ("a", "b", "c"))
    assert subtests.passed == ["a"]
    assert subtests.failed == ["b", "c"]  # both reported, not just the first


def test_check_fields_ref_override():
    subtests = _SubtestsRecorder()
    verification.check_fields(
        subtests,
        {"computed_name": _field([[1.0, 2.0]])},
        {"reference_name": _field([[1.0, 2.0]])},
        (verification.Check("computed_name", ref="reference_name"),),
    )
    assert subtests.failed == []


def test_check_fields_slicing():
    subtests = _SubtestsRecorder()
    # reference holds an extra leading column the computed field does not.
    verification.check_fields(
        subtests,
        {"out": _field([[1.0, 2.0]])},
        {"out": _field([[7.0, 1.0, 2.0]])},
        (verification.Check("out", refslice=(slice(None), slice(1, None))),),
    )
    assert subtests.failed == []


def test_check_fields_per_field_tolerance_honored():
    subtests = _SubtestsRecorder()
    actual = {"loose": _field([[1.0]]), "tight": _field([[1.0]])}
    expected = {"loose": _field([[1.0 + 1e-9]]), "tight": _field([[1.0 + 1e-9]])}
    verification.check_fields(
        subtests,
        actual,
        expected,
        (verification.Check("loose", atol=1e-8), verification.Check("tight")),
    )
    assert subtests.passed == ["loose"]  # passes under its own atol
    assert subtests.failed == ["tight"]  # fails under the strict default


def test_check_fields_missing_computed_key():
    subtests = _SubtestsRecorder()
    verification.check_fields(subtests, {}, {"out": _field([[1.0]])}, ("out",))
    assert subtests.failed == ["out"]
    assert "missing from the computed" in subtests.message("out")


def test_check_fields_missing_reference_key():
    subtests = _SubtestsRecorder()
    verification.check_fields(subtests, {"out": _field([[1.0]])}, {}, ("out",))
    assert subtests.failed == ["out"]
    assert "missing from the expected" in subtests.message("out")


def test_check_fields_stencil_default_is_looser():
    subtests = _SubtestsRecorder()
    actual = {"x": _field([[1.0]])}
    expected = {"x": _field([[1.0 + 1e-7]])}  # within stencil rtol 3e-6, outside data rtol 1e-12
    verification.check_fields(
        subtests, actual, expected, ("x",), defaults=verification.DATA_DEFAULT
    )
    verification.check_fields(
        subtests, actual, expected, ("x",), defaults=verification.STENCIL_DEFAULT
    )
    assert subtests.results[0][1] is False  # data default fails
    assert subtests.results[1][1] is True  # stencil default passes


def test_check_fields_with_real_subtests_fixture(subtests):
    fields = {"out": _field([[1.0, 2.0]]), "other": _field([[3.0, 4.0]])}
    verification.check_fields(subtests, fields, dict(fields), ("out", "other"))
