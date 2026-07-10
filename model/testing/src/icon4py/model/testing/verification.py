# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Declarative per-field verification.

A test declares the fields to verify as a sequence of `Check` specifications and
hands the computed and reference fields to `check_fields`. Every field is
compared independently and reported as its own pytest subtest, so all mismatches
in a test surface together instead of aborting on the first one.

`compare_field` is the pure, framework-independent comparison and can be used on
its own. The tolerance formula matches `numpy.isclose` (and hence
`test_utils.dallclose`): a value passes when ``|actual - desired| <= atol + rtol
* |desired|``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np
import pytest


@dataclasses.dataclass(frozen=True)
class Tolerance:
    atol: float = 0.0
    rtol: float = 1.0e-12
    equal_nan: bool = False


#: Default policy for serialbox data tests (matches `test_utils.dallclose`).
DATA_DEFAULT = Tolerance(atol=0.0, rtol=1.0e-12, equal_nan=False)
#: Default policy for stencil tests (matches the tolerance hardcoded in `StencilTest`).
STENCIL_DEFAULT = Tolerance(atol=0.0, rtol=3.0e-6, equal_nan=True)


@dataclasses.dataclass(frozen=True)
class Check:
    """Specification of a single field comparison.

    `name` is the key into the computed-fields mapping. `ref` overrides the key
    into the reference mapping when it differs. Per-field `atol`/`rtol`/
    `equal_nan` override the policy default when not `None`. `refslice`/`gtslice`
    are applied to the reference and computed arrays respectively (e.g. to trim a
    halo present in only one of them).
    """

    name: str
    ref: str | None = None
    atol: float | None = None
    rtol: float | None = None
    equal_nan: bool | None = None
    refslice: tuple[slice, ...] = (slice(None),)
    gtslice: tuple[slice, ...] = (slice(None),)

    def tolerance(self, defaults: Tolerance) -> Tolerance:
        return Tolerance(
            atol=defaults.atol if self.atol is None else self.atol,
            rtol=defaults.rtol if self.rtol is None else self.rtol,
            equal_nan=defaults.equal_nan if self.equal_nan is None else self.equal_nan,
        )


@dataclasses.dataclass(frozen=True)
class FieldResult:
    ok: bool
    max_abs_diff: float
    n_mismatch: int
    size: int
    message: str = ""


class SupportsSubtests(Protocol):
    """The subset of the `pytest-subtests` fixture that `check_fields` uses."""

    def test(self, msg: str | None = ..., **kwargs: Any) -> Any: ...


def _as_numpy(field: Any) -> np.ndarray:
    return field.asnumpy() if hasattr(field, "asnumpy") else np.asarray(field)


def compare_field(
    actual: np.ndarray,
    desired: np.ndarray,
    *,
    atol: float,
    rtol: float,
    equal_nan: bool,
) -> FieldResult:
    actual = np.asarray(actual)
    desired = np.asarray(desired)
    if actual.shape != desired.shape:
        return FieldResult(
            ok=False,
            max_abs_diff=float("nan"),
            n_mismatch=actual.size,
            size=actual.size,
            message=f"shape mismatch: computed {actual.shape}, reference {desired.shape}.",
        )
    if actual.size == 0:
        return FieldResult(ok=True, max_abs_diff=0.0, n_mismatch=0, size=0)

    close = np.isclose(actual, desired, rtol=rtol, atol=atol, equal_nan=equal_nan)
    diff = np.abs(actual.astype(np.float64) - desired.astype(np.float64))
    finite = np.isfinite(diff)
    max_abs_diff = float(diff[finite].max()) if finite.any() else float("nan")
    n_mismatch = int((~close).sum())
    if n_mismatch == 0:
        return FieldResult(ok=True, max_abs_diff=max_abs_diff, n_mismatch=0, size=actual.size)

    worst = np.unravel_index(int(np.argmax(np.where(finite, diff, -np.inf))), diff.shape)
    pct = 100.0 * n_mismatch / actual.size
    message = (
        f"{n_mismatch}/{actual.size} entries ({pct:.2f}%) exceed atol={atol:g}, rtol={rtol:g}; "
        f"max abs diff {max_abs_diff:.3e} at index {tuple(int(i) for i in worst)} "
        f"(computed {float(actual[worst]):.6e}, reference {float(desired[worst]):.6e})."
    )
    return FieldResult(
        ok=False,
        max_abs_diff=max_abs_diff,
        n_mismatch=n_mismatch,
        size=actual.size,
        message=message,
    )


def check_fields(
    subtests: SupportsSubtests,
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    checks: Sequence[str | Check],
    *,
    defaults: Tolerance = DATA_DEFAULT,
) -> None:
    """Verify every check as an independent subtest, collecting all mismatches.

    `actual` maps a field name to the computed field, `expected` maps a (possibly
    different, see `Check.ref`) name to the reference field. Both accept gt4py
    fields or anything `numpy.asarray` understands.
    """
    for entry in checks:
        check = entry if isinstance(entry, Check) else Check(entry)
        ref_key = check.ref or check.name
        with subtests.test(field=check.name):
            if check.name not in actual:
                pytest.fail(
                    f"Field '{check.name}' is missing from the computed fields: {sorted(actual)}.",
                    pytrace=False,
                )
            if ref_key not in expected:
                pytest.fail(
                    f"Reference '{ref_key}' is missing from the expected fields: {sorted(expected)}.",
                    pytrace=False,
                )
            tolerance = check.tolerance(defaults)
            result = compare_field(
                _as_numpy(actual[check.name])[check.gtslice],
                _as_numpy(expected[ref_key])[check.refslice],
                atol=tolerance.atol,
                rtol=tolerance.rtol,
                equal_nan=tolerance.equal_nan,
            )
            if not result.ok:
                pytest.fail(f"'{check.name}': {result.message}", pytrace=False)
