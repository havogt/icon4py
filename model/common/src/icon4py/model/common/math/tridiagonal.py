# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Solve the tridiagonal systems ``a_k x_{k-1} + b_k x_k + c_k x_{k+1} = d_k`` along the vertical
with the Thomas algorithm: a forward sweep followed by a back substitution, both over the
half-open vertical range ``[vertical_start, vertical_end)``.

The forward sweep carries ``q = -c'`` (ICON's ``z_q``) and ``d'``. Its init state makes the first row independent of its sub-diagonal entry, and the back substitution's init state makes the last row independent of its super-diagonal entry.
"""

import gt4py.next as gtx
from gt4py.next import astype, scan

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _forward_sweep_pass(
    state_kminus1: tuple[wpfloat, wpfloat],
    a: wpfloat,
    b: wpfloat,
    c: wpfloat,
    d: wpfloat,
) -> tuple[wpfloat, wpfloat]:
    q_kminus1, d_prime_kminus1 = state_kminus1
    normalization = wpfloat("1.0") / (b + a * q_kminus1)
    return (wpfloat("0.0") - c) * normalization, (d - a * d_prime_kminus1) * normalization


@gtx.field_operator
def _back_substitution_pass(x_kplus1: wpfloat, q: wpfloat, d_prime: wpfloat) -> wpfloat:
    return d_prime + x_kplus1 * q


@gtx.field_operator
def _forward_sweep_mixed_precision_pass(
    state_kminus1: tuple[vpfloat, wpfloat],
    a: vpfloat,
    b: vpfloat,
    c: vpfloat,
    d: wpfloat,
) -> tuple[vpfloat, wpfloat]:
    """The matrix coefficients and ``q`` in vpfloat, ``d'`` in wpfloat."""
    q_kminus1, d_prime_kminus1 = state_kminus1
    normalization = vpfloat("1.0") / (b + a * q_kminus1)
    q = (vpfloat("0.0") - c) * normalization
    d_prime = (d - astype(a, wpfloat) * d_prime_kminus1) * astype(normalization, wpfloat)
    return q, d_prime


@gtx.field_operator
def _back_substitution_mixed_precision_pass(
    x_kplus1: wpfloat, q: vpfloat, d_prime: wpfloat
) -> wpfloat:
    return d_prime + x_kplus1 * astype(q, wpfloat)


@gtx.field_operator
def _solve_tridiagonal_matrix_forward_sweep_on_half_levels_mixed_precision(  # noqa: PLR0917 [too-many-positional-arguments]
    a: fa.CellKHalfField[vpfloat],
    b: fa.CellKHalfField[vpfloat],
    c: fa.CellKHalfField[vpfloat],
    d: fa.CellKHalfField[wpfloat],
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> tuple[fa.CellKHalfField[vpfloat], fa.CellKHalfField[wpfloat]]:
    return scan(
        _forward_sweep_mixed_precision_pass,
        range=(dims.KHalfDim, vertical_start, vertical_end),
        forward=True,
        init=(vpfloat("0.0"), wpfloat("0.0")),
    )(a, b, c, d)


@gtx.field_operator
def _solve_tridiagonal_matrix_back_substitution_on_half_levels_mixed_precision(
    q: fa.CellKHalfField[vpfloat],
    d_prime: fa.CellKHalfField[wpfloat],
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> fa.CellKHalfField[wpfloat]:
    return scan(
        _back_substitution_mixed_precision_pass,
        range=(dims.KHalfDim, vertical_start, vertical_end),
        forward=False,
        init=wpfloat("0.0"),
    )(q, d_prime)


# one per field type: field operators are not generic over dimensions
@gtx.field_operator(grid_type=gtx.GridType.UNSTRUCTURED)
def _solve_tridiagonal_matrix_on_cells(  # noqa: PLR0917 [too-many-positional-arguments]
    a: fa.CellKField[wpfloat],
    b: fa.CellKField[wpfloat],
    c: fa.CellKField[wpfloat],
    d: fa.CellKField[wpfloat],
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> fa.CellKField[wpfloat]:
    q, d_prime = scan(
        _forward_sweep_pass,
        range=(dims.KDim, vertical_start, vertical_end),
        forward=True,
        init=(wpfloat("0.0"), wpfloat("0.0")),
    )(a, b, c, d)
    return scan(
        _back_substitution_pass,
        range=(dims.KDim, vertical_start, vertical_end),
        forward=False,
        init=wpfloat("0.0"),
    )(q, d_prime)


@gtx.field_operator(grid_type=gtx.GridType.UNSTRUCTURED)
def _solve_tridiagonal_matrix_on_cell_half_levels(  # noqa: PLR0917 [too-many-positional-arguments]
    a: fa.CellKHalfField[wpfloat],
    b: fa.CellKHalfField[wpfloat],
    c: fa.CellKHalfField[wpfloat],
    d: fa.CellKHalfField[wpfloat],
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> fa.CellKHalfField[wpfloat]:
    q, d_prime = scan(
        _forward_sweep_pass,
        range=(dims.KHalfDim, vertical_start, vertical_end),
        forward=True,
        init=(wpfloat("0.0"), wpfloat("0.0")),
    )(a, b, c, d)
    return scan(
        _back_substitution_pass,
        range=(dims.KHalfDim, vertical_start, vertical_end),
        forward=False,
        init=wpfloat("0.0"),
    )(q, d_prime)


@gtx.field_operator(grid_type=gtx.GridType.UNSTRUCTURED)
def _solve_tridiagonal_matrix_on_edges(  # noqa: PLR0917 [too-many-positional-arguments]
    a: fa.EdgeKField[wpfloat],
    b: fa.EdgeKField[wpfloat],
    c: fa.EdgeKField[wpfloat],
    d: fa.EdgeKField[wpfloat],
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> fa.EdgeKField[wpfloat]:
    q, d_prime = scan(
        _forward_sweep_pass,
        range=(dims.KDim, vertical_start, vertical_end),
        forward=True,
        init=(wpfloat("0.0"), wpfloat("0.0")),
    )(a, b, c, d)
    return scan(
        _back_substitution_pass,
        range=(dims.KDim, vertical_start, vertical_end),
        forward=False,
        init=wpfloat("0.0"),
    )(q, d_prime)
