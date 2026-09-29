# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""
Horizontal differential operators of the ICON triangular grid.

Equation numbers refer to Zängl et al. (2015), https://doi.org/10.1002/qj.2378. The diamond of an
edge is the two triangles sharing it (their Fig. 1): vertices 1 and 2 are adjacent to the edge,
3 and 4 are the outer vertices.
"""

from typing import NamedTuple

import gt4py.next as gtx
from gt4py.next import neighbor_sum, sqrt

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import C2E, E2C, E2C2V, V2E
from icon4py.model.common.type_alias import wpfloat


class VertexVector(NamedTuple):
    u: fa.VertexKField[wpfloat]
    v: fa.VertexKField[wpfloat]


class RbfVectorCoefficients(NamedTuple):
    """Weights of the edge-normal components in the RBF reconstruction of (u, v) at a vertex."""

    u: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], wpfloat]
    v: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], wpfloat]


class DiamondDirection(NamedTuple):
    """A unit vector given at each of the four diamond vertices."""

    x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat]
    y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat]


class DiamondLengths(NamedTuple):
    """Inverse lengths of the diamond diagonals: l_p between vertices 1 and 2, l_vv between 3 and 4."""

    inv_l_p: fa.EdgeField[wpfloat]
    inv_l_vv: fa.EdgeField[wpfloat]


@gtx.field_operator
def rbf_vector_at_vertices(
    psi_n: fa.EdgeKField[wpfloat], coeff: RbfVectorCoefficients
) -> VertexVector:
    """Vector at the vertices reconstructed by RBF from its edge-normal components."""
    return VertexVector(
        u=neighbor_sum(coeff.u * psi_n(V2E), axis=dims.V2EDim),
        v=neighbor_sum(coeff.v * psi_n(V2E), axis=dims.V2EDim),
    )


@gtx.field_operator
def grad_n(
    psi: fa.CellKField[wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
) -> fa.EdgeKField[wpfloat]:
    """Normal gradient Δψ/Δn across the edge."""
    return (psi(E2C[1]) - psi(E2C[0])) * inv_dual_edge_length


@gtx.field_operator
def grad_n_khalf(
    psi: fa.CellKHalfField[wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
) -> fa.EdgeKHalfField[wpfloat]:
    """Normal gradient Δψ/Δn across the edge."""
    return (psi(E2C[1]) - psi(E2C[0])) * inv_dual_edge_length


@gtx.field_operator
def div(
    phi_e: fa.EdgeKField[wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
) -> fa.CellKField[wpfloat]:
    """Divergence of an edge-normal flux on cells (Eq 19)."""
    return neighbor_sum(phi_e(C2E) * geofac_div, axis=dims.C2EDim)


@gtx.field_operator
def div_khalf(
    phi_e: fa.EdgeKHalfField[wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
) -> fa.CellKHalfField[wpfloat]:
    """Divergence of an edge-normal flux on cells (Eq 19)."""
    return neighbor_sum(phi_e(C2E) * geofac_div, axis=dims.C2EDim)


@gtx.field_operator
def nabla2_khalf(
    psi: fa.CellKHalfField[wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
) -> fa.CellKHalfField[wpfloat]:
    """Laplacian of a scalar on cells, the divergence of its normal gradient (Eq 41)."""
    return div_khalf(grad_n_khalf(psi, inv_dual_edge_length), geofac_div)


@gtx.field_operator
def components_at_diamond_vertices(
    vector: VertexVector, direction: DiamondDirection
) -> tuple[
    fa.EdgeKField[wpfloat],
    fa.EdgeKField[wpfloat],
    fa.EdgeKField[wpfloat],
    fa.EdgeKField[wpfloat],
]:
    """Component of the vertex vector along the direction given at each diamond vertex 1..4."""
    return (
        vector.u(E2C2V[0]) * direction.x[dims.E2C2VDim(0)]
        + vector.v(E2C2V[0]) * direction.y[dims.E2C2VDim(0)],
        vector.u(E2C2V[1]) * direction.x[dims.E2C2VDim(1)]
        + vector.v(E2C2V[1]) * direction.y[dims.E2C2VDim(1)],
        vector.u(E2C2V[2]) * direction.x[dims.E2C2VDim(2)]
        + vector.v(E2C2V[2]) * direction.y[dims.E2C2VDim(2)],
        vector.u(E2C2V[3]) * direction.x[dims.E2C2VDim(3)]
        + vector.v(E2C2V[3]) * direction.y[dims.E2C2VDim(3)],
    )


@gtx.field_operator
def nabla2_diamond(  # noqa: PLR0917 [too-many-positional-arguments]
    psi: fa.EdgeKField[wpfloat],
    psi_1: fa.EdgeKField[wpfloat],
    psi_2: fa.EdgeKField[wpfloat],
    psi_3: fa.EdgeKField[wpfloat],
    psi_4: fa.EdgeKField[wpfloat],
    lengths: DiamondLengths,
) -> fa.EdgeKField[wpfloat]:
    """Second differences of psi along the two diagonals of the diamond (Eq 36)."""
    return (psi_2 + psi_1 - 2.0 * psi) * lengths.inv_l_p**2 + (
        psi_4 + psi_3 - 2.0 * psi
    ) * lengths.inv_l_vv**2


@gtx.field_operator
def nabla2_n(
    vn: fa.EdgeKField[wpfloat],
    rbf_coeff: RbfVectorCoefficients,
    primal_normal: DiamondDirection,
    lengths: DiamondLengths,
) -> fa.EdgeKField[wpfloat]:
    """Laplacian of an edge-normal vector component (Eq 36), from the RBF vector at the diamond vertices."""
    vn1, vn2, vn3, vn4 = components_at_diamond_vertices(
        rbf_vector_at_vertices(vn, rbf_coeff), primal_normal
    )
    return nabla2_diamond(vn, vn1, vn2, vn3, vn4, lengths)


@gtx.field_operator
def horizontal_deformation(  # noqa: PLR0917 [too-many-positional-arguments]
    vn1: fa.EdgeKField[wpfloat],
    vn2: fa.EdgeKField[wpfloat],
    vn3: fa.EdgeKField[wpfloat],
    vn4: fa.EdgeKField[wpfloat],
    vt1: fa.EdgeKField[wpfloat],
    vt2: fa.EdgeKField[wpfloat],
    vt3: fa.EdgeKField[wpfloat],
    vt4: fa.EdgeKField[wpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
    lengths: DiamondLengths,
) -> fa.EdgeKField[wpfloat]:
    """Magnitude of the tension and shear deformation on the diamond (the root in Eq 37)."""
    # Vertices 1 and 2 follow the edge tangent, which points either way along the edge.
    tension = (vn4 - vn3) * lengths.inv_l_vv - (vt2 - vt1) * tangent_orientation * lengths.inv_l_p
    shear = (vn2 - vn1) * tangent_orientation * lengths.inv_l_p + (vt4 - vt3) * lengths.inv_l_vv
    return sqrt(tension**2 + shear**2)
