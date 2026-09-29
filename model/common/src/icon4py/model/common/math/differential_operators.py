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
edge is the two triangles sharing it (their Fig. 1): vertices 1 and 2 are the edge's end points,
3 and 4 the opposite vertices of the two triangles.
"""

import gt4py.next as gtx
from gt4py.next import neighbor_sum, sqrt

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import C2E, E2C, E2C2V, V2E
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def rbf_vector_at_vertices(
    psi_n: fa.EdgeKField[wpfloat],
    rbf_coeff_1: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], wpfloat],
    rbf_coeff_2: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], wpfloat],
) -> tuple[fa.VertexKField[wpfloat], fa.VertexKField[wpfloat]]:
    """Vector at the vertices reconstructed by RBF from its edge-normal components."""
    u = neighbor_sum(rbf_coeff_1 * psi_n(V2E), axis=dims.V2EDim)
    v = neighbor_sum(rbf_coeff_2 * psi_n(V2E), axis=dims.V2EDim)
    return u, v


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
    flux: fa.EdgeKField[wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
) -> fa.CellKField[wpfloat]:
    """Divergence of an edge-normal flux on cells (Eq 19)."""
    return neighbor_sum(flux(C2E) * geofac_div, axis=dims.C2EDim)


@gtx.field_operator
def div_khalf(
    flux: fa.EdgeKHalfField[wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
) -> fa.CellKHalfField[wpfloat]:
    """Divergence of an edge-normal flux on cells (Eq 19)."""
    return neighbor_sum(flux(C2E) * geofac_div, axis=dims.C2EDim)


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
    u: fa.VertexKField[wpfloat],
    v: fa.VertexKField[wpfloat],
    direction_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    direction_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
) -> tuple[
    fa.EdgeKField[wpfloat],
    fa.EdgeKField[wpfloat],
    fa.EdgeKField[wpfloat],
    fa.EdgeKField[wpfloat],
]:
    """Component of the vertex vector (u, v) along the direction given at each diamond vertex 1..4."""
    psi_1 = (
        u(E2C2V[0]) * direction_x[dims.E2C2VDim(0)] + v(E2C2V[0]) * direction_y[dims.E2C2VDim(0)]
    )
    psi_2 = (
        u(E2C2V[1]) * direction_x[dims.E2C2VDim(1)] + v(E2C2V[1]) * direction_y[dims.E2C2VDim(1)]
    )
    psi_3 = (
        u(E2C2V[2]) * direction_x[dims.E2C2VDim(2)] + v(E2C2V[2]) * direction_y[dims.E2C2VDim(2)]
    )
    psi_4 = (
        u(E2C2V[3]) * direction_x[dims.E2C2VDim(3)] + v(E2C2V[3]) * direction_y[dims.E2C2VDim(3)]
    )
    return psi_1, psi_2, psi_3, psi_4


@gtx.field_operator
def nabla2_diamond(  # noqa: PLR0917 [too-many-positional-arguments]
    psi: fa.EdgeKField[wpfloat],
    psi_1: fa.EdgeKField[wpfloat],
    psi_2: fa.EdgeKField[wpfloat],
    psi_3: fa.EdgeKField[wpfloat],
    psi_4: fa.EdgeKField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    inv_vert_vert_length: fa.EdgeField[wpfloat],
) -> fa.EdgeKField[wpfloat]:
    """Second differences of psi along the two diagonals of the diamond (Eq 36)."""
    return (psi_2 + psi_1 - 2.0 * psi) * inv_primal_edge_length**2 + (
        psi_4 + psi_3 - 2.0 * psi
    ) * inv_vert_vert_length**2


@gtx.field_operator
def nabla2_n(  # noqa: PLR0917 [too-many-positional-arguments]
    vn: fa.EdgeKField[wpfloat],
    rbf_coeff_1: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], wpfloat],
    rbf_coeff_2: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], wpfloat],
    primal_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    primal_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    inv_vert_vert_length: fa.EdgeField[wpfloat],
) -> fa.EdgeKField[wpfloat]:
    """Laplacian of an edge-normal vector component (Eq 36), from the RBF vector at the diamond vertices."""
    u, v = rbf_vector_at_vertices(vn, rbf_coeff_1, rbf_coeff_2)
    vn_1, vn_2, vn_3, vn_4 = components_at_diamond_vertices(
        u, v, primal_normal_vert_x, primal_normal_vert_y
    )
    return nabla2_diamond(vn, vn_1, vn_2, vn_3, vn_4, inv_primal_edge_length, inv_vert_vert_length)


@gtx.field_operator
def horizontal_deformation(  # noqa: PLR0917 [too-many-positional-arguments]
    vn_1: fa.EdgeKField[wpfloat],
    vn_2: fa.EdgeKField[wpfloat],
    vn_3: fa.EdgeKField[wpfloat],
    vn_4: fa.EdgeKField[wpfloat],
    vt_1: fa.EdgeKField[wpfloat],
    vt_2: fa.EdgeKField[wpfloat],
    vt_3: fa.EdgeKField[wpfloat],
    vt_4: fa.EdgeKField[wpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    inv_vert_vert_length: fa.EdgeField[wpfloat],
) -> fa.EdgeKField[wpfloat]:
    """Magnitude of the tension and shear deformation on the diamond (the root in Eq 37)."""
    # Vertices 1 and 2 follow the edge tangent, which points either way along the edge.
    inv_l_p = tangent_orientation * inv_primal_edge_length
    tension = (vn_4 - vn_3) * inv_vert_vert_length - (vt_2 - vt_1) * inv_l_p
    shear = (vn_2 - vn_1) * inv_l_p + (vt_4 - vt_3) * inv_vert_vert_length
    return sqrt(tension**2 + shear**2)
