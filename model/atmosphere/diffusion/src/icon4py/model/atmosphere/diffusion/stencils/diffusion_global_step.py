# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""
One horizontal diffusion step, following Zängl et al. (2015) §2.5, https://doi.org/10.1002/qj.2378.

The Smagorinsky coefficient `kh` is K_h Δt / a_e (Eq 37).
"""

from typing import NamedTuple

import gt4py.next as gtx
from gt4py.next import max_over, maximum, minimum, neighbor_sum, where
from gt4py.next.experimental import as_offset, concat_where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import C2E2C, E2C, Koff
from icon4py.model.common.math.differential_operators import (
    DiamondDirection,
    DiamondLengths,
    RbfVectorCoefficients,
    components_at_diamond_vertices,
    div,
    grad_n,
    horizontal_deformation,
    nabla2_khalf,
    nabla2_n,
    rbf_vector_at_vertices,
)
from icon4py.model.common.type_alias import wpfloat


class DiffusionCoefficients(NamedTuple):
    #: f_s(z) of Eq 37
    f_s: fa.KField[wpfloat]
    #: offset subtracted from K_h Δt / a_e, 0.75 k_4 a_e in the paper
    kh_offset: wpfloat
    #: upper bound of K_h Δt / a_e
    kh_limit: fa.KField[wpfloat]
    #: k_4 Δt of Eq 39
    k4_dt: fa.KField[wpfloat]
    #: k_w Δt of Eq 40
    kw_dt: wpfloat
    #: coefficient of the second-order w diffusion in the upper damping layer
    k2w_damping_layer: fa.KHalfField[wpfloat]


class SteepPointInterpolation(NamedTuple):
    """Vertical interpolation of the C2E2C neighbours to a cell's height, zero off the steep points."""

    diffusion_coefficient: fa.CellKField[wpfloat]
    vertical_offset: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim, dims.KDim], gtx.int32]
    weight: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim, dims.KDim], wpfloat]


class CellNeighbourWeights(NamedTuple):
    center: fa.CellField[wpfloat]
    neighbours: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], wpfloat]


@gtx.field_operator
def cold_pool_diffusion_coefficient(
    theta_v: fa.CellKField[wpfloat],
    theta_ref_mc: fa.CellKField[wpfloat],
    thresh_tdiff: wpfloat,
    smallest_coefficient: wpfloat,
) -> fa.CellKField[wpfloat]:
    """Extra diffusion coefficient where a cell is much colder than its neighbours."""
    tdiff = theta_v - neighbor_sum(theta_v(C2E2C), axis=dims.C2E2CDim) / 3.0
    trefdiff = theta_ref_mc - neighbor_sum(theta_ref_mc(C2E2C), axis=dims.C2E2CDim) / 3.0
    return where(
        ((tdiff - trefdiff < thresh_tdiff) & (trefdiff < 0.0))
        | (tdiff - trefdiff < 1.5 * thresh_tdiff),
        5.0e-4 * (thresh_tdiff - tdiff + trefdiff),
        smallest_coefficient,
    )


@gtx.field_operator
def nabla2_at_constant_height(
    psi: fa.CellKField[wpfloat],
    steep: SteepPointInterpolation,
    n2s: CellNeighbourWeights,
) -> fa.CellKField[wpfloat]:
    """Laplacian with the neighbours interpolated vertically to the cell's height."""
    psi_1 = steep.weight[dims.C2E2CDim(0)] * psi(C2E2C[0])(
        as_offset(Koff, steep.vertical_offset[dims.C2E2CDim(0)])
    ) + (1.0 - steep.weight[dims.C2E2CDim(0)]) * psi(C2E2C[0])(
        as_offset(Koff, steep.vertical_offset[dims.C2E2CDim(0)] + 1)
    )
    psi_2 = steep.weight[dims.C2E2CDim(1)] * psi(C2E2C[1])(
        as_offset(Koff, steep.vertical_offset[dims.C2E2CDim(1)])
    ) + (1.0 - steep.weight[dims.C2E2CDim(1)]) * psi(C2E2C[1])(
        as_offset(Koff, steep.vertical_offset[dims.C2E2CDim(1)] + 1)
    )
    psi_3 = steep.weight[dims.C2E2CDim(2)] * psi(C2E2C[2])(
        as_offset(Koff, steep.vertical_offset[dims.C2E2CDim(2)])
    ) + (1.0 - steep.weight[dims.C2E2CDim(2)]) * psi(C2E2C[2])(
        as_offset(Koff, steep.vertical_offset[dims.C2E2CDim(2)] + 1)
    )
    return (
        n2s.center * psi
        + n2s.neighbours[dims.C2E2CDim(0)] * psi_1
        + n2s.neighbours[dims.C2E2CDim(1)] * psi_2
        + n2s.neighbours[dims.C2E2CDim(2)] * psi_3
    )


@gtx.field_operator
def exner_at_constant_density(
    exner: fa.CellKField[wpfloat],
    theta_v_new: fa.CellKField[wpfloat],
    theta_v: fa.CellKField[wpfloat],
    rd_o_cvd: wpfloat,
) -> fa.CellKField[wpfloat]:
    """Exner pressure after a change of theta_v at fixed density, linearised."""
    return exner * (1.0 + rd_o_cvd * (theta_v_new / theta_v - 1.0))


@gtx.field_operator
def smagorinsky_coefficient(
    vn: fa.EdgeKField[wpfloat],
    coefficients: DiffusionCoefficients,
    rbf_coeff: RbfVectorCoefficients,
    primal_normal: DiamondDirection,
    dual_normal: DiamondDirection,
    lengths: DiamondLengths,
    tangent_orientation: fa.EdgeField[wpfloat],
    dtime: wpfloat,
) -> fa.EdgeKField[wpfloat]:
    """K_h Δt / a_e of Eq 37, less the offset and bounded by the stability limit."""
    vertex_wind = rbf_vector_at_vertices(vn, rbf_coeff)
    vn1, vn2, vn3, vn4 = components_at_diamond_vertices(vertex_wind, primal_normal)
    vt1, vt2, vt3, vt4 = components_at_diamond_vertices(vertex_wind, dual_normal)
    deformation = horizontal_deformation(
        vn1, vn2, vn3, vn4, vt1, vt2, vt3, vt4, tangent_orientation, lengths
    )
    return minimum(
        maximum(0.0, coefficients.f_s * dtime * deformation - coefficients.kh_offset),
        coefficients.kh_limit,
    )


@gtx.field_operator
def diffused_vn(
    vn: fa.EdgeKField[wpfloat],
    kh: fa.EdgeKField[wpfloat],
    coefficients: DiffusionCoefficients,
    rbf_coeff: RbfVectorCoefficients,
    primal_normal: DiamondDirection,
    lengths: DiamondLengths,
    edge_area: fa.EdgeField[wpfloat],
) -> fa.EdgeKField[wpfloat]:
    """vn after the second-order Smagorinsky and fourth-order background diffusion (Eqs 35, 36 and 39)."""
    nabla2_vn = nabla2_n(vn, rbf_coeff, primal_normal, lengths)
    # The factor 4 of Eq 35 is applied inside both Laplacians of the fourth-order term, as in ICON.
    return (
        vn
        + 4.0 * edge_area * kh * nabla2_vn
        - 16.0
        * coefficients.k4_dt
        * edge_area**2
        * nabla2_n(nabla2_vn, rbf_coeff, primal_normal, lengths)
    )


@gtx.field_operator
def diffused_w(
    w: fa.CellKHalfField[wpfloat],
    coefficients: DiffusionCoefficients,
    cell_area: fa.CellField[wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    nrdmax: gtx.int32,
    num_levels: gtx.int32,
) -> fa.CellKHalfField[wpfloat]:
    """w after the fourth-order diffusion (Eqs 40 and 41) and a second-order one in the upper damping layer, the surface level kept."""
    nabla2_w = nabla2_khalf(w, inv_dual_edge_length, geofac_div)
    return concat_where(
        dims.KHalfDim < num_levels,
        w
        - coefficients.kw_dt
        * cell_area**2
        * nabla2_khalf(nabla2_w, inv_dual_edge_length, geofac_div)
        + concat_where(
            (dims.KHalfDim >= 1) & (dims.KHalfDim < nrdmax),
            coefficients.k2w_damping_layer * cell_area * nabla2_w,
            0.0,
        ),
        w,
    )


@gtx.field_operator
def diffused_theta_v_and_exner(
    theta_v: fa.CellKField[wpfloat],
    exner: fa.CellKField[wpfloat],
    kh: fa.EdgeKField[wpfloat],
    theta_ref_mc: fa.CellKField[wpfloat],
    cell_area: fa.CellField[wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    steep: SteepPointInterpolation,
    n2s: CellNeighbourWeights,
    thresh_tdiff: wpfloat,
    smallest_coefficient: wpfloat,
    rd_o_cvd: wpfloat,
    num_levels: gtx.int32,
    apply_zdiffusion_t: bool,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """theta_v after the Smagorinsky diffusion (Eq 38), the coefficient raised in cold pools on the two lowest levels, and exner at fixed density."""
    cold_pool = cold_pool_diffusion_coefficient(
        theta_v, theta_ref_mc, thresh_tdiff, smallest_coefficient
    )
    kh_theta = concat_where(
        num_levels - 2 <= dims.KDim,
        maximum(kh, max_over(cold_pool(E2C), axis=dims.E2CDim)),
        kh,
    )
    tendency = div(kh_theta * grad_n(theta_v, inv_dual_edge_length), geofac_div)
    if apply_zdiffusion_t:
        tendency = where(
            steep.diffusion_coefficient != 0.0,
            tendency + steep.diffusion_coefficient * nabla2_at_constant_height(theta_v, steep, n2s),
            tendency,
        )
    theta_v_new = theta_v + cell_area * tendency
    return theta_v_new, exner_at_constant_density(exner, theta_v_new, theta_v, rd_o_cvd)


@gtx.field_operator
def _diffusion_global_step(
    vn: fa.EdgeKField[wpfloat],
    w: fa.CellKHalfField[wpfloat],
    theta_v: fa.CellKField[wpfloat],
    exner: fa.CellKField[wpfloat],
    coefficients: DiffusionCoefficients,
    rbf_coeff: RbfVectorCoefficients,
    primal_normal: DiamondDirection,
    dual_normal: DiamondDirection,
    lengths: DiamondLengths,
    tangent_orientation: fa.EdgeField[wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    edge_area: fa.EdgeField[wpfloat],
    cell_area: fa.CellField[wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    n2s: CellNeighbourWeights,
    theta_ref_mc: fa.CellKField[wpfloat],
    steep: SteepPointInterpolation,
    dtime: wpfloat,
    thresh_tdiff: wpfloat,
    smallest_coefficient: wpfloat,
    rd_o_cvd: wpfloat,
    nrdmax: gtx.int32,
    num_levels: gtx.int32,
    apply_to_temperature: bool,
    apply_zdiffusion_t: bool,
) -> tuple[
    fa.EdgeKField[wpfloat],
    fa.CellKHalfField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
]:
    kh = smagorinsky_coefficient(
        vn, coefficients, rbf_coeff, primal_normal, dual_normal, lengths, tangent_orientation, dtime
    )
    vn_new = diffused_vn(vn, kh, coefficients, rbf_coeff, primal_normal, lengths, edge_area)
    w_new = diffused_w(
        w, coefficients, cell_area, inv_dual_edge_length, geofac_div, nrdmax, num_levels
    )
    theta_v_new, exner_new = (
        diffused_theta_v_and_exner(
            theta_v,
            exner,
            kh,
            theta_ref_mc,
            cell_area,
            inv_dual_edge_length,
            geofac_div,
            steep,
            n2s,
            thresh_tdiff,
            smallest_coefficient,
            rd_o_cvd,
            num_levels,
            apply_zdiffusion_t,
        )
        if apply_to_temperature
        else (theta_v, exner)
    )
    return vn_new, w_new, theta_v_new, exner_new
