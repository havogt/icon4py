# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""
One Godunov-split tracer advection step on a grid without lateral boundaries, as one field operator.

Embedded execution evaluates every expression on the largest domain its inputs allow, so a vertical
shift of a full-level field narrows the column. Every shift that would leave the column below is
therefore taken from a `concat_where` whose other branch is a zero field spanning the full column;
where the stencils this step is built from would have read outside the column, the value is masked
away, so the zero never reaches a result.
"""

import gt4py.next as gtx
from gt4py.next import abs, astype, floor, minimum, where  # noqa: A004
from gt4py.next.experimental import concat_where

from icon4py.model.atmosphere.tracer_advection.stencils.apply_density_increment import (
    _apply_density_increment,
)
from icon4py.model.atmosphere.tracer_advection.stencils.apply_positive_definite_horizontal_multiplicative_flux_factor import (
    _apply_positive_definite_horizontal_multiplicative_flux_factor,
)
from icon4py.model.atmosphere.tracer_advection.stencils.compute_barycentric_backtrajectory_alt import (
    _compute_barycentric_backtrajectory_alt,
)
from icon4py.model.atmosphere.tracer_advection.stencils.compute_horizontal_tracer_flux_from_linear_coefficients_alt import (
    _compute_horizontal_tracer_flux_from_linear_coefficients_alt,
)
from icon4py.model.atmosphere.tracer_advection.stencils.compute_horizontal_tracer_flux_upwind import (
    _compute_horizontal_tracer_flux_upwind,
)
from icon4py.model.atmosphere.tracer_advection.stencils.compute_positive_definite_horizontal_multiplicative_flux_factor import (
    _compute_positive_definite_horizontal_multiplicative_flux_factor,
)
from icon4py.model.atmosphere.tracer_advection.stencils.compute_ppm4gpu_parabola_coefficients import (
    _compute_ppm4gpu_parabola_coefficients,
)
from icon4py.model.atmosphere.tracer_advection.stencils.compute_ppm_quadratic_face_values import (
    _compute_ppm_quadratic_face_values,
)
from icon4py.model.atmosphere.tracer_advection.stencils.compute_ppm_quartic_face_values import (
    _compute_ppm_quartic_face_values,
)
from icon4py.model.atmosphere.tracer_advection.stencils.compute_ppm_slope import _compute_ppm_slope
from icon4py.model.atmosphere.tracer_advection.stencils.compute_vertical_parabola_limiter_condition import (
    _compute_vertical_parabola_limiter_condition,
)
from icon4py.model.atmosphere.tracer_advection.stencils.compute_vertical_tracer_flux_upwind import (
    _compute_vertical_tracer_flux_upwind,
)
from icon4py.model.atmosphere.tracer_advection.stencils.integrate_tracer_horizontally import (
    _integrate_tracer_horizontally,
)
from icon4py.model.atmosphere.tracer_advection.stencils.integrate_tracer_vertically import (
    _integrate_tracer_vertically,
)
from icon4py.model.atmosphere.tracer_advection.stencils.limit_vertical_parabola_semi_monotonically import (
    _limit_vertical_parabola_semi_monotonically,
)
from icon4py.model.atmosphere.tracer_advection.stencils.reconstruct_linear_coefficients_svd import (
    _reconstruct_linear_coefficients_svd,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.interpolation.stencils.compute_tangential_wind import (
    _compute_tangential_wind,
)
from icon4py.model.common.math.vertical_operations import (
    _copy_half_level_above_to_model_levels_on_cells,
    _copy_half_level_below_to_model_levels_on_cells,
)
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _below_0(
    f: fa.CellKField[ta.wpfloat], zero: fa.CellKHalfField[ta.wpfloat], nlev: gtx.int32
) -> fa.CellKHalfField[ta.wpfloat]:
    return concat_where(dims.KHalfDim < nlev, f(dims.KHalfDim + 0.5), zero)


@gtx.field_operator
def _below_1(
    f: fa.CellKField[ta.wpfloat], zero: fa.CellKHalfField[ta.wpfloat], nlev: gtx.int32
) -> fa.CellKHalfField[ta.wpfloat]:
    return concat_where(dims.KHalfDim < nlev - 1, f(dims.KHalfDim + 1.5), zero)


@gtx.field_operator
def _below_2(
    f: fa.CellKField[ta.wpfloat], zero: fa.CellKHalfField[ta.wpfloat], nlev: gtx.int32
) -> fa.CellKHalfField[ta.wpfloat]:
    return concat_where(dims.KHalfDim < nlev - 2, f(dims.KHalfDim + 2.5), zero)


@gtx.field_operator
def _below_3(
    f: fa.CellKField[ta.wpfloat], zero: fa.CellKHalfField[ta.wpfloat], nlev: gtx.int32
) -> fa.CellKHalfField[ta.wpfloat]:
    return concat_where(dims.KHalfDim < nlev - 3, f(dims.KHalfDim + 3.5), zero)


@gtx.field_operator
def _below_4(
    f: fa.CellKField[ta.wpfloat], zero: fa.CellKHalfField[ta.wpfloat], nlev: gtx.int32
) -> fa.CellKHalfField[ta.wpfloat]:
    return concat_where(dims.KHalfDim < nlev - 4, f(dims.KHalfDim + 4.5), zero)


@gtx.field_operator
def _above_0(
    f: fa.CellKField[ta.wpfloat], zero: fa.CellKHalfField[ta.wpfloat]
) -> fa.CellKHalfField[ta.wpfloat]:
    return concat_where(dims.KHalfDim > 0, f(dims.KHalfDim - 0.5), zero)


@gtx.field_operator
def _above_1(
    f: fa.CellKField[ta.wpfloat], zero: fa.CellKHalfField[ta.wpfloat]
) -> fa.CellKHalfField[ta.wpfloat]:
    return concat_where(dims.KHalfDim > 1, f(dims.KHalfDim - 1.5), zero)


@gtx.field_operator
def _above_2(
    f: fa.CellKField[ta.wpfloat], zero: fa.CellKHalfField[ta.wpfloat]
) -> fa.CellKHalfField[ta.wpfloat]:
    return concat_where(dims.KHalfDim > 2, f(dims.KHalfDim - 2.5), zero)


@gtx.field_operator
def _above_3(
    f: fa.CellKField[ta.wpfloat], zero: fa.CellKHalfField[ta.wpfloat]
) -> fa.CellKHalfField[ta.wpfloat]:
    return concat_where(dims.KHalfDim > 3, f(dims.KHalfDim - 3.5), zero)


@gtx.field_operator
def _above_4(
    f: fa.CellKField[ta.wpfloat], zero: fa.CellKHalfField[ta.wpfloat]
) -> fa.CellKHalfField[ta.wpfloat]:
    return concat_where(dims.KHalfDim > 4, f(dims.KHalfDim - 4.5), zero)


@gtx.field_operator
def _courant_number(
    p_mflx_contra_v: fa.CellKHalfField[ta.wpfloat],
    p_cellmass_now: fa.CellKField[ta.wpfloat],
    k_half: fa.KHalfField[gtx.int32],
    zero_half: fa.CellKHalfField[ta.wpfloat],
    num_levels: gtx.int32,
    wp_eps: ta.wpfloat,
    p_dtime: ta.wpfloat,
) -> fa.CellKHalfField[ta.wpfloat]:
    # `compute_ppm4gpu_courant_number` with nlev = num_levels - 1 and slevp1_ti = 1.
    nlev = num_levels - 1
    m_b0 = _below_0(p_cellmass_now, zero_half, num_levels)
    m_b1 = _below_1(p_cellmass_now, zero_half, num_levels)
    m_b2 = _below_2(p_cellmass_now, zero_half, num_levels)
    m_b3 = _below_3(p_cellmass_now, zero_half, num_levels)
    m_b4 = _below_4(p_cellmass_now, zero_half, num_levels)
    m_a0 = _above_0(p_cellmass_now, zero_half)
    m_a1 = _above_1(p_cellmass_now, zero_half)
    m_a2 = _above_2(p_cellmass_now, zero_half)
    m_a3 = _above_3(p_cellmass_now, zero_half)
    m_a4 = _above_4(p_cellmass_now, zero_half)
    z_mass = p_dtime * p_mflx_contra_v

    # below
    z_mass_pos = z_mass > 0.0
    zm = z_mass
    gt_p0 = where(z_mass_pos & (k_half <= nlev - 1), zm >= m_b0, False)
    zm = zm - where(gt_p0, m_b0, 0.0)
    gt_p1 = gt_p0 & where(z_mass_pos & (k_half <= nlev - 2), zm >= m_b1, False)
    zm = zm - where(gt_p1, m_b1, 0.0)
    gt_p2 = gt_p1 & where(z_mass_pos & (k_half <= nlev - 3), zm >= m_b2, False)
    zm = zm - where(gt_p2, m_b2, 0.0)
    gt_p3 = gt_p2 & where(z_mass_pos & (k_half <= nlev - 4), zm >= m_b3, False)
    zm = zm - where(gt_p3, m_b3, 0.0)
    cfl_below = (
        where(gt_p0, 1.0, 0.0)
        + where(gt_p1, 1.0, 0.0)
        + where(gt_p2, 1.0, 0.0)
        + where(gt_p3, 1.0, 0.0)
    )
    jks_below = where(gt_p3, m_b4, where(gt_p2, m_b3, where(gt_p1, m_b2, where(gt_p0, m_b1, m_b0))))
    # Only the boundary half levels, whose result is discarded, divide by the zero mass of the
    # cell beyond the column; the 0/0 there would make a gradient NaN.
    frac_below = where(z_mass_pos, zm / where(jks_below == 0.0, 1.0, jks_below), 0.0)
    cfl_below = cfl_below + where(frac_below < 1.0, frac_below, 1.0 - wp_eps)

    # above
    z_mass_neg = z_mass <= 0.0
    zm = z_mass
    gt_m0 = where(z_mass_neg & (k_half >= 2), abs(zm) >= m_a0, False)
    zm = zm + where(gt_m0, m_a0, 0.0)
    gt_m1 = gt_m0 & where(z_mass_neg & (k_half >= 3), abs(zm) >= m_a1, False)
    zm = zm + where(gt_m1, m_a1, 0.0)
    gt_m2 = gt_m1 & where(z_mass_neg & (k_half >= 4), abs(zm) >= m_a2, False)
    zm = zm + where(gt_m2, m_a2, 0.0)
    gt_m3 = gt_m2 & where(z_mass_neg & (k_half >= 5), abs(zm) >= m_a3, False)
    zm = zm + where(gt_m3, m_a3, 0.0)
    jks_above = where(gt_m3, m_a4, where(gt_m2, m_a3, where(gt_m1, m_a2, where(gt_m0, m_a1, m_a0))))
    cfl_above = -(
        where(gt_m0, 1.0, 0.0)
        + where(gt_m1, 1.0, 0.0)
        + where(gt_m2, 1.0, 0.0)
        + where(gt_m3, 1.0, 0.0)
    )
    frac_above = where(z_mass_neg, zm / where(jks_above == 0.0, 1.0, jks_above), 0.0)
    cfl_above = cfl_above + where(abs(frac_above) < 1.0, frac_above, wp_eps - 1.0)

    return concat_where(
        (dims.KHalfDim >= 1) & (dims.KHalfDim < num_levels), cfl_below + cfl_above, zero_half
    )


@gtx.field_operator
def _upwind_cells(
    f: fa.CellKField[ta.wpfloat],
    js: fa.CellKHalfField[ta.wpfloat],
    upward: fa.CellKHalfField[bool],
    downward: fa.CellKHalfField[bool],
    zero_half: fa.CellKHalfField[ta.wpfloat],
    num_levels: gtx.int32,
) -> fa.CellKHalfField[ta.wpfloat]:
    # The value of `f` in the cell js levels below (upward flow) or above (downward flow) the
    # face, as `_sum_neighbor_contributions` in `compute_ppm4gpu_fractional_flux`.
    below = where(
        js == 0.0,
        _below_0(f, zero_half, num_levels),
        where(
            js == 1.0,
            _below_1(f, zero_half, num_levels),
            where(
                js == 2.0,
                _below_2(f, zero_half, num_levels),
                where(
                    js == 3.0,
                    _below_3(f, zero_half, num_levels),
                    where(js == 4.0, _below_4(f, zero_half, num_levels), zero_half),
                ),
            ),
        ),
    )
    above = where(
        js == 0.0,
        _above_0(f, zero_half),
        where(
            js == 1.0,
            _above_1(f, zero_half),
            where(
                js == 2.0,
                _above_2(f, zero_half),
                where(
                    js == 3.0,
                    _above_3(f, zero_half),
                    where(js == 4.0, _above_4(f, zero_half), zero_half),
                ),
            ),
        ),
    )
    return where(upward, below, where(downward, above, zero_half))


@gtx.field_operator
def _upwind_cells_integer(
    f: fa.CellKField[ta.wpfloat],
    js: fa.CellKHalfField[ta.wpfloat],
    upward: fa.CellKHalfField[bool],
    downward: fa.CellKHalfField[bool],
    zero_half: fa.CellKHalfField[ta.wpfloat],
    num_levels: gtx.int32,
) -> fa.CellKHalfField[ta.wpfloat]:
    # The sum of `f` over the js + 1 whole cells below (upward flow) or above (downward flow) the
    # face, as `_sum_neighbor_contributions_all` in `compute_ppm4gpu_integer_flux`.
    below = (
        where(js >= 0.0, _below_0(f, zero_half, num_levels), zero_half)
        + where(js >= 1.0, _below_1(f, zero_half, num_levels), zero_half)
        + where(js >= 2.0, _below_2(f, zero_half, num_levels), zero_half)
        + where(js >= 3.0, _below_3(f, zero_half, num_levels), zero_half)
        + where(js >= 4.0, _below_4(f, zero_half, num_levels), zero_half)
    )
    above = (
        where(js >= 0.0, _above_0(f, zero_half), zero_half)
        + where(js >= 1.0, _above_1(f, zero_half), zero_half)
        + where(js >= 2.0, _above_2(f, zero_half), zero_half)
        + where(js >= 3.0, _above_3(f, zero_half), zero_half)
        + where(js >= 4.0, _above_4(f, zero_half), zero_half)
    )
    return where(upward, below, zero_half) + where(downward, above, zero_half)


@gtx.field_operator
def _ppm_vertical_flux(
    p_cc: fa.CellKField[ta.wpfloat],
    p_cellmass_now: fa.CellKField[ta.wpfloat],
    p_mflx_contra_v: fa.CellKHalfField[ta.wpfloat],
    p_cellhgt_mc_now: fa.CellKField[ta.wpfloat],
    k: fa.KField[gtx.int32],
    k_half: fa.KHalfField[gtx.int32],
    zero_full: fa.CellKField[ta.wpfloat],
    zero_half: fa.CellKHalfField[ta.wpfloat],
    num_levels: gtx.int32,
    wp_eps: ta.wpfloat,
    p_dtime: ta.wpfloat,
    vertical_limiter: gtx.int32,
) -> fa.CellKHalfField[ta.wpfloat]:
    elev = num_levels - 1
    z_cfl = _courant_number(
        p_mflx_contra_v=p_mflx_contra_v,
        p_cellmass_now=p_cellmass_now,
        k_half=k_half,
        zero_half=zero_half,
        num_levels=num_levels,
        wp_eps=wp_eps,
        p_dtime=p_dtime,
    )

    slope = _compute_ppm_slope(p_cc=p_cc, p_cellhgt_mc_now=p_cellhgt_mc_now, elev=elev)
    if vertical_limiter == 1:
        # `limit_vertical_slope_semi_monotonically`, with the shift below the last level bounded
        p_cc_below = concat_where(dims.KDim < elev, p_cc(dims.KDim + 1), p_cc)
        p_cc_min_last = minimum(p_cc(dims.KDim - 1), p_cc)
        p_cc_min = where(k == elev, p_cc_min_last, minimum(p_cc_min_last, p_cc_below))
        slope_l = minimum(abs(slope), 2.0 * (p_cc - p_cc_min))
        slope = where(slope >= 0.0, slope_l, -slope_l)
    z_slope = concat_where(dims.KDim >= 1, slope, zero_full)

    quadratic = _compute_ppm_quadratic_face_values(p_cc=p_cc, p_cellhgt_mc_now=p_cellhgt_mc_now)
    quartic = _compute_ppm_quartic_face_values(
        p_cc=p_cc, p_cellhgt_mc_now=p_cellhgt_mc_now, z_slope=z_slope
    )
    z_face = concat_where(
        dims.KHalfDim < 1,
        p_cc(dims.KHalfDim + 0.5),
        concat_where(
            dims.KHalfDim < 2,
            quadratic,
            concat_where(
                dims.KHalfDim < num_levels - 1,
                quartic,
                concat_where(dims.KHalfDim < num_levels, quadratic, p_cc(dims.KHalfDim - 0.5)),
            ),
        ),
    )

    if vertical_limiter == 1:
        l_limit = _compute_vertical_parabola_limiter_condition(p_face=z_face, p_cc=p_cc)
        z_face_up, z_face_low = _limit_vertical_parabola_semi_monotonically(
            l_limit=l_limit, p_face=z_face, p_cc=p_cc
        )
    else:
        z_face_up = _copy_half_level_above_to_model_levels_on_cells(z_face)
        z_face_low = _copy_half_level_below_to_model_levels_on_cells(z_face)
    z_delta_q, z_a1 = _compute_ppm4gpu_parabola_coefficients(
        z_face_up=z_face_up, z_face_low=z_face_low, p_cc=p_cc
    )

    # fractional flux, `compute_ppm4gpu_fractional_flux` with slev = 0
    js = floor(abs(z_cfl))
    z_cflfrac = abs(z_cfl) - js
    up = z_cfl > 0.0
    down = z_cfl < 0.0
    wsign = where(up, 1.0, -1.0)
    up_frac = up & (z_cflfrac != 0.0)
    down_frac = down & (z_cflfrac != 0.0)
    p_cc_jks = _upwind_cells(p_cc, js, up_frac, down_frac, zero_half, num_levels)
    mass_jks = _upwind_cells(p_cellmass_now, js, up_frac, down_frac, zero_half, num_levels)
    delta_q_jks = _upwind_cells(z_delta_q, js, up_frac, down_frac, zero_half, num_levels)
    a1_jks = _upwind_cells(z_a1, js, up_frac, down_frac, zero_half, num_levels)
    z_q_int = (
        p_cc_jks
        + wsign * (delta_q_jks * (1.0 - z_cflfrac))
        - a1_jks * (1.0 - 3.0 * z_cflfrac + 2.0 * z_cflfrac * z_cflfrac)
    )
    k_half_float = astype(k_half, wpfloat)
    flux = where(k_half_float - js >= 0.0, wsign * mass_jks * z_cflfrac * z_q_int / p_dtime, 0.0)

    # integer flux, `compute_ppm4gpu_integer_flux` with slev = 0
    js_int = js - 1.0
    tracer_mass_jks = _upwind_cells_integer(
        p_cc * p_cellmass_now, js_int, up, down, zero_half, num_levels
    )
    flux = flux + where(k_half_float - js_int >= 0.0, wsign * tracer_mass_jks / p_dtime, 0.0)

    return concat_where((dims.KHalfDim >= 1) & (dims.KHalfDim < num_levels), flux, zero_half)


@gtx.field_operator
def _vertical_advection(
    p_tracer_now: fa.CellKField[ta.wpfloat],
    rhodz_now: fa.CellKField[ta.wpfloat],
    rhodz_new: fa.CellKField[ta.wpfloat],
    p_mflx_contra_v: fa.CellKHalfField[ta.wpfloat],
    p_cellhgt_mc_now: fa.CellKField[ta.wpfloat],
    deepatmo_divzl: fa.KField[ta.wpfloat],
    deepatmo_divzu: fa.KField[ta.wpfloat],
    k: fa.KField[gtx.int32],
    k_half: fa.KHalfField[gtx.int32],
    zero_full: fa.CellKField[ta.wpfloat],
    zero_half: fa.CellKHalfField[ta.wpfloat],
    num_levels: gtx.int32,
    wp_eps: ta.wpfloat,
    p_dtime: ta.wpfloat,
    vertical_advection_type: gtx.int32,
    vertical_limiter: gtx.int32,
) -> tuple[fa.CellKField[ta.wpfloat], fa.CellKHalfField[ta.wpfloat]]:
    if vertical_advection_type == 1:
        flux = concat_where(
            (dims.KHalfDim >= 1) & (dims.KHalfDim < num_levels),
            _compute_vertical_tracer_flux_upwind(
                p_cc=p_tracer_now, p_mflx_contra_v=p_mflx_contra_v
            ),
            zero_half,
        )
    elif vertical_advection_type == 3:
        flux = _ppm_vertical_flux(
            p_cc=p_tracer_now,
            p_cellmass_now=rhodz_now,
            p_mflx_contra_v=p_mflx_contra_v,
            p_cellhgt_mc_now=p_cellhgt_mc_now,
            k=k,
            k_half=k_half,
            zero_full=zero_full,
            zero_half=zero_half,
            num_levels=num_levels,
            wp_eps=wp_eps,
            p_dtime=p_dtime,
            vertical_limiter=vertical_limiter,
        )
    else:
        flux = zero_half

    if vertical_advection_type == 0:
        tracer = p_tracer_now
    else:
        tracer = _integrate_tracer_vertically(
            tracer_now=p_tracer_now,
            rhodz_now=rhodz_now,
            p_mflx_tracer_v=flux,
            deepatmo_divzl=deepatmo_divzl,
            deepatmo_divzu=deepatmo_divzu,
            rhodz_new=rhodz_new,
            k=k,
            p_dtime=p_dtime,
            ivadv_tracer=1,
            iadv_slev_jt=0,
        )
    return tracer, flux


@gtx.field_operator
def _horizontal_advection(
    p_tracer_now: fa.CellKField[ta.wpfloat],
    rhodz_now: fa.CellKField[ta.wpfloat],
    rhodz_new: fa.CellKField[ta.wpfloat],
    p_mass_flx_e: fa.EdgeKField[ta.wpfloat],
    p_distv_bary_1: fa.EdgeKField[ta.vpfloat],
    p_distv_bary_2: fa.EdgeKField[ta.vpfloat],
    p_vn: fa.EdgeKField[ta.wpfloat],
    zero_edge: fa.EdgeKField[ta.wpfloat],
    deepatmo_divh: fa.KField[ta.wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat],
    lsq_pseudoinv_1: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_2: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    wp_eps: ta.wpfloat,
    p_dtime: ta.wpfloat,
    horizontal_advection_type: gtx.int32,
    horizontal_limiter: gtx.int32,
) -> tuple[fa.CellKField[ta.wpfloat], fa.EdgeKField[ta.wpfloat]]:
    if horizontal_advection_type == 1:
        flux = _compute_horizontal_tracer_flux_upwind(
            p_cc=p_tracer_now, p_mass_flx_e=p_mass_flx_e, p_vn=p_vn
        )
    elif horizontal_advection_type == 2:
        coeff_1, coeff_2, coeff_3 = _reconstruct_linear_coefficients_svd(
            p_cc=p_tracer_now, lsq_pseudoinv_1=lsq_pseudoinv_1, lsq_pseudoinv_2=lsq_pseudoinv_2
        )
        flux = _compute_horizontal_tracer_flux_from_linear_coefficients_alt(
            z_lsq_coeff_1=coeff_1,
            z_lsq_coeff_2=coeff_2,
            z_lsq_coeff_3=coeff_3,
            distv_bary_1=p_distv_bary_1,
            distv_bary_2=p_distv_bary_2,
            p_mass_flx_e=p_mass_flx_e,
            p_vn=p_vn,
        )
        if horizontal_limiter == 4:
            r_m = _compute_positive_definite_horizontal_multiplicative_flux_factor(
                geofac_div=geofac_div,
                p_cc=p_tracer_now,
                p_rhodz_now=rhodz_now,
                p_mflx_tracer_h=flux,
                p_dtime=p_dtime,
                wp_eps=wp_eps,
            )
            flux = _apply_positive_definite_horizontal_multiplicative_flux_factor(
                r_m=r_m, p_mflx_tracer_h=flux
            )
    else:
        flux = zero_edge

    if horizontal_advection_type == 0:
        tracer = p_tracer_now
    else:
        tracer = _integrate_tracer_horizontally(
            p_mflx_tracer_h=flux,
            deepatmo_divh=deepatmo_divh,
            tracer_now=p_tracer_now,
            rhodz_now=rhodz_now,
            rhodz_new=rhodz_new,
            geofac_div=geofac_div,
            p_dtime=p_dtime,
        )
    return tracer, flux


@gtx.field_operator
def _tracer_advection_global_step(
    p_tracer_now: fa.CellKField[ta.wpfloat],
    airmass_now: fa.CellKField[ta.wpfloat],
    airmass_new: fa.CellKField[ta.wpfloat],
    mass_flx_ic: fa.CellKHalfField[ta.wpfloat],
    mass_flx_me: fa.EdgeKField[ta.wpfloat],
    vn_traj: fa.EdgeKField[ta.wpfloat],
    ddqz_z_full: fa.CellKField[ta.wpfloat],
    deepatmo_divh: fa.KField[ta.wpfloat],
    deepatmo_divzl: fa.KField[ta.wpfloat],
    deepatmo_divzu: fa.KField[ta.wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat],
    rbf_vec_coeff_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], ta.wpfloat],
    pos_on_tplane_e_1: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    pos_on_tplane_e_2: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    primal_normal_cell_1: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    dual_normal_cell_1: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    primal_normal_cell_2: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    dual_normal_cell_2: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    lsq_pseudoinv_1: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_2: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    k: fa.KField[gtx.int32],
    k_half: fa.KHalfField[gtx.int32],
    zero_full: fa.CellKField[ta.wpfloat],
    zero_half: fa.CellKHalfField[ta.wpfloat],
    zero_edge: fa.EdgeKField[ta.wpfloat],
    num_levels: gtx.int32,
    wp_eps: ta.wpfloat,
    p_dtime: ta.wpfloat,
    even_timestep: bool,
    horizontal_advection_type: gtx.int32,
    horizontal_limiter: gtx.int32,
    vertical_advection_type: gtx.int32,
    vertical_limiter: gtx.int32,
) -> tuple[
    fa.CellKField[ta.wpfloat],
    fa.EdgeKField[ta.wpfloat],
    fa.CellKHalfField[ta.wpfloat],
]:
    rhodz_ast2 = _apply_density_increment(
        rhodz_in=airmass_now if even_timestep else airmass_new,
        p_mflx_contra_v=mass_flx_ic,
        deepatmo_divzl=deepatmo_divzl,
        deepatmo_divzu=deepatmo_divzu,
        p_dtime=p_dtime,
        even_timestep=even_timestep,
    )

    if horizontal_advection_type == 2:
        vt = _compute_tangential_wind(vn=vn_traj, rbf_vec_coeff_e=rbf_vec_coeff_e)
        distv_bary_1, distv_bary_2 = _compute_barycentric_backtrajectory_alt(
            p_vn=vn_traj,
            p_vt=vt,
            pos_on_tplane_e_1=pos_on_tplane_e_1,
            pos_on_tplane_e_2=pos_on_tplane_e_2,
            primal_normal_cell_1=primal_normal_cell_1,
            dual_normal_cell_1=dual_normal_cell_1,
            primal_normal_cell_2=primal_normal_cell_2,
            dual_normal_cell_2=dual_normal_cell_2,
            p_dthalf=0.5 * p_dtime,
        )
    else:
        distv_bary_1 = zero_edge
        distv_bary_2 = zero_edge

    if even_timestep:
        tracer_v, vfl = _vertical_advection(
            p_tracer_now=p_tracer_now,
            rhodz_now=airmass_now,
            rhodz_new=rhodz_ast2,
            p_mflx_contra_v=mass_flx_ic,
            p_cellhgt_mc_now=ddqz_z_full,
            deepatmo_divzl=deepatmo_divzl,
            deepatmo_divzu=deepatmo_divzu,
            k=k,
            k_half=k_half,
            zero_full=zero_full,
            zero_half=zero_half,
            num_levels=num_levels,
            wp_eps=wp_eps,
            p_dtime=p_dtime,
            vertical_advection_type=vertical_advection_type,
            vertical_limiter=vertical_limiter,
        )
        tracer_new, hfl = _horizontal_advection(
            p_tracer_now=tracer_v,
            rhodz_now=rhodz_ast2,
            rhodz_new=airmass_new,
            p_mass_flx_e=mass_flx_me,
            p_distv_bary_1=distv_bary_1,
            p_distv_bary_2=distv_bary_2,
            p_vn=vn_traj,
            zero_edge=zero_edge,
            deepatmo_divh=deepatmo_divh,
            geofac_div=geofac_div,
            lsq_pseudoinv_1=lsq_pseudoinv_1,
            lsq_pseudoinv_2=lsq_pseudoinv_2,
            wp_eps=wp_eps,
            p_dtime=p_dtime,
            horizontal_advection_type=horizontal_advection_type,
            horizontal_limiter=horizontal_limiter,
        )
    else:
        tracer_h, hfl = _horizontal_advection(
            p_tracer_now=p_tracer_now,
            rhodz_now=airmass_now,
            rhodz_new=rhodz_ast2,
            p_mass_flx_e=mass_flx_me,
            p_distv_bary_1=distv_bary_1,
            p_distv_bary_2=distv_bary_2,
            p_vn=vn_traj,
            zero_edge=zero_edge,
            deepatmo_divh=deepatmo_divh,
            geofac_div=geofac_div,
            lsq_pseudoinv_1=lsq_pseudoinv_1,
            lsq_pseudoinv_2=lsq_pseudoinv_2,
            wp_eps=wp_eps,
            p_dtime=p_dtime,
            horizontal_advection_type=horizontal_advection_type,
            horizontal_limiter=horizontal_limiter,
        )
        tracer_new, vfl = _vertical_advection(
            p_tracer_now=tracer_h,
            rhodz_now=rhodz_ast2,
            rhodz_new=airmass_new,
            p_mflx_contra_v=mass_flx_ic,
            p_cellhgt_mc_now=ddqz_z_full,
            deepatmo_divzl=deepatmo_divzl,
            deepatmo_divzu=deepatmo_divzu,
            k=k,
            k_half=k_half,
            zero_full=zero_full,
            zero_half=zero_half,
            num_levels=num_levels,
            wp_eps=wp_eps,
            p_dtime=p_dtime,
            vertical_advection_type=vertical_advection_type,
            vertical_limiter=vertical_limiter,
        )
    return tracer_new, hfl, vfl
