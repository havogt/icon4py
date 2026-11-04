# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""
Saturation adjustment, AES graupel and saturation adjustment as one field operator.

The same composition as `muphys._muphys` with masking enabled. The scan pass of the sedimentation
does not branch on values: embedded execution can then run it on whole levels, which is what lets
JAX trace it.
"""

import gt4py.next as gtx
from gt4py.next import astype, exp, maximum, minimum, power, scan, sqrt, where
from gt4py.next.experimental import concat_where

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.common.constants import (
    AesGraupelConsts,
    GraupelConsts,
    ThermodynamicConsts,
)
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.definitions import Q, Q_scalar
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.properties import (
    _vm_graupel_scalar,
    _vm_ice_scalar,
    _vm_rain_scalar,
)
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.saturation_adjustment import (
    _saturation_adjustment,
)
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.thermo import _qsat_ice_rho
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.implementations.graupel import (
    IntegrationState,
    PrecipStateQx,
    TempState,
    _q_t_update,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.physics.thermodynamics.compute_energy import (
    compute_internal_energy_per_area_scalar,
)
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _select(condition: bool, true_value: ta.wpfloat, false_value: ta.wpfloat) -> ta.wpfloat:
    # `where` takes no scalar condition and a value branch stops JAX from tracing the scan; this
    # is exact as long as both values are finite.
    c = astype(condition, ta.wpfloat)
    return c * true_value + (wpfloat(1.0) - c) * false_value


@gtx.field_operator
def _snow_number_scalar_branchless(t: ta.wpfloat, rho_s: ta.wpfloat) -> ta.wpfloat:
    TMIN = ThermodynamicConsts.tmelt - wpfloat(40.0)
    TMAX = ThermodynamicConsts.tmelt
    XA1 = wpfloat(-1.65e0)
    XA2 = wpfloat(5.45e-2)
    XA3 = wpfloat(3.27e-4)
    XB1 = wpfloat(1.42e0)
    XB2 = wpfloat(1.19e-2)
    XB3 = wpfloat(9.60e-5)
    N0S0 = wpfloat(8.00e5)
    N0S1 = wpfloat(13.5) * wpfloat(5.65e05)
    N0S2 = wpfloat(-0.107)
    N0S3 = wpfloat(13.5)
    N0S4 = wpfloat(0.5) * N0S1
    N0S5 = wpfloat(1.0e6)
    N0S6 = wpfloat(1.0e2) * N0S1
    N0S7 = wpfloat(1.0e9)

    tc = maximum(minimum(t, TMAX), TMIN) - ThermodynamicConsts.tmelt
    alf = power(wpfloat(10.0), (XA1 + tc * (XA2 + tc * XA3)))
    bet = XB1 + tc * (XB2 + tc * XB3)
    n0s = (
        N0S3
        * power(
            (maximum(rho_s, AesGraupelConsts.rho_s_mn) / GraupelConsts.ams),
            (wpfloat(4.0) - wpfloat(3.0) * bet),
        )
        / (alf * alf * alf)
    )
    y = exp(N0S2 * tc)
    n0smn = maximum(N0S4 * y, N0S5)
    n0smx = minimum(N0S6 * y, N0S7)
    return _select(rho_s > GraupelConsts.qmin, minimum(n0smx, maximum(n0smn, n0s)), N0S0)


@gtx.field_operator
def _vm_snow_scalar_branchless(rho_x: ta.wpfloat, rho: ta.wpfloat, t: ta.wpfloat) -> ta.wpfloat:
    B_S = wpfloat(-0.16666666666666667)
    x = minimum(AesGraupelConsts.rhox_mx, maximum(AesGraupelConsts.rhox_mn, rho_x))
    return (
        AesGraupelConsts.vm_prefactor_s
        * power(x, AesGraupelConsts.vm_exponent_s)
        * sqrt(GraupelConsts.rho_00 / rho)
        * power(_snow_number_scalar_branchless(t=t, rho_s=x), B_S)
    )


@gtx.field_operator
def _precip_qx_level_update_branchless(  # noqa: PLR0917 [too-many-positional-arguments]
    previous_level_q: PrecipStateQx,
    zeta: ta.wpfloat,
    q: ta.wpfloat,
    rho: ta.wpfloat,
    fall_speed: ta.wpfloat,
    vt_prev: ta.wpfloat,
    mask: bool,
) -> PrecipStateQx:
    activated = previous_level_q.activated | mask
    rho_x = q * rho
    flx_eff = (rho_x / zeta) + wpfloat(2.0) * previous_level_q.p
    flx_partial = minimum(rho_x * fall_speed, flx_eff)
    vt = _select(previous_level_q.activated, vt_prev, wpfloat(0.0))
    x_new = (zeta * (flx_eff - flx_partial)) / ((wpfloat(1.0) + zeta * vt) * rho)
    p_new = (x_new * rho * vt + flx_partial) * wpfloat(0.5)
    return PrecipStateQx(
        x=_select(activated, x_new, q),
        p=_select(activated, p_new, wpfloat(0.0)),
        activated=activated,
    )


@gtx.field_operator
def _temperature_update_branchless(  # noqa: PLR0917 [too-many-positional-arguments]
    previous_level: TempState,
    t: ta.wpfloat,
    t_kp1: ta.wpfloat,
    pr: ta.wpfloat,
    pflx_tot: ta.wpfloat,
    q: Q_scalar,
    qliq: ta.wpfloat,
    qice: ta.wpfloat,
    rho: ta.wpfloat,
    dz: ta.wpfloat,
    dt: ta.wpfloat,
    mask: bool,
) -> TempState:
    activated = previous_level.activated | mask
    eflx = pr * (
        ThermodynamicConsts.clw * t - ThermodynamicConsts.cvd * t_kp1 - GraupelConsts.lvc
    ) + pflx_tot * (GraupelConsts.ci * t - ThermodynamicConsts.cvd * t_kp1 - GraupelConsts.lsc)
    e_int = (
        compute_internal_energy_per_area_scalar(
            t=t, qv=q.v, qliq=q.c + q.r, qice=q.s + q.i + q.g, rho=rho, dz=dz
        )
        + dt * previous_level.eflx
        - dt * eflx
    )
    qtot = qliq + qice + q.v
    cv = (
        (
            ThermodynamicConsts.cvd * (wpfloat(1.0) - qtot)
            + ThermodynamicConsts.cvv * q.v
            + ThermodynamicConsts.clw * qliq
            + GraupelConsts.ci * qice
        )
        * rho
        * dz
    )
    t_new = (e_int + rho * dz * (qliq * GraupelConsts.lvc + qice * GraupelConsts.lsc)) / cv
    return TempState(
        t=_select(activated, t_new, t),
        eflx=_select(activated, eflx, previous_level.eflx),
        activated=activated,
    )


@gtx.field_operator
def _precip_and_t_branchless(  # noqa: PLR0917 [too-many-positional-arguments]
    previous_level: IntegrationState,
    t: ta.wpfloat,
    t_kp1: ta.wpfloat,
    rho: ta.wpfloat,
    q: Q_scalar,
    mask_r: bool,
    mask_s: bool,
    mask_i: bool,
    mask_g: bool,
    dt: ta.wpfloat,
    dz: ta.wpfloat,
) -> IntegrationState:
    zeta = dt / (wpfloat(2.0) * dz)
    r_update = _precip_qx_level_update_branchless(
        previous_level.r,
        zeta,
        q.r,
        rho,
        _vm_rain_scalar(q.r * rho, rho),
        _vm_rain_scalar(
            (previous_level.r.x + q.r) * wpfloat(0.5) * previous_level.rho, previous_level.rho
        ),
        mask_r,
    )
    s_update = _precip_qx_level_update_branchless(
        previous_level.s,
        zeta,
        q.s,
        rho,
        _vm_snow_scalar_branchless(q.s * rho, rho, t),
        _vm_snow_scalar_branchless(
            (previous_level.s.x + q.s) * wpfloat(0.5) * previous_level.rho,
            previous_level.rho,
            previous_level.t_in,
        ),
        mask_s,
    )
    i_update = _precip_qx_level_update_branchless(
        previous_level.i,
        zeta,
        q.i,
        rho,
        _vm_ice_scalar(q.i * rho, rho),
        _vm_ice_scalar(
            (previous_level.i.x + q.i) * wpfloat(0.5) * previous_level.rho, previous_level.rho
        ),
        mask_i,
    )
    g_update = _precip_qx_level_update_branchless(
        previous_level.g,
        zeta,
        q.g,
        rho,
        _vm_graupel_scalar(q.g * rho, rho),
        _vm_graupel_scalar(
            (previous_level.g.x + q.g) * wpfloat(0.5) * previous_level.rho, previous_level.rho
        ),
        mask_g,
    )
    t_update = _temperature_update_branchless(
        previous_level.t_state,
        t=t,
        t_kp1=t_kp1,
        pr=r_update.p,
        pflx_tot=s_update.p + i_update.p + g_update.p,
        q=q,
        qliq=q.c + r_update.x,
        qice=s_update.x + i_update.x + g_update.x,
        rho=rho,
        dz=dz,
        dt=dt,
        mask=mask_r | mask_s | mask_i | mask_g,
    )
    return IntegrationState(
        r=r_update,
        s=s_update,
        i=i_update,
        g=g_update,
        t_state=t_update,
        rho=rho,
        pflx_tot=s_update.p + i_update.p + g_update.p + r_update.p,
        t_in=t,
    )


@gtx.field_operator
def _muphys_global_step(  # noqa: PLR0917 [too-many-positional-arguments]
    num_levels: gtx.int32,
    dz: fa.CellKField[ta.wpfloat],
    te: fa.CellKField[ta.wpfloat],
    p: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
    q_in: Q,
    dt: ta.wpfloat,
    qnc: ta.wpfloat,
) -> tuple[
    fa.CellKField[ta.wpfloat],
    Q,
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
]:
    te_sa, qv_sa, qc_sa = _saturation_adjustment(te=te, q_in=q_in, rho=rho)
    q = Q(v=qv_sa, c=qc_sa, r=q_in.r, s=q_in.s, i=q_in.i, g=q_in.g)

    mask = (maximum(q.c, maximum(q.g, maximum(q.i, maximum(q.r, q.s)))) > GraupelConsts.qmin) | (
        (te_sa < GraupelConsts.tfrz_het2) & (q.v > _qsat_ice_rho(te_sa, rho))
    )
    q_upd, t_upd = where(
        mask, _q_t_update(te_sa, p, rho, q, dt, qnc, enable_masking=True), (q, te_sa)
    )

    t_kp1 = concat_where(dims.KDim < num_levels - 1, t_upd(dims.KDim + 1), t_upd)
    # The density carried into the first level only enters fall speeds that `where` discards
    # there; it is positive so that their gradients stay finite.
    state = scan(
        _precip_and_t_branchless,
        range=(dims.KDim, 0, num_levels),
        forward=True,
        init=IntegrationState(
            r=PrecipStateQx(x=0.0, p=0.0, activated=False),
            s=PrecipStateQx(x=0.0, p=0.0, activated=False),
            i=PrecipStateQx(x=0.0, p=0.0, activated=False),
            g=PrecipStateQx(x=0.0, p=0.0, activated=False),
            t_state=TempState(t=0.0, eflx=0.0, activated=False),
            rho=1.0,
            pflx_tot=0.0,
            t_in=0.0,
        ),
    )(
        t_upd,
        t_kp1,
        rho,
        q_upd,
        q.r > GraupelConsts.qmin,
        q.s > GraupelConsts.qmin,
        q.i > GraupelConsts.qmin,
        q.g > GraupelConsts.qmin,
        dt,
        dz,
    )

    te_out, qv_out, qc_out = _saturation_adjustment(
        te=state.t_state.t,
        q_in=Q(v=q_upd.v, c=q_upd.c, r=state.r.x, s=state.s.x, i=state.i.x, g=state.g.x),
        rho=rho,
    )
    return (
        te_out,
        Q(v=qv_out, c=qc_out, r=state.r.x, s=state.s.x, i=state.i.x, g=state.g.x),
        state.pflx_tot,
        state.r.p,
        state.s.p,
        state.i.p,
        state.g.p,
        state.t_state.eflx,
    )
