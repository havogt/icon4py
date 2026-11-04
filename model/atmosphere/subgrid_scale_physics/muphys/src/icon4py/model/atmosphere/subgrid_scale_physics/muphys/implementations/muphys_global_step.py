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
from gt4py.next import astype, broadcast, exp, log, maximum, minimum, power, scan, sqrt, where
from gt4py.next.experimental import concat_where

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.common.constants import (
    AesGraupelConsts,
    GraupelConsts,
    ThermodynamicConsts,
)
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.definitions import Q, Q_scalar
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.properties import (
    _deposition_auto_conversion,
    _deposition_factor,
    _ice_deposition_nucleation,
    _ice_mass,
    _ice_number,
    _ice_sticking,
    _snow_number,
    _vm_graupel_scalar,
    _vm_ice_scalar,
    _vm_rain_scalar,
)
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.saturation_adjustment import (
    _saturation_adjustment,
)
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.thermo import (
    _qsat_ice_rho,
    _qsat_rho,
    _qsat_rho_tmelt,
)
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.transitions import (
    _cloud_to_snow,
    _cloud_x_ice,
    _ice_to_snow,
    _rain_to_graupel,
    _rain_to_vapor,
    _vapor_x_ice,
)
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.implementations.graupel import (
    IntegrationState,
    PrecipStateQx,
    TempState,
    cond_symmetric,
    symmetric,
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
def _power_of_nonnegative(x: fa.CellKField[ta.wpfloat], exponent: ta.wpfloat) -> fa.CellKField[ta.wpfloat]:
    # 0**exponent is 0 for a positive exponent, but its derivative is infinite, which `where`
    # turns into NaN in the gradient of the branch it discards.
    positive = x > wpfloat(0.0)
    return where(positive, power(where(positive, x, wpfloat(1.0)), exponent), wpfloat(0.0))


@gtx.field_operator
def _cloud_to_graupel_guarded(
    t: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
    qc: fa.CellKField[ta.wpfloat],
    qg: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the conversion rate from cloud to graupel

    Args:
        t:                  Temperature
        rho:                Ambient density
        qc:                 Cloud specific mass
        qg:                 Graupel specific mass

    Return:                 Conversion rate
    """
    A_RIM = wpfloat(4.43)
    B_RIM = wpfloat(0.94878)
    return where(
        (minimum(qc, qg) > GraupelConsts.qmin) & (t > GraupelConsts.tfrz_hom),
        A_RIM * qc * _power_of_nonnegative(qg * rho, B_RIM),
        wpfloat(0.0),
    )


@gtx.field_operator
def _graupel_to_rain_guarded(
    t: fa.CellKField[ta.wpfloat],
    p: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
    dvsw0: fa.CellKField[ta.wpfloat],
    qg: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the conversion rate from graupel to rain

    Args:
        t:                  Temperature
        p:                  Ambient pressure
        rho:                Ambient density
        dvsw0:              qv-qsat_water(T0)
        qg:                 Graupel specific mass

    Return:                 Conversion rate
    """
    A_MELT = wpfloat(GraupelConsts.tx) - wpfloat(389.5)  # melting prefactor
    B_MELT = wpfloat(0.6)  # melting exponent
    C1_MELT = wpfloat(12.31698)  # Constants in melting formula
    C2_MELT = wpfloat(7.39441e-05)  # Constants in melting formula
    return where(
        (
            t
            > maximum(
                ThermodynamicConsts.tmelt,
                ThermodynamicConsts.tmelt - GraupelConsts.tx * dvsw0,
            )
        )
        & (qg > GraupelConsts.qmin),
        (C1_MELT / p + C2_MELT)
        * (t - ThermodynamicConsts.tmelt + A_MELT * dvsw0)
        * _power_of_nonnegative(qg * rho, B_MELT),
        wpfloat(0.0),
    )


@gtx.field_operator
def _ice_to_graupel_guarded(
    rho: fa.CellKField[ta.wpfloat],
    qr: fa.CellKField[ta.wpfloat],
    qg: fa.CellKField[ta.wpfloat],
    qi: fa.CellKField[ta.wpfloat],
    sticking_eff: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the conversion rate from ice to graupel

    Args:
        rho:                Ambient density
        qr:                 Rain specific mass
        qg:                 Graupel specific mass
        qi:                 Ice specific mass
        sticking_eff:       Sticking efficiency

    Return:                 Conversion rate
    """
    A_CT = wpfloat(1.72)  # (15/32)*(PI**0.5)*(EIR/RHOW)*V0R*AR**(1/8)
    B_CT = wpfloat(0.875)  # Exponent = 7/8
    C_AGG_CT = wpfloat(2.46)
    B_AGG_CT = wpfloat(0.94878)  # Exponent
    result = where(
        (qi > GraupelConsts.qmin) & (qg > GraupelConsts.qmin),
        sticking_eff * qi * C_AGG_CT * _power_of_nonnegative(rho * qg, B_AGG_CT),
        wpfloat(0.0),
    )
    result = where(
        (qi > GraupelConsts.qmin) & (qr > GraupelConsts.qmin),
        result + A_CT * qi * _power_of_nonnegative(rho * qr, B_CT),
        result,
    )
    return result


@gtx.field_operator
def _snow_to_graupel_guarded(
    t: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
    qc: fa.CellKField[ta.wpfloat],
    qs: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the conversion rate from snow to graupel

    Args:
        t:                  Temperature
        rho:                Ambient density
        qc:                 Cloud specific mass
        qs:                 Snow specific mass

    Return:                 Conversion rate
    """
    A_RIM_CT = wpfloat(0.5)  # Constants in riming formula
    B_RIM_CT = wpfloat(0.75)
    return where(
        (minimum(qc, qs) > GraupelConsts.qmin) & (t > GraupelConsts.tfrz_hom),
        A_RIM_CT * qc * _power_of_nonnegative(qs * rho, B_RIM_CT),
        wpfloat(0.0),
    )


@gtx.field_operator
def _snow_to_rain_guarded(
    t: fa.CellKField[ta.wpfloat],
    p: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
    dvsw0: fa.CellKField[ta.wpfloat],
    qs: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the conversion rate from snow to rain

    Args:
        t:                  Temperature
        p:                  Ambient pressure
        rho:                Ambient density
        dvsw0:              qv-qsat_water (T0)
        qs:                 Snow specific mass

    Return:                 Conversion rate
    """
    C1_SR = wpfloat(79.6863)  # Constants in melting formula
    C2_SR = wpfloat(0.612654e-3)  # Constants in melting formula
    A_SR = GraupelConsts.tx - wpfloat(389.5)  # Melting prefactor
    B_SR = wpfloat(0.8)  # Melting exponent
    return where(
        (
            t
            > maximum(
                ThermodynamicConsts.tmelt,
                ThermodynamicConsts.tmelt - GraupelConsts.tx * dvsw0,
            )
        )
        & (qs > GraupelConsts.qmin),
        (C1_SR / p + C2_SR)
        * (t - ThermodynamicConsts.tmelt + A_SR * dvsw0)
        * _power_of_nonnegative(qs * rho, B_SR),
        wpfloat(0.0),
    )


@gtx.field_operator
def _vapor_x_graupel_guarded(  # noqa: PLR0917 [too-many-positional-arguments]
    t: fa.CellKField[ta.wpfloat],
    p: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
    qg: fa.CellKField[ta.wpfloat],
    dvsw: fa.CellKField[ta.wpfloat],
    dvsi: fa.CellKField[ta.wpfloat],
    dvsw0: fa.CellKField[ta.wpfloat],
    dt: ta.wpfloat,
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the conversion rate from vapor to graupel

    Args:
        t:                  Temperature
        p:                  Ambient pressure
        rho:                Ambient density
        qg:                 Graupel specific mass
        dvsw:               qv-qsat_water (T)
        dvsi:               qv-qsat_ice (T)
        dvsw0:              qv-qsat_water (T0)
        dt:                 Time step

    Return:                 Conversion rate
    """
    A1_VG = wpfloat(0.398561)
    A2_VG = wpfloat(-0.00152398)
    A3 = wpfloat(2554.99)
    A4 = wpfloat(2.6531e-7)
    A5 = wpfloat(0.153907)
    A6 = wpfloat(-7.86703e-07)
    A7 = wpfloat(0.0418521)
    A8 = wpfloat(-4.7524e-8)
    B_VG = wpfloat(0.6)
    result = where(
        (t < ThermodynamicConsts.tmelt),
        (A1_VG + A2_VG * t + A3 / p + A4 * p) * dvsi * _power_of_nonnegative(qg * rho, B_VG),
        where(
            (t > (ThermodynamicConsts.tmelt - GraupelConsts.tx * dvsw0)),
            (A5 + A6 * p) * minimum(wpfloat(0.0), dvsw0) * _power_of_nonnegative(qg * rho, B_VG),
            (A7 + A8 * p) * dvsw * _power_of_nonnegative(qg * rho, B_VG),
        ),
    )
    return where(qg > GraupelConsts.qmin, maximum(result, -qg / dt), wpfloat(0.0))


@gtx.field_operator
def _vapor_x_snow_guarded(  # noqa: PLR0917 [too-many-positional-arguments]
    t: fa.CellKField[ta.wpfloat],
    p: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
    qs: fa.CellKField[ta.wpfloat],
    ns: fa.CellKField[ta.wpfloat],
    lam: fa.CellKField[ta.wpfloat],
    eta: fa.CellKField[ta.wpfloat],
    ice_dep: fa.CellKField[ta.wpfloat],
    dvsw: fa.CellKField[ta.wpfloat],
    dvsi: fa.CellKField[ta.wpfloat],
    dvsw0: fa.CellKField[ta.wpfloat],
    dt: ta.wpfloat,
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the conversion rate from vapor to snow

    Args:
        t:                  Temperature
        p:                  Ambient pressure
        rho:                Ambient density
        qs:                 Snow specific mass
        ns:                 Snow number
        lam:                Slope parameter (lambda) snow
        eta:                Deposition factor
        ice_dep:            Limiter for vapor dep on snow
        dvsw:               qv-qsat_water (T)
        dvsi:               qv-qsat_ice(T)
        dvsw0:              qv-qsat_water(T0)
        dt:                 Time step

    Return:                 Conversion rate
    """
    NU = wpfloat(1.75e-5)  # kinematic viscosity of air
    A0_VS = wpfloat(1.0)
    A1_VS = wpfloat(0.4182) * sqrt(GraupelConsts.v0s / NU)
    A2_VS = -(GraupelConsts.v1s + wpfloat(1.0)) / wpfloat(2.0)
    EPS = wpfloat(1.0e-15)
    QS_LIM = wpfloat(1.0e-7)
    CNX = wpfloat(4.0)
    B_VS = wpfloat(0.8)
    C1_VS = wpfloat(31282.3)
    C2_VS = wpfloat(0.241897)
    C3_VS = wpfloat(0.28003)
    C4_VS = wpfloat(-0.146293e-6)

    # See if this can be incorporated into WHERE statement
    result = where(
        (t < ThermodynamicConsts.tmelt),
        (CNX * ns * eta / rho) * (A0_VS + A1_VS * power(lam, A2_VS)) * dvsi / (lam * lam + EPS),
        wpfloat(0.0),
    )

    # GZ: This mask>0 limitation, which was missing in the original graupel scheme,
    # is crucial for numerical stability in the tropics!
    # a meaningful distinction between cloud ice and snow
    result = where(
        (t < ThermodynamicConsts.tmelt) & (result > wpfloat(0.0)),
        minimum(result, dvsi / dt - ice_dep),
        result,
    )
    result = where(
        (t < ThermodynamicConsts.tmelt) & (qs <= QS_LIM), minimum(result, wpfloat(0.0)), result
    )
    # ELSE section
    result = where(
        (t >= ThermodynamicConsts.tmelt)
        & (t > (ThermodynamicConsts.tmelt - GraupelConsts.tx * dvsw0)),
        (C1_VS / p + C2_VS) * minimum(wpfloat(0.0), dvsw0) * _power_of_nonnegative(qs * rho, B_VS),
        result,
    )
    result = where(
        (t >= ThermodynamicConsts.tmelt)
        & (t <= (ThermodynamicConsts.tmelt - GraupelConsts.tx * dvsw0)),
        (C3_VS + C4_VS * p) * dvsw * _power_of_nonnegative(qs * rho, B_VS),
        result,
    )
    return where((qs > GraupelConsts.qmin), maximum(result, -qs / dt), wpfloat(0.0))



@gtx.field_operator
def _snow_lambda_guarded(
    rho_s: fa.CellKField[ta.wpfloat],
    ns: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the snow slope parameter.

    Args:
        rho_s:        Snow mass density rho*qs
        ns:           Snow number

    Result:           Snow slope parameter (lambda)
    """
    A2 = GraupelConsts.ams * wpfloat(2.0)  # (with ams*gam(bms+1.0_wp) where gam(3) = 2)
    LMD_0 = wpfloat(1.0e10)  # no snow value of lambda
    BX = wpfloat(1.0) / (GraupelConsts.bms + wpfloat(1.0))  # Exponent

    has_snow = rho_s > GraupelConsts.qmin
    return where(has_snow, power((A2 * ns / where(has_snow, rho_s, wpfloat(1.0))), BX), LMD_0)



@gtx.field_operator
def _cloud_to_rain_guarded(
    t: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
    qc: fa.CellKField[ta.wpfloat],
    qr: fa.CellKField[ta.wpfloat],
    nc: ta.wpfloat,
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the conversion rate from cloud to rain.

    Args:
        t:                  Temperature
        rho:                Ambient density
        qc:                 Cloud specific mass
        qr:                 Rain water specific mass
        nc:                 Cloud water number concentration

    Return:                 Conversion rate
    """
    QMIN_AC = wpfloat(1.0e-6)  # threshold for auto conversion
    TAU_MAX = wpfloat(0.90e0)  # maximum allowed value of tau
    TAU_MIN = wpfloat(1.0e-30)  # minimum allowed value of tau
    A_PHI = wpfloat(6.0e2)  # constant in phi-function for autoconversion
    B_PHI = wpfloat(0.68e0)  # exponent in phi-function for autoconversion
    C_PHI = wpfloat(5.0e-5)  # exponent in phi-function for accretion
    X3 = wpfloat(2.0e0)  # gamma exponent for cloud distribution
    X2 = wpfloat(2.6e-10)  # separating mass between cloud and rain
    X1 = wpfloat(9.44e9)  # kernel coeff for SB2001 autoconversion
    AU_KERNEL = (
        X1
        / (wpfloat(20.0) * X2)
        * (X3 + wpfloat(2.0))
        * (X3 + wpfloat(4.0))
        / ((X3 + wpfloat(1.0)) * (X3 + wpfloat(1.0)))
    )

    x = log(minimum(AesGraupelConsts.rhox_mx, maximum(AesGraupelConsts.rhox_mn, rho * qr)))
    ac_kernel = AesGraupelConsts.a_ac_1 + x * (
        AesGraupelConsts.a_ac_2
        + x
        * (AesGraupelConsts.a_ac_3 + x * (AesGraupelConsts.a_ac_4 + x * AesGraupelConsts.a_ac_5))
    )
    tau = maximum(TAU_MIN, minimum(wpfloat(1.0) - qc / where(qc + qr > wpfloat(0.0), qc + qr, wpfloat(1.0)), TAU_MAX))
    phi = power(tau, B_PHI)
    phi = A_PHI * phi * power(wpfloat(1.0) - phi, wpfloat(3.0))
    xau = (
        AU_KERNEL
        * power(qc * qc / nc, wpfloat(2.0))
        * (wpfloat(1.0) + phi / power(wpfloat(1.0) - tau, wpfloat(2.0)))
    )
    xac = ac_kernel * qc * qr * power(tau / (tau + C_PHI), wpfloat(4.0))
    return where((qc > QMIN_AC) & (t > GraupelConsts.tfrz_hom), xau + xac, wpfloat(0.0))



@gtx.field_operator
def _sink_saturation_guarded(
    t: tuple[
        fa.CellKField[ta.wpfloat],
        fa.CellKField[ta.wpfloat],
        fa.CellKField[ta.wpfloat],
        fa.CellKField[ta.wpfloat],
    ],
    x: fa.CellKField[ta.wpfloat],
    dt: ta.wpfloat,
    where_: fa.CellKField[bool],
):
    sink = where(where_, t[0] + t[1] + t[2] + t[3], wpfloat(0.0))
    stot = x / dt
    sink_saturated = (sink > stot) & (x > GraupelConsts.qmin)
    safe_sink = where(sink_saturated, sink, wpfloat(1.0))
    t0 = where(sink_saturated, t[0] * stot / safe_sink, t[0])
    t1 = where(sink_saturated, t[1] * stot / safe_sink, t[1])
    t2 = where(sink_saturated, t[2] * stot / safe_sink, t[2])
    t3 = where(sink_saturated, t[3] * stot / safe_sink, t[3])
    sink = where(sink_saturated, t0 + t1 + t2 + t3, sink)
    return sink, t0, t1, t2, t3


@gtx.field_operator
def _q_t_update_guarded(  # noqa: PLR0917 [too-many-positional-arguments]
    t: fa.CellKField[ta.wpfloat],
    p: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
    q: Q,
    dt: ta.wpfloat,
    qnc: ta.wpfloat,
    enable_masking: bool,
) -> tuple[
    Q,
    fa.CellKField[ta.wpfloat],
]:
    if enable_masking:
        is_sig_present = maximum(q.g, maximum(q.i, q.s)) > GraupelConsts.qmin
    else:
        is_sig_present = broadcast(True, (dims.CellDim, dims.KDim))

    dvsw = q.v - _qsat_rho(t, rho)
    qvsi = _qsat_ice_rho(t, rho)
    dvsi = q.v - qvsi
    n_snow = _snow_number(t, rho * q.s)
    l_snow = _snow_lambda_guarded(rho * q.s, n_snow)

    t_below_tmelt = t < ThermodynamicConsts.tmelt
    t_at_least_tmelt = ~t_below_tmelt

    # Define conversion 'matrix'
    c2r = _cloud_to_rain_guarded(t, rho, q.c, q.r, qnc)
    r2v = _rain_to_vapor(t, rho, q.c, q.r, dvsw, dt)
    c2i, i2c = symmetric(_cloud_x_ice(t, q.c, q.i, dt))

    c2s = _cloud_to_snow(t, q.c, q.s, n_snow, l_snow)
    c2g = _cloud_to_graupel_guarded(t, rho, q.c, q.g)

    c2r = where(t_at_least_tmelt, c2r + c2s + c2g, c2r)
    c2s = where(t_at_least_tmelt, wpfloat(0.0), c2s)
    c2g = where(t_at_least_tmelt, wpfloat(0.0), c2g)

    n_ice = _ice_number(t, rho)
    m_ice = _ice_mass(q.i, n_ice)
    x_ice = _ice_sticking(t)

    eta = where(t_below_tmelt, _deposition_factor(t, qvsi), wpfloat(0.0))
    v2i, i2v = cond_symmetric(
        t_below_tmelt & is_sig_present, _vapor_x_ice(q.i, m_ice, eta, dvsi, rho, dt)
    )
    # ICON computes ice_dep from the vapor_x_ice deposition alone, BEFORE
    # the nucleation contribution is added (and only where is_sig_present)
    ice_dep = where(t_below_tmelt & is_sig_present, minimum(v2i, dvsi / dt), wpfloat(0.0))
    v2i = where(
        t_below_tmelt,
        v2i + _ice_deposition_nucleation(t, q.c, q.i, n_ice, dvsi, dt),
        wpfloat(0.0),
    )
    # TODO(): _deposition_auto_conversion yields roundoff differences in i2s
    i2s = where(
        t_below_tmelt & is_sig_present,
        _deposition_auto_conversion(q.i, m_ice, ice_dep) + _ice_to_snow(q.i, n_snow, l_snow, x_ice),
        wpfloat(0.0),
    )
    i2g = where(
        t_below_tmelt & is_sig_present, _ice_to_graupel_guarded(rho, q.r, q.g, q.i, x_ice), wpfloat(0.0)
    )
    s2g = where(t_below_tmelt & is_sig_present, _snow_to_graupel_guarded(t, rho, q.c, q.s), wpfloat(0.0))
    r2g = where(
        t_below_tmelt & is_sig_present,
        _rain_to_graupel(t, rho, q.c, q.r, q.i, q.s, m_ice, dvsw, dt),
        wpfloat(0.0),
    )

    dvsw0 = q.v - _qsat_rho_tmelt(rho)
    v2s, s2v = cond_symmetric(
        is_sig_present,
        _vapor_x_snow_guarded(t, p, rho, q.s, n_snow, l_snow, eta, ice_dep, dvsw, dvsi, dvsw0, dt),
    )

    v2g, g2v = cond_symmetric(
        is_sig_present, _vapor_x_graupel_guarded(t, p, rho, q.g, dvsw, dvsi, dvsw0, dt)
    )

    s2r = where(is_sig_present, _snow_to_rain_guarded(t, p, rho, dvsw0, q.s), wpfloat(0.0))
    g2r = where(is_sig_present, _graupel_to_rain_guarded(t, p, rho, dvsw0, q.g), wpfloat(0.0))

    # The following transitions are not physically meaningful, would be 0.0 in other implementation
    # here they are simply never used:
    # identity transitions v_v, c_c, ... g_g
    # unphysical transitions: v_c, v_r, c_v, r_c, r_s, r_i, s_c, s_i, i_r, g_c, g_s, g_i
    # Physical: v_s, v_i, v_g, c_r, c_s, c_i, c_g, r_v, r_g, s_v, s_r, s_g, i_v, i_c, i_s, i_g, g_v, g_r
    # SINK calculation

    UNUSED = broadcast(wpfloat(0.0), (dims.CellDim, dims.KDim))
    EVERYWHERE = broadcast(True, (dims.CellDim, dims.KDim))
    sink_v, v2s, v2i, v2g, _ = _sink_saturation_guarded((v2s, v2i, v2g, UNUSED), q.v, dt, where_=EVERYWHERE)
    sink_c, c2r, c2s, c2i, c2g = _sink_saturation_guarded((c2r, c2s, c2i, c2g), q.c, dt, where_=EVERYWHERE)
    sink_r, r2v, r2g, _, __ = _sink_saturation_guarded(
        (r2v, r2g, UNUSED, UNUSED), q.r, dt, where_=EVERYWHERE
    )
    sink_s, s2v, s2r, s2g, _ = _sink_saturation_guarded(
        (s2v, s2r, s2g, UNUSED), q.s, dt, where_=is_sig_present
    )
    sink_i, i2v, i2c, i2s, i2g = _sink_saturation_guarded(
        (i2v, i2c, i2s, i2g), q.i, dt, where_=is_sig_present
    )
    sink_g, g2v, g2r, _, __ = _sink_saturation_guarded(
        (g2v, g2r, UNUSED, UNUSED), q.g, dt, where_=is_sig_present
    )

    # water content updates:
    # Physical: v_s, v_i, v_g, c_r, c_s, c_i, c_g, r_v, r_g, s_v, s_r, s_g, i_v, i_c, i_s, i_g, g_v, g_r
    dqdt_v = r2v + s2v + i2v + g2v - sink_v  # Missing: c2v
    qv = maximum(wpfloat(0.0), q.v + dqdt_v * dt)
    dqdt_c = i2c - sink_c  # Missing: v2c, r2c, s2c, g2c
    qc = maximum(wpfloat(0.0), q.c + dqdt_c * dt)
    dqdt_r = c2r + s2r + g2r - sink_r  # Missing: v2r + i2r
    qr = maximum(wpfloat(0.0), q.r + dqdt_r * dt)
    dqdt_s = v2s + c2s + i2s - sink_s  # Missing: r2s + g2s
    qs = maximum(wpfloat(0.0), q.s + dqdt_s * dt)
    dqdt_i = v2i + c2i - sink_i  # Missing: r2i + s2i + g2i
    qi = maximum(wpfloat(0.0), q.i + dqdt_i * dt)
    dqdt_g = v2g + c2g + r2g + s2g + i2g - sink_g
    qg = maximum(wpfloat(0.0), q.g + dqdt_g * dt)

    qice = qs + qi + qg
    qliq = qc + qr
    qtot = qv + qice + qliq

    cv = (
        ThermodynamicConsts.cvd
        + (ThermodynamicConsts.cvv - ThermodynamicConsts.cvd) * qtot
        + (ThermodynamicConsts.clw - ThermodynamicConsts.cvv) * qliq
        + (GraupelConsts.ci - ThermodynamicConsts.cvv) * qice
    )
    t = (
        t
        + dt
        * (
            (dqdt_c + dqdt_r)
            * (GraupelConsts.lvc - (ThermodynamicConsts.clw - ThermodynamicConsts.cvv) * t)
            + (dqdt_i + dqdt_s + dqdt_g)
            * (GraupelConsts.lsc - (GraupelConsts.ci - ThermodynamicConsts.cvv) * t)
        )
        / cv
    )
    return Q(v=qv, c=qc, r=qr, s=qs, i=qi, g=qg), t




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
        mask, _q_t_update_guarded(te_sa, p, rho, q, dt, qnc, enable_masking=True), (q, te_sa)
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
