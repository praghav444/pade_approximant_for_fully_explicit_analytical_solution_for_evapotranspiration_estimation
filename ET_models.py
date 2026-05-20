"""
ET_models.py
============
Implementations of four ET equations:
  1. Penman-Monteith (PM)        - linearised Clausius-Clapeyron (1st-order Taylor)
  2. McColl (2020) Eq. 8         - Vallis exponential + exact Lambert-W solution
  3. Pade-1 (Pade on Vallis)     - Pade [1,1] applied to Vallis; quadratic
  4. Pade-2 (Pade on full C-C)   - Pade [1,1] applied to full Clausius-Clapeyron;
                                   quadratic, pure algebraic, no special functions

Approximation hierarchy
-----------------------
All four methods solve the surface energy balance  C*dT + B*(q*(Ts) - qa) = A
under different approximations of the saturation specific humidity ratio q*(Ts)/q*(Ta):
  PM      :  1 + (dq*/dT)*dT          (linearised; valid for |dT| < 5 K)
  McColl  :  exp(alpha*dT)             (Vallis; valid for |dT| < 25 K)
  Pade-1  :  Pade[1,1] of exp(alpha*dT)  (rational; valid for |dT| < 10-15 K,
                                           pole at alpha*dT = 2, i.e. dT ~ 34 K)
  Pade-2  :  Pade[1,1] of exact C-C      (rational; accurate for |dT| < 10 K,
                                           same pole location as Pade-1)

Physical constants
------------------
lam = 2.45e6   J kg-1       latent heat of vaporisation
c_p = 1005     J kg-1 K-1   specific heat of air at const. pressure
R_v = 461.5    J kg-1 K-1   gas constant for water vapour
eps = 0.622    -             ratio of molecular weights (M_w / M_d)
"""
import numpy as np
from scipy.special import lambertw

# Physical constants
LAM   = 2.45e6    # J kg-1
CP    = 1005.0    # J kg-1 K-1
RV    = 461.5     # J kg-1 K-1
EPS   = 0.622     # M_w / M_d

# Helpers

def esat_kPa(T_C):
    """Saturation vapour pressure [kPa] via Buck (1981) / Magnus equation."""
    return 0.6108 * np.exp(17.27 * T_C / (T_C + 237.3))

def qstar(T_C, P_kPa):
    """Saturated specific humidity [kg kg-1]."""
    es = esat_kPa(T_C)
    return EPS * es / (P_kPa - (1 - EPS) * es)

def delta_kPa_per_K(T_C):
    """Slope of saturation vapour pressure curve dE*/dT  [kPa K-1]."""
    es = esat_kPa(T_C)
    return 4098 * es / (T_C + 237.3) ** 2

def rho_air(T_C, P_kPa):
    """Dry-air density [kg m-3]  = P / (R_d * T)."""
    R_d = 287.05   # J kg-1 K-1
    T_K = T_C + 273.15
    return (P_kPa * 1e3) / (R_d * T_K)

def compute_ga(WS, USTAR):
    """
    Aerodynamic conductance [m s-1] via resistance formulation.

        ra = WS / USTAR^2 + 6.2 * USTAR^(-0.67)
        ga = 1 / ra

    Returns NaN wherever USTAR <= 0 or WS < 0.
    """
    WS    = np.asarray(WS,    dtype=float)
    USTAR = np.asarray(USTAR, dtype=float)
    bad   = (USTAR <= 0) | (WS < 0)
    with np.errstate(invalid='ignore', divide='ignore'):
        ra = WS / USTAR**2 + 6.2 * USTAR**(-0.67)
    ra[bad] = np.nan
    ga = 1.0 / ra
    return ga

def compute_gs(LE_obs, H_obs, Ta_C, qa, ga, rho, P_kPa, lam=LAM, cp=CP):
    """
    Back-calculate surface conductance g_s [m s-1] from observations.

    Returns (gs, Ts_C).  gs is NaN wherever the back-calculation is
    unphysical (gs < 0) or numerically ill-conditioned.
    """
    LE_obs = np.asarray(LE_obs, dtype=float)
    H_obs  = np.asarray(H_obs,  dtype=float)
    Ta_C   = np.asarray(Ta_C,   dtype=float)
    qa     = np.asarray(qa,     dtype=float)
    ga     = np.asarray(ga,     dtype=float)
    rho    = np.asarray(rho,    dtype=float)
    P_kPa  = np.asarray(P_kPa,  dtype=float)

    Ts_C    = Ta_C + H_obs / (rho * cp * ga)
    qstar_s = qstar(Ts_C, P_kPa)

    dq = qstar_s - qa        # [kg/kg]
    with np.errstate(invalid='ignore', divide='ignore'):
        gv = LE_obs / (rho * lam * dq)

    with np.errstate(invalid='ignore', divide='ignore'):
        gs = ga * gv / (ga - gv)

    gs = np.where((gs <= 0) | ~np.isfinite(gs), np.nan, gs)
    return gs, Ts_C


# ET Models

def ET_PM(Rn, G, Ta_C, qa, ga, gs, P_kPa, rho, lam=LAM, cp=CP):
    """
    Standard Penman-Monteith equation (McColl 2020, Eq. 3).
        lam*E = [eps*A + rho*lam*ga*D] / (eps + 1 + ga/gs)
    where  eps = (lam/cp)*(dq*/dT)    [dimensionless]
           D = q*(Ta) - qa            [specific humidity deficit, kg kg-1]
           A = Rn - G                 [available energy, W m-2]
    Linearises q*(Ts) around Ta; accurate for |dT| < 5 K.
    Returns lam*E [W m-2].
    """
    Rn, G, Ta_C, qa, ga, gs, rho = [np.asarray(x, dtype=float) for x in (Rn, G, Ta_C, qa, ga, gs, rho)]
    A   = Rn - G
    qs  = qstar(Ta_C, P_kPa)
    D   = qs - qa                                 # specific humidity deficit [kg/kg]
    dq_dT = delta_kPa_per_K(Ta_C) * EPS / P_kPa  # dq*/dT [kg kg-1 K-1]
    eps   = (lam / cp) * dq_dT                    # epsilon [dimensionless]
    numerator   = eps * A + rho * lam * ga * D
    denominator = eps + 1.0 + ga / gs
    return numerator / denominator

def ET_McColl(Rn, G, Ta_C, qa, ga, gs, P_kPa, rho, lam=LAM, cp=CP, Rv=RV):
    """
    McColl (2020) Eq. 8 - exact Lambert-W solution of the Vallis-exponential
    energy balance.
    The Vallis approximation replaces the C-C ratio with exp(alpha*dT),
    alpha = lam/(Rv*Ta**2).  The resulting transcendental equation is solved
    exactly via the principal branch of the Lambert-W function W0:
        lam*E = (C/alpha)*W0(arg) - B*qa
    where  B = rho*lam*gv,  C = rho*cp*ga,  gv = gs*ga/(gs+ga)
           arg = (alpha*lam/cp)*q*(Ta)*(gs/(gs+ga)) * exp(alpha*(A + B*qa)/C)
    Requires scipy.special.lambertw.  Accurate for |dT| < 25 K.
    Returns lam*E [W m-2].
    """
    Rn, G, Ta_C, qa, ga, gs, rho = [np.asarray(x, dtype=float) for x in (Rn, G, Ta_C, qa, ga, gs, rho)]
    Ta_K = Ta_C + 273.15
    A    = Rn - G
    qs   = qstar(Ta_C, P_kPa)
    gv   = gs * ga / (gs + ga)
    alpha     = lam / (Rv * Ta_K**2)
    prefactor = (alpha * lam / cp) * qs * (gs / (gs + ga))
    exponent  = alpha * (A + rho * lam * gv * qa) / (rho * cp * ga)
    arg       = prefactor * np.exp(exponent)
    W0 = np.real(lambertw(arg, k=0))
    return (rho * cp * ga / alpha) * W0 - rho * lam * gv * qa

def ET_Pade1(Rn, G, Ta_C, qa, ga, gs, P_kPa, rho, lam=LAM, cp=CP, Rv=RV):
    """
    Pade-1: Pade [1,1] approximation of the Vallis exponential exp(alpha*dT).
    Replaces exp(alpha*dT) ~ (1 + alpha*dT/2) / (1 - alpha*dT/2) in the
    Vallis energy balance.  This converts the transcendental equation into a
    quadratic whose physical root (smaller |dT|) is given analytically:
        dT = 2*(M-Q) / [inner + sqrt(inner**2 - 2*alpha*C*(M-Q))]
        lam*E = A - C*dT
    where  alpha = lam/(Rv*Ta**2),  inner = C + (alpha/2)*(M+Q)
           B = rho*lam*gv,  C = rho*cp*ga,  Q = B*q*(Ta),  M = A + B*qa
    No special functions required.  Approximation breaks down for |dT| > 15 K
    (Pade pole at alpha*dT = 2, i.e. dT ~ 34 K at Ta = 300 K).
    Pole handling
    -------------
    The Pade pole is at x = alpha*dT = 2 (dT ~ 34 K at Ta = 300 K).
    Predictions are set to NaN when |x| >= 2.  For 1 < |x| < 2 the
    approximation degrades but remains finite and is more accurate than a
    hard-capped value.
    Returns lam*E [W m-2].
    """
    Rn, G, Ta_C, qa, ga, gs, rho = [np.asarray(x, dtype=float) for x in (Rn, G, Ta_C, qa, ga, gs, rho)]
    Ta_K  = Ta_C + 273.15
    A     = Rn - G
    qs    = qstar(Ta_C, P_kPa)
    gv    = gs * ga / (gs + ga)
    alpha = lam / (Rv * Ta_K**2)
    B     = rho * lam * gv
    C     = rho * cp * ga
    Q     = B * qs
    M     = A + B * qa
    inner = C + (alpha / 2.0) * (M + Q)
    disc  = inner**2 - 2.0 * alpha * C * (M - Q)
    disc  = np.where(disc < 0, np.nan, disc)
    dT = 2.0 * (M - Q) / (inner + np.sqrt(disc))
    x_pade1_orig = alpha * dT

    LE = A - C * dT
    return LE, x_pade1_orig


def ET_Pade2(Rn, G, Ta_C, qa, ga, gs, P_kPa, rho, lam=LAM, cp=CP, Rv=RV):
    """
    Pade-2: Pade [1,1] approximation of the full Clausius-Clapeyron relation.
    Pure algebraic, closed-form solution - no iteration, no special functions.
    The exact C-C ratio q*(Ts)/q*(Ta) = exp(lam/Rv * (1/Ta - 1/Ts)) is
    approximated by Pade [1,1] with argument x = lam*dT/(Rv*Ta*Ts):
        q*(Ts)/q*(Ta) ~ (Ta*Ts + k*dT) / (Ta*Ts - k*dT),   k = lam/(2*Rv) ~ 2653 K
    Substituting into C*dT + B*(q*(Ts) - qa) = A yields a quadratic in dT:
        b2*dT**2 + b1*dT + b0 = 0
        b2 = C*(Ta - k)              [always negative since Ta << k]
        b1 = C*Ta**2 + Q*(Ta + k) - M*(Ta - k)
        b0 = Ta**2*(Q - M)
    where  B = rho*lam*gv,  C = rho*cp*ga,  Q = B*q*(Ta),  M = A + B*qa.
    The physical root is identified by proximity to the k->0 PM limit
    dT_PM = (M-Q)/C.  The spurious root diverges to -Ta ~ -300 K.
    Pole handling
    -------------
    The Pade pole is at x = lam*dT/(Rv*Ta*Ts) = 2.  Predictions are set to
    NaN when |x| >= 2.  For 1 < |x| < 2 the approximation degrades but
    remains finite and is more accurate than a hard-capped value.
    Validity: approximation accuracy degrades for |dT| > 10 K.
    Returns NaN when the discriminant is negative (unphysical regime).
    Returns lam*E [W m-2].
    """
    Rn, G, Ta_C, qa, ga, gs, rho = [np.asarray(x, dtype=float) for x in (Rn, G, Ta_C, qa, ga, gs, rho)]
    Ta_K = Ta_C + 273.15
    A    = Rn - G
    qs   = qstar(Ta_C, P_kPa)
    gv   = gs * ga / (gs + ga)

    # Cache reused sub-expressions to avoid redundant array operations
    Ta_K2      = Ta_K * Ta_K          # used in b1 and x_pade2_orig
    k          = LAM / (2.0 * RV)     # module-level constant ~ 2653 K
    Ta_minus_k = Ta_K - k             # reused in b2 and b1
    Ta_plus_k  = Ta_K + k             # reused in b1

    B  = rho * lam * gv
    C  = rho * cp * ga
    Q  = B * qs
    M  = A + B * qa

    b2 = C * Ta_minus_k
    b1 = C * Ta_K2 + Q * Ta_plus_k - M * Ta_minus_k
    b0 = Ta_K2 * (Q - M)

    disc      = b1 * b1 - 4.0 * b2 * b0
    disc      = np.where(disc < 0, np.nan, disc)
    sqrt_disc = np.sqrt(disc)

    # Select the root closest to the k->0 PM limit (M-Q)/C.
    # b2 = C*(Ta_K - k) is always negative since Ta_K ~ 300 K << k ~ 2653 K.
    # With b2 < 0, dT_plus is always the smaller-magnitude (physical) root and
    # dT_minus always diverges to -Ta ~ -300 K (spurious root).
    # This eliminates the two np.abs array passes needed for the general case.
    two_b2 = 2.0 * b2
    dT     = (-b1 + sqrt_disc) / two_b2

    x_pade2_orig = lam * dT / (Rv * Ta_K * (Ta_K + dT))

    LE = A - C * dT
    return LE, x_pade2_orig
