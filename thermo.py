# -*- coding: utf-8 -*-
""" Aerosol/atmospheric thermodynamics functions.

The following sets of functions calculate useful thermodynamic quantities
that arise in aerosol-cloud studies. Where possible, the source of the
parameterization for each function is documented.

"""
import numpy as np
from scipy.optimize import fminbound

from constants import *

use_MK2005=True # use the empirical formulae for saturation vapour pressures and latent heats from Murphy and Koop 2005

# THERMODYNAMIC FUNCTIONS


def dv_cont(T, P):
    """Diffusivity of water vapor in air, neglecting non-continuum effects.

    See :func:`dv` for details.

    Parameters
    ----------
    T : float
        ambient temperature of air surrounding droplets, K
    P : float
        ambient pressure of surrounding air, Pa

    Returns
    -------
    float
        :math:`D_v(T, P)` in m^2/s

    See Also
    --------
    dv : includes correction for non-continuum effects

    """
    P_atm = P * 1.01325e-5  # Pa -> atm
    return 1e-4 * (0.211 / P_atm) * ((T / 273.0) ** 1.94)


def dv(T, r, P, accom=ac):
    """Diffusivity of water vapor in air, modified for non-continuum effects.

    The diffusivity of water vapor in air as a function of temperature and pressure
    is given by

    .. math::
        \\begin{equation}
        D_v = 10^{-4}\\frac{0.211}{P}\left(\\frac{T}{273}\\right)^{1.94}
              \\tag{SP2006, 17.61}
        \end{equation}

    where :math:`P` is in atm [SP2006]. Aerosols much smaller than the mean free path
    of the  air surrounding them (:math:`K_n >> 1`) perturb the flow around them
    moreso than larger particles, which affects this value. We account for corrections
    to :math:`D_v` in the non-continuum regime via the parameterization

    .. math::
        \\begin{equation}
        D'_v = \\frac{D_v}{1+ \\frac{D_v}{\\alpha_c r}
                \left(\\frac{2\pi M_w}{RT}\\right)^{1/2}} \\tag{SP2006, 17.62}
        \end{equation}

    where :math:`\\alpha_c` is the condensation coefficient (:const:`constants.ac`).

    Parameters
    ----------
    T : float
        ambient temperature of air surrounding droplets, K
    r : float
        radius of aerosol/droplet, m
    P : float
        ambient pressure of surrounding air, Pa
    accom : float, optional (default=:const:`constants.ac`)
        condensation coefficient

    Returns
    -------
    float
        :math:`D'_v(T, r, P)` in m^2/s

    References
    ----------

    .. [SP2006] Seinfeld, John H, and Spyros N Pandis. Atmospheric Chemistry
       and Physics: From Air Pollution to Climate Change. Vol. 2nd. Wiley, 2006.

    See Also
    --------
    dv_cont : neglecting correction for non-continuum effects

    """
    dv_t = dv_cont(T, P)
    denom = 1.0 + (dv_t / (accom * r)) * np.sqrt((2.0 * np.pi * Mw) / (R * T))
    return dv_t / denom


def rho_air(T, P, RH=1.0):
    """Density of moist air with a given relative humidity, temperature, and pressure.

    Uses the traditional formula from the ideal gas law (3.41)[Petty2006].

    .. math::
        \\begin{equation}
        \\rho_a = \\frac{P}{R_d T_v}
        \end{equation}

    where :math:`T_v = T(1 + 0.61w)` and :math:`w` is the water vapor mixing ratio.

    Parameters
    ----------
    T : float
        ambient air temperature, K
    P : float
        ambient air pressure, Pa
    RH : float, optional (default=1.0)
        relative humidity, decimal

    Returns
    -------
    float
        :math:`\\rho_{a}` in kg m**-3

    References
    ----------

    .. [Petty2006] Petty, Grant Williams. A First Course in Atmospheric Radiation.
       Sundog Publishing, 2006. Print.

    """
    qsat = RH * 0.622 * (es(T) / P)
    Tv = T * (1.0 + 0.61 * qsat)
    rho_a = P / Rd / Tv  # air density

    return rho_a

def rho_air_dry(P,T,S):
    e=(S+1)*es(T) # actual vapour pressure of water
    rho_ad=(P - e) / Rd / T
    return rho_ad
 
def es(T):
    """Calculates the saturation vapor pressure over water for a given temperature.

    If use_MK2005, use empirical fit from Murphy & Koop 2005 (eq 10) - handles better range of temperatures
 
    Otherwise uses an empirical fit [Bolton1980], which is accurate to :math:`0.1\%` over the
    temperature range :math:`-30^oC \leq T \leq 35^oC`,

    .. math::
        \\begin{equation}
        e_s(T) = 611.2 \exp\left(\\frac{17.67T}{T + 243.5}\\right) \\tag{RY1989, 2.17}
        \end{equation}

    where :math:`e_s` is in Pa and :math:`T` is in K.

    Parameters
    ----------
    T : float
        ambient air temperature, degrees K

    Returns
    -------
    float
        :math:`e_s(T)` in Pa

    References
    ----------

    .. [Bolton1980] Bolton, David. "The Computation of Equivalent Potential
       Temperature". Monthly Weather Review 108.8 (1980): 1046-1053

    .. [RY1989] Rogers, R. R., and M. K. Yau. A Short Course in Cloud Physics.
       Burlington, MA: Butterworth Heinemann, 1989.

    """
    if use_MK2005:
        tanh_part=np.tanh(0.0415*(T - 218.8))*(53.878 - 1331.22/T - 9.44523*np.log(T) + 0.014025*T)
        e_s = np.exp(54.842763 - 6763.22/T - 4.210*np.log(T) + 0.000367*T + tanh_part)
    else:
        T_c=T-273.15
        e_s = 611.2 * np.exp(17.67 * T_c / (T_c + 243.5))

    return e_s


def p_ice(T):
    """Calculates the saturation vapor pressure in Pa over ice for a given temperature.
       T: float, ambient temperature in Kelvin
       From Murphy & Koop 2005 (eq 7)
    """
    p_i=np.exp(9.550426 - 5723.265/T + 3.53068 * np.log(T) - 0.00728332*T) # saturation vapour pressure above a flat surface of ice
    return p_i

def es_from_p_ice(T):
    """Calculates the saturation vapor pressure in Pa over water for a given temperature using p_ice.
       T: float, ambient temperature in Kelvin
       From Koop et al 2000 (Murphy & Koop 2005 Appendix A)
    """
    e_s=p_ice(T)*np.exp(-(210368 + 131.438*T - 3.32373e6/T - 41729.1*np.log(T))/(R*T))
    return e_s
      
def L_ice_sub(T):
    """ calculates latent heat of sublimation of ice for a given temperature
       T: float, ambient temperature in Kelvin
       From Murphy & Koop 2005 (eq 5) and turned into J/kg
    """
    L_ice_sub=(46782.5 + 35.8925*T - 0.07414*(T**2) + 541.5*np.exp(-(T /123.75)**2))/Mw
    return L_ice_sub

def L_freezing(T):
    if use_MK2005:
        # freezing of pure water - see Muphy & Koop 2005
        L_freezing= 333000.4 + 2076.0*(T-273.15)
    else:
        L_freezing=Li
    return L_freezing


def L_cond(T):
    if use_MK2005:
        """ calculates latent heat of condensation of water for a given temperature
           T: float, ambient temperature in Kelvin
           From Murphy & Koop 2005 (eq 9) and turned into J/kg
        """
        L_cond=(56579 - 42.212*T + np.exp(0.1149*(281.6-T )))/Mw
    else:
        L_cond=L
    return L_cond

def Cp_ice(T):
    """ calculates heat capacity of ice for a given temperature
       T: float, ambient temperature in Kelvin
       From Murphy & Koop 2005 (eq 4) and turned into J/kg per K
    """
    cp_ice =-2.0572 + 0.14644*T + 0.06163*T*np.exp(-(T /125.1)**2)
    return cp_ice/Mw

def Cp_vapour(T):
    """ calculates heat capacity of water vapour for a given temperature
       T: float, ambient temperature in Kelvin
       From Murphy & Koop 2005 (eq 4 and text following) and turned into J/kg per K
    """
    cp_v = -3.5319 + 0.14457*T + 0.06155*T*np.exp(-(T/129.85)**2)
    return cp_v/Mw

def saturation_ratio(P,T,wv):
    """Calculates the saturation ratio for a given pressure, temperature and 
       water vapour mass mixing ratio.

   Parameters
    ----------
    P : float
        ambient air pressure, Pa
    T : float
        ambient air temperature, K
    ww : float
        water vapour mass mixing ratio, kg/kg

    Returns
    -------
    float
        : supersaturatiom
    """
    e_s = es(T)
    ws = epsilon * e_s / (P - e_s) # Accurate formulation for mixing ratio
    return wv / ws

def wv_from_saturation_ratio(P,T,s):
    e_s = es(T)
    ws = epsilon * e_s / (P - e_s) # Accurate formulation for mixing ratio
    wv = s*ws
    return wv

def ka_cont(T):
    """Thermal conductivity of air, neglecting non-continuum effects.

    See :func:`ka` for details.

    Parameters
    ----------
    T :
        ambient air temperature surrounding droplet, K

    Returns
    -------
    float
        :math:`k_a(T)` in J/m/s/K

    See Also
    --------
    ka : includes correction for non-continuum effects.

    """
    return 1e-3 * (4.39 + 0.071 * T)

# I found this formula for k air when searching about handling heat coming from walls
def ka_cont_google(T):
    ka = 1.5207e-11*T**3 - 4.8574e-8*T**2 + 1.0184e-4*T - 3.933e-4
    return ka

def ka(T, rho, r):
    """Thermal conductivity of air, modified for non-continuum effects.

    The thermal conductivity of air is given by

    .. math::
        \\begin{equation}
        k_a = 10^{-3}(4.39 + 0.071T) \\tag{SP2006, 17.71}
        \end{equation}

    Modification to account for non-continuum effects (small aerosol/droplet
    size) yields the equation

    .. math::
        \\begin{equation}
        k'_a = \\frac{k_a}{1 + \\frac{k_a}{\\alpha_t r_p \\rho C_p}
               \\frac{2\pi M_a}{RT}^{1/2}} \\tag{SP2006, 17.72}
        \end{equation}

    where :math:`\\alpha_t` is a thermal accommodation coefficient
    (:const:`constants.at`).

    Parameters
    ----------
    T : float
        ambient air temperature, K
    rho : float
        ambient air density, kg/m^3
    r : float
        droplet radius, m

    Returns
    -------
    float
        :math:`k'_a(T, \\rho, r)` in J/m/s/K

    References
    ----------

    .. [SP2006] Seinfeld, John H, and Spyros N Pandis. Atmospheric Chemistry
       and Physics: From Air Pollution to Climate Change. Vol. 2nd. Wiley, 2006.

    See Also
    --------
    ka_cont : neglecting correction for non-continuum effects

    """
    ka_t = ka_cont(T)
    denom = 1.0 + (ka_t / (at * r * rho * Cp)) * np.sqrt((2.0 * np.pi * Ma) / (R * T))
    return ka_t / denom


def sigma_w(T):
    """Surface tension of water for a given temperature.

    .. math::
        \\begin{equation}
        \sigma_w = 0.0761 - 1.55\\times 10^{-4}(T - 273.15)
        \end{equation}

    Parameters
    ----------
    T : float
        ambient air temperature, degrees K

    Returns
    -------
    float
        :math:`\sigma_w(T)` in J/m^2

    """
    return 0.0761 - 1.55e-4 * (T - 273.15)


# KOHLER THEORY FUNCTIONS


def Seq(r, r_dry, T, kappa):
    """ κ-Kohler theory equilibrium saturation over aerosol.

    Calculates the equilibrium supersaturation (relative to 100% RH) over an
    aerosol particle of given dry/wet radius and of specified hygroscopicity
    bathed in gas at a particular temperature

    Following the technique of [PK2007], classical
    Kohler theory can be modified to account for the hygroscopicity of an aerosol
    particle using a single parameter, :math:`\kappa`. The modified theory predicts
    that the supersaturation with respect to a given aerosol particle is,

    .. math::
        S_\\text{eq} &= a_w \exp \\left( \\frac{2\sigma_{w} M_w}{RT\\rho_w r} \\right)\\\\
        a_w &= \\left(1 + \kappa\\left(\\frac{r_d}{r}^3\\right) \\right)^{-1}

    with the relevant thermodynamic properties of water defined elsewhere in this
    module, :math:`r_d` is the particle dry radius (``r_dry``), :math:`r` is the
    radius of the droplet containing the particle (``r``), :math:`T` is the temperature
    of the environment (``T``), and :math:`\kappa` is the hygroscopicity parameter
    of the particle (``kappa``).


    Parameters
    ----------
    r : float
        droplet radius, m
    r_dry : float
        dry particle radius, m
    T : float
        ambient air temperature, K
    kappa: float
        particle hygroscopicity parameter

    Returns
    -------
    float
        :math:`S_\\text{eq}` for the given aerosol/droplet system

    References
    ----------

    .. [PK2007] Petters, M. D., and S. M. Kreidenweis. "A Single Parameter
        Representation of Hygroscopic Growth and Cloud Condensation Nucleus
        Activity." Atmospheric Chemistry and Physics 7.8 (2007): 1961-1971

    See Also
    --------
    Seq_approx : compute equilibrium supersaturation using an approximation
    kohler_crit : compute critical radius and equilibrium supersaturation

    """
    A = (2.0 * Mw * sigma_w(T)) / (R * T * rho_w * r)
    B = (r**3 - (r_dry**3)) / (r**3 - (r_dry**3) * (1.0 - kappa))
    s = np.exp(A) * B - 1.0
    return s


def Seq_approx(r, r_dry, T, kappa):
    """Approximate κ-Kohler theory equilibrium saturation over aerosol.

    Calculates the equilibrium supersaturation (relative to 100% RH) over an
    aerosol particle of given dry/wet radius and of specified hygroscopicity
    bathed in gas at a particular temperature, using a simplified expression
    derived by Taylor-expanding the original equation,

    .. math::
        S_\\text{eq} = \\frac{2\sigma_{w} M_w}{RT\\rho_w r} - \kappa\\frac{r_d^3}{r^3}

    which is valid when the equilibrium supersaturation is small, i.e. in
    most terrestrial atmosphere applications.

    Parameters
    ----------
    r : float
        droplet radius, m
    r_dry : float
        dry particle radius, m
    T : float
        ambient air temperature, K
    kappa: float
        particle hygroscopicity parameter

    Returns
    -------
    float
        :math:`S_\\text{eq}` for the given aerosol/droplet system

    References
    ----------

    See Also
    --------
    Seq : compute equilibrium supersaturation using full theory
    kohler_crit : compute critical radius and equilibrium supersaturation

    """
    A = (2.0 * Mw * sigma_w(T)) / (R * T * rho_w * r)
    return A - kappa * (r_dry**3) / (
        r**3
    )  # the minus 1.0 is built into this  expression


def kohler_crit(T, r_dry, kappa, approx=False):
    """Critical radius and supersaturation of an aerosol particle.

    The critical size of an aerosol particle corresponds to the maximum equilibrium
    supersaturation achieved on its Kohler curve. If a particle grows beyond this
    size, then it is said to "activate", and will continue to freely grow even
    if the environmental supersaturation decreases.

    This function computes the critical size and and corresponding supersaturation
    for a given aerosol particle. Typically, it will analyze :func:`Seq` for the
    given particle and numerically compute its inflection point. However, if the
    ``approx`` flag is passed, then it will compute the analytical critical point
    for the approximated kappa-Kohler equation.

    Parameters
    ----------
    T : float
        ambient air temperature, K
    r_dry : float
        dry particle radius, m
    kappa : float
        particle hygroscopicity parameter
    approx : boolean, optional (default=False)
        use the approximate kappa-kohler equation

    Returns
    -------
    (r_crit, s_crit) : tuple of floats
        Tuple of :math:`(r_\\text{crit},\, S_\\text{crit})`, the critical radius (m)
        and supersaturation of the aerosol droplet.

    See Also
    --------
    Seq : equilibrium supersaturation calculation

    """
    if approx:
        A = (2.0 * Mw * sigma_w(T)) / (R * T * rho_w)
        s_crit = np.sqrt((4.0 * (A**3)) / (27 * kappa * (r_dry**3)))
        r_crit = np.sqrt((3.0 * kappa * (r_dry**3)) / A)

    else:
        neg_Seq = lambda r: -1.0 * Seq(r, r_dry, T, kappa)
        out = fminbound(
            neg_Seq, r_dry, r_dry * 1e4, xtol=1e-10, full_output=True, disp=0
        )
        r_crit, s_crit = out[:2]
        s_crit *= -1.0  # multiply by -1 to undo negative flag for Seq

    return r_crit, s_crit


def critical_curve(T, r_a, r_b, kappa, approx=False):
    """Calculates curves of critical radii and supersaturations for aerosol.

    Calls :func:`kohler_crit` for values of ``r_dry`` between ``r_a`` and ``r_b``
    to calculate how the critical supersaturation changes with the dry radius for a
    particle of specified ``kappa``

    Parameters
    ----------
    T : float
        ambient air temperature, K
    r_a, r_b : floats
        left/right bounds of parcel dry radii, m
    kappa : float
        particle hygroscopicity parameter

    Returns
    -------
    rs, rcrits, scrits : np.ndarrays
        arrays containing particle dry radii (between ``r_a`` and ``r_b``)
        and their corresponding criticall wet radii and supersaturations

    See Also
    --------
    kohler_crit : critical supersaturation calculation

    """

    def crit_func(rd):
        kohler_crit(T, rd, kappa, approx)

    rs = np.logspace(np.log10(r_a), np.log10(r_b), 200)
    ss = np.array(list(map(crit_func, rs)))

    rcrits = ss[:, 0]
    scrits = ss[:, 1]

    return rs, rcrits, scrits


# MICROPHYSICS


def r_eff(rho, wc, Ni):
    """Calculates the cloud droplet effective radius given the parcel liquid
    water mixing ratio and the number of activated droplets, as well as the parcel
    air density

    Assuming the droplet population is monodisperse or close to it, the cloud droplet
    effective radius can be computed by

    .. math::
        \\begin{equation}
        r_{\\text{eff}} = \left(\\frac{3 \\rho_a w_c}{4 \pi N_i \\rho_w}\\right)^{1/3}
        \end{equation}

    Parameters
    ----------
    rho : float
        parcel air density, kg/m^3
    wc : float
        liquid water mixing ratio, kg/kg
    Ni : float
        droplet number concentration, m^-3

    Returns
    -------
    Cloud droplet effective radius, m

    .. warning::

        Not completely implemented yet.

    """
    return (3.0 * rho * wc / (4.0 * np.pi * rho_w * Ni)) ** (1.0 / 3.0)


