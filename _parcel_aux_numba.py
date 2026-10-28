import numba as nb
import numpy as np
from numba.pycc import CC
from thermo import use_MK2005
import constants as c


## Define double DTYPE
DTYPE = np.float64

PI = 3.14159265358979323846264338328

# AOT/numba stuff
auxcc = CC("parcel_aux_numba")
auxcc.verbose = True

## Auxiliary, single-value calculations with GIL released for derivative
## calculations
@nb.njit()
@auxcc.export("sigma_w", "f8(f8)")
def sigma_w(T):
    """See :func:`pyrcel.thermo.sigma_w` for full documentation"""
    return 0.0761 - (1.55e-4) * (T - 273.15)


@nb.njit()
@auxcc.export("ka_cont", "f8(f8)")
def ka_cont(T):
    """See :func:`pyrcel.thermo.ka_cont` for full documentation"""
    return 1e-3 * (4.39 + 0.071 * T)
    
@nb.njit()
@auxcc.export("ka", "f8(f8, f8, f8)")
def ka(T, r, rho):
    """See :func:`pyrcel.thermo.ka` for full documentation"""
    ka_cont = 1e-3 * (4.39 + 0.071 * T)
    denom = 1.0 + (ka_cont / (c.at * r * rho * c.Cp)) * np.sqrt(
        (2 * PI * c.Ma) / (c.R * T)
    )
    return ka_cont / denom


@nb.njit()
@auxcc.export("dv", "f8(f8, f8, f8, f8)")
def dv(T, r, P, accom):
    """See :func:`pyrcel.thermo.dv` for full documentation"""
    P_atm = P * 1.01325e-5  # Pa -> atm
    dv_cont = 1e-4 * (0.211 / P_atm) * ((T / 273.0) ** 1.94)
    denom = 1.0 + (dv_cont / (accom * r)) * np.sqrt((2 * PI * c.Mw) / (c.R * T))
    return dv_cont / denom


@nb.njit()
@auxcc.export("es", "f8(f8)")
def es(T):
    """See :func:`pyrcel.thermo.es` for full documentation"""
    if use_MK2005:
        tanh_part=np.tanh(0.0415*(T - 218.8))*(53.878 - 1331.22/T - 9.44523*np.log(T) + 0.014025*T)
        e_s = np.exp(54.842763 - 6763.22/T - 4.210*np.log(T) + 0.000367*T + tanh_part)
    else:
        T_c=T-273.15
        e_s = 611.2 * np.exp(17.67 * T_c / (T_c + 243.5))
    return e_s


@nb.njit()
@auxcc.export("Seq", "f8(f8, f8, f8)")
def Seq(r, r_dry, T, kappa):
    """See :func:`pyrcel.thermo.Seq` for full documentation."""
    A = (2.0 * c.Mw * sigma_w(T)) / (c.R * T * c.rho_w * r)
    B = 1.0
    if kappa > 0.0:
        B = (r**3 - (r_dry**3)) / (r**3 - (r_dry**3) * (1.0 - kappa))
    return np.exp(A) * B - 1.0

@nb.njit()
@auxcc.export("temperature_and_pressure_tendency", "f8[:](f8, f8, f8, f8, f8, f8, f8, f8, f8)")
def temperature_and_pressure_tendency(T0,T,P, chamber_vol, chamber_area, chamber_diameter, chamber_vol_flowrate):

    # this gets the temperature and pressure tendancy caused by air being pumped out and
    # by heat from the walls
    # dU_dt = dQ_dt + dH_dt where dQ_dt is change in energy due to heat transfer from the walls
    #                       and dH_dt is the change in energy due to adiabatic expansion (it is negative)
    # dU_dt = dQ_dt + Cp_mol.T.dn_dt
    Cp_mol = c.Cp*c.Ma # get molar Cp from specific Cp
    Cv_mol = Cp_mol - c.R

    # get convective heat transfer coefficent of air in chamber - assume laminar flow over the walls
    if chamber_area>0:
        Nu = 3.66 # Nusselt number
        h_int=Nu*ka_cont(T)/chamber_diameter
        dQ_dt=h_int*chamber_area*(T0-T)
    else:
        dQ_dt=0
    # U = n.Cv_mol.T
    # dU_dt = Cv_mol(T.dn_dt+n.dT_dt)
    # dT_dt = (dU_dt - Cv_mol.T.dn_dt)/(Cv_mol.n)
    # dT_dt = (dQ_dt + Cp_mol.T.dn_dt - Cv_mol.T.dn_dt)/(Cv_mol.n)
    # dT_dt = (dQ_dt + T.dn_dt(Cp_mol-Cv_mol)/(Cv_mol.n)
    # dT_dt = (dQ_dt + R.T.dn_dt)/(Cv_mol.n)
    # replace 1/n with R.T/P.V and R.T.dn_dt with -P.Vrate
    # dT_dt = R.T.(dQ_dt - P.Vrate)/(Cv_mol.P.V)
    dT_dt = c.R*T*(dQ_dt - P*chamber_vol_flowrate)/(Cv_mol*P*chamber_vol)

    # determining pressure tendency PV = nRT so nT = PV/R
    # U = Cv_mol.P.V/R (substitute nT for PV/R with V and R const)
    # dU_dt = Cv_mol.V.dP_dt/R
    # so Cv_mol.V.dP_dt/R = dQ_dt - Cp_mol.P.V_rate/R 
    # so dP_dt = R.(dQ_dt - Cp_mol.P.V_rate/R)/(Cv_mol.V)
    dP_dt = c.R*(dQ_dt - Cp_mol*P*chamber_vol_flowrate/c.R)/(Cv_mol*chamber_vol)
    return np.asarray([dT_dt, dP_dt])

## RHS Derivative callback function
@nb.njit(parallel=True)
@auxcc.export("parcel_ode_rhs", "f8[:](f8, f8[:], f8[:], f8, i4, f8[:], f8[:], f8[:], f8, f8[:])")
def parcel_ode_rhs(t, y, sw, T0, nr, r_drys, Nis, kappas, accom, Chamber):
    """Calculates the instantaneous time-derivative of the parcel model system.

    Given a current state vector `y` of the parcel model, computes the tendency
    of each term including thermodynamic (pressure, temperature, etc) and aerosol
    terms. The basic aerosol properties used in the model must be passed along
    with the state vector (i.e. if being used as the callback function in an ODE
    solver).

    Parameters
    ----------
    t : float
        Current simulation time, in seconds.
    y : array_like
        Current state of the parcel model system,
            * y[0] = Pressure, Pa
            * y[1] = temperature, K
            * y[2] = water vapor mass mixing ratio, kg/kg
            * y[3] = water condensed (liquid or ice)  mass mixing ratio, kg/kg
            * y[4:4+nr] = aerosol bin sizes (radii), m
    sw: - array_like bool indicating if freezing has triggered
    nr : Integer
        Number of aerosol radii being tracked.
    r_drys : array_like
        Array recording original aerosol dry radii, m.
    Nis : array_like
        Array recording aerosol number concentrations, 1/(kg of dry air).
    kappas : array_like
        Array recording aerosol hygroscopicities.
    accom : float, optional (default=:const:`constants.ac`)
        Condensation coefficient. 
    Chamber : floats
        holds the chamber set up - volume, area, diameter, vol flow rate

    Returns
    -------
    x : array_like
        Array of shape (``nr``+5, ) containing the evaluated parcel model
        instaneous derivative.

    Notes
    -----
    This function is implemented using numba; it does not need to be just-in-
    time compiled in order to function correctly, but it is set up ahead of time
    so that the internal loop over each bin growth term is parallelized.

    """

    P = y[0]
    T = y[1]
    wv = y[2]
    rs = y[c.N_STATE_VARS:]
 
    pv_sat = es(T)  # saturation vapor pressure
    # saturation ratio
    s = wv * (P-pv_sat) / (c.epsilon * pv_sat) # we don't need to do get tendancies for s as it depends on other variables
    S = s-1 # supersaturation

    ## Compute air densities from current state for ka calc
    Tv = (1.0 + 0.61 * wv) * T
    rho_air = P / c.Rd / Tv

    ## Begin computing tendencies
    ## HEAT FROM WALLS plus ADIABATIC COOLING - assume dry here and add on latent heat later
    tendencies = temperature_and_pressure_tendency(T0,T,P, Chamber[0], Chamber[1], Chamber[2], Chamber[3])
    dT_dt_dry=tendencies[0]
    dP_dt_dry=tendencies[1] 

    dT_dt_i=0.0
    dT_dt_l=0.0
    dwl_dt = 0.0
    dwi_dt = 0.0
    drs_dt = np.empty_like(rs)
   

    # get saturation vapour pressure above a flat surface of ice according to Murphy and Koop, 2005
    p_i=np.exp(9.550426 - 5723.265/T + 3.53068 * np.log(T) - 0.00728332*T) # as in thermo.py p_ice(T)
    if use_MK2005:
        # get temperature dependent latent heat of condensation as in thermo.py L_cond(T)
        L_cond=(56579 - 42.212*T + np.exp(0.1149*(281.6-T )))/c.Mw
    else:
        L_cond=c.L
    # get temperature dependent latent heat of ice sublimation as in thermo.py L_ice_sub(T)
    L_ice_sub=(46782.5 + 35.8925*T - 0.07414*(T**2) + 541.5*np.exp(-(T /123.75)**2))/c.Mw
    npart_shrinking_too_much=0
    npart_shrinking=0
    npart_growing=0
    for i in nb.prange(nr):
        r = rs[i]
        r_dry = r_drys[i]
        kappa = kappas[i]
        Ni=Nis[i]
 
        ## Non-continuum diffusivity/thermal conductivity of air near
        ## near particle
        dv_r = dv(T, r, P, accom)
        ka_r = ka(T, r, rho_air)

        if sw[i]:
            ## Provided the droplet has frozen, determine the contribution of ice to dwi_dt.
            # Condensation coefficient
            # This is deposition of ice so need L_ice_sub not L_freezing
            L=L_ice_sub
            G_a = (c.rho_i * c.R * T) / (p_i * dv_r * c.Mw)
            G_b = (L * c.rho_i * ((L * c.Mw / (c.R * T)) - 1.0)) / (ka_r * T)
            G = 1.0 / (G_a + G_b)

            ## Converting water supersaturation to ice supersaturation according to Korolev and Mazin, 2003 (not necessary to correct for particle equilibrium saturation)
            xi = pv_sat/p_i 
            Si = xi(S + 1) - 1 # xi(S+1) is the saturation wrt to ice 

            ## Size and ice tendencies
            dr_dt = (G / r) * Si
            if r>r_dry: 
                dwi_dt += (
                    4.0 * PI * c.rho_i  * Ni * r * r * dr_dt 
                )  # Contribution to ice water tendency due to growth
                drs_dt[i] = dr_dt
                dT_dt_i+=L*dwi_dt/c.Cp

        else:
                
            ## Provided the droplet has not frozen, determine the contribution of liquid droplets to dwc_dt.
            ## Condensation coefficient
            G_a = (c.rho_w * c.R * T) / (pv_sat * dv_r * c.Mw)
            G_b = (L_cond * c.rho_w * ((L_cond * c.Mw / (c.R * T)) - 1.0)) / (ka_r * T)
            G = 1.0 / (G_a + G_b)

            ## Difference between ambient and particle equilibrium supersaturation
            Seq_r = Seq(r, r_dry, T, kappa)
            delta_S = S - Seq_r

            ## Size and liquid water tendencies
            dr_dt = (G / r) * delta_S
            if r>r_dry:
                if dr_dt>0:
                    npart_growing+=1
                elif dr_dt<0:
                    npart_shrinking+=1
                dwl_dt +=  4.0 * PI * c.rho_w * (Ni * r * r * dr_dt)  # Contribution to liq. water tendency due to growth
                drs_dt[i] = dr_dt
                dT_dt_l+=L_cond*dwl_dt/c.Cp

    ## add on latent heat release from phase changes - assume this is much smaller than dT_dt due to expansion
    dT_dt= dT_dt_dry + dT_dt_l + dT_dt_i
    # this phase change also modifies pressure dP_dt=P/T * dT_dt
    dP_dt = dP_dt_dry + P*(dT_dt_l+dT_dt_i)/T
    
    ## MASS BALANCE CONSTRAINT to get dwv_dt
    dwv_dt = -1.0 * (dwl_dt + dwi_dt)
    #if npart_shrinking_too_much>0:
    #    print(npart_shrinking_too_much, 'particles want to shrink too much, S, T, P', S, T, P)

    #if npart_shrinking>0:
    #    print('ode_sys: npart growing', npart_growing, 'npart shrinking', npart_shrinking, 'S,T,P', S,T,P)

    x = np.empty_like(y)
    x[0] = dP_dt
    x[1] = dT_dt
    x[2] = dwv_dt
    x[c.N_STATE_VARS:] = drs_dt[:]

    return x
