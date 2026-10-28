import numpy as np
import constants as c
import pdb


"""
 Calculates the homogeneous freezing rate for a given temperature, wet and dry droplet radius, and kappa value. 
 This approach parameterizes the water activity according to kappa-Köhler theory (Petters and
 Kreidenweis, 2007) and uses this in combination with the parameterization for homogeneous ice nucleation
 proposed by T Koop 2000. 
 I have also introduced a modified water activity using the results in C Marcolli, 2020, that result in
 a reduced freezing temperature for nanodroplets. The overall approach has previously been used by
 D Lewellen, 2020.

        Parameters
        ----------
        T : float
            Plume temperature, in Kelvin
        r_wet : float
            Wet particle radii, in m
        kappa : float
            Particle hygroscopicity, no units
        r_dry : float
            Dry particle radii, in m

        Returns
        -------
        x : J
            Homogeneous freezing rate, in units of: ice nucleation events per unit volume per unit time

"""   
def J_hom_Koop(T, r_wet, r_dry, kappa):
     ## Parameterize the surface tension
    T_c = 647.096
    tau = 1 - T/T_c
    mu = 1.256
    B_ = 0.2358
    b_ = -0.625

    gamma_vw = B_ * tau ** mu * (1 + b_ * tau)

    ## Specify the input conditions
    P0 = 1e5/1e6 # 101325 ~ 1e5 normalised by 1e6 (for MPa)
    P_droplet = (P0 + (2 * gamma_vw / r_wet)/1e6) / 1e3 # converting to GPa
        
    a_w = (r_wet ** 3 - r_dry ** 3)/(r_wet **3 - r_dry ** 3 * (1-kappa))

    u_w_i_minus_u_w_0 = 210368 + 131.438 * T - 3.32373e6/T - 41729.1 * np.log(T)
    a_w_i = np.exp(u_w_i_minus_u_w_0/(8.3145 * T))

    v_w_0 = -230.76 - 0.1478 * T + 4099.2/T + 48.8341 * np.log(T)
    v_i_0 = 19.43 - 2.2e-3 * T + 1.08e-5 * T ** 2

    v_w_minus_v_i = v_w_0 * (P_droplet - 0.5 * 1.6 * P_droplet ** 2 - 1/6 * -8.8 * P_droplet ** 3) - v_i_0 * (P_droplet - 0.5 * 0.22 * P_droplet ** 2 - 1/6 * -0.17 * P_droplet **3)

    delta_a_w = a_w * np.exp(v_w_minus_v_i/(8.3145 * T)) - a_w_i
    # if delta_a_w gets > 0.4 J gets very big and P cannot be determined so cap at 0.4
    ix=np.where(delta_a_w>0.4)
    if len(ix[0])>0:
        delta_a_w[ix]=0.4

    J = 10 ** (-906.7 + 8502 * delta_a_w - 26924 * delta_a_w ** 2 + 29180 * delta_a_w ** 3)
 
    ix=np.where(np.isfinite(J)==False)
    if len(ix[0])>0:
         min_delta_a_w=np.amin(delta_a_w)
         max_delta_a_w=np.amax(delta_a_w)
         print('J is bad min max delta_a_w',min_delta_a_w, max_delta_a_w)
         pdb.set_trace()
             
    return J

"""
Determine probability of homogeneous freezing within timescale dt

        Parameters
        ----------
        T : float
            Plume temperature, in Kelvin
        r_wet : float
            Wet particle radii, in m
        kappa : float
            Particle hygroscopicity, no units
        r_dry : float
            Dry particle radii, in m

        Returns
        -------
        P: float
           probability
"""
def probability_hom_freezing(T, r_wet, r_dry, kappa, dt):
    J=J_hom_Koop(T, r_wet, r_dry, kappa)
    ## Determine the frozen fraction via the liquid water volume
    volume_of_dry_particle = 4/3 * np.pi * r_dry ** 3 # volume of dry particle without the soluble material included
    volume_of_wet_particle = 4/3 * np.pi * r_wet ** 3 # volume of wet particle including water added during hygroscopic growth
    LWV = volume_of_wet_particle - volume_of_dry_particle # total volume of metastable liquid (original coating + water)
    ix=np.where(LWV<0)
    if len(ix[0])>0:
        print('LWV is negative for some particles')
        pdb.set_trace()
    P = 1 - np.exp(-LWV * J*1e6 * dt)
    ix=np.where(np.isfinite(P)==False)
    if len(ix[0])>0:
        print('Prob not finite', P)
        pdb.set_trace()
    return P

"""
Calculates the heterogeneous nucelation rate coefficient as define in Knopf and Alpert 2013
using the water activity aw
"""
def J_het_KnopfAlpert(T, r_wet, r_dry, kappa, m, c):
    
    a_w = (r_wet ** 3 - r_dry ** 3)/(r_wet **3 - r_dry ** 3 * (1-kappa))
    u_w_i_minus_u_w_0 = 210368 + 131.438 * T - 3.32373e6/T - 41729.1 * np.log(T)
    a_w_i = np.exp(u_w_i_minus_u_w_0/(8.3145 * T))
    delta_a_w = a_w - a_w_i
    J=m*delta_a_w+c

    return J

"""
Determine probability of heterogeneous freezing within timescale dt

        Parameters
        ----------
        T : float
            Plume temperature, in Kelvin
        r_wet : float
            Wet particle radii, in m
        kappa : float
            Particle hygroscopicity, no units
        m,c : floats
            The constants used for heterogeneous nucleation for this type of particle
        r_dry : float
            Dry particle radii, in m

        Returns
        -------
        P: float
           probability
"""
def probability_het_freezing(T, r_wet, r_dry, kappa, m, c, dt):
    
    J=J_het_KnopAlpert(T, r_wet, r_dry, kappa)
    A=4*np.pi * r_wet **3
    P = 1 - np.exp(-A * J*1e4 * dt)

    return P
