# -*- coding: utf-8 -*-
""" Interface to numerical ODE solvers.
"""
from assimulo.exception import TimeLimitExceeded
from assimulo.problem import Explicit_Problem
from assimulo.solvers.sundials import CVode, CVodeError
import warnings

from math import cbrt
from thermo import L_freezing
from nucleation import  probability_hom_freezing, probability_het_freezing
import constants as c

import numpy as np


# indices into rhs_args
T0_IX=0
NR_IX=1
RDRY_IX=2
NIS_IX=3
KAPPA_IX=4
ACCOM_IX=5
CHAMBER_IX=6

def get_liquid_water_ice(r_drys, rs, Nis, is_frozen):

    # check state of water and ice
    ice=np.zeros(len(rs))
    water=np.zeros(len(rs))
    vols=4.0/3.0 * np.pi * (rs**3-r_drys**3) * Nis # per unit mass of dry air
    ixi=np.where(is_frozen)
    ixw=np.where(is_frozen==False)
    ice[ixi]=vols[ixi]*c.rho_i
    water[ixw]=vols[ixw]*c.rho_w
    return np.sum(water), np.sum(ice)
 

class ExtendedProblem(Explicit_Problem):
    """This extension of the Assimulo 'Explicit_Problem' class
    encodes detecting freezing events"""

    name = "Parcel model ODEs"

    def __init__(self, rhs_fcn, rhs_args, het_ms, het_cs, y0, freeze_time, *args, **kwargs):
        self.rhs_fcn = rhs_fcn
        self.rhs_args = rhs_args
        self.het_ms=het_ms  # heterogeneous nucleation params for aerosols
        self.het_cs=het_cs
        self.freeze_time=freeze_time
        sw0=np.zeros(freeze_time.shape, np.bool)

        super(Explicit_Problem, self).__init__(y0=y0, sw0=sw0) #*args, **kwargs)

    def rhs(self, t, y, sw):
        dode_dt = self.rhs_fcn(t, y, np.asarray(sw), *self.rhs_args)

        return dode_dt
            
    # The event function
    def state_events(self, t, y, sw):
        """Check whether an 'event' has occurred. We want to see if any particles are frozen or not."""

        sw_array=np.asarray(sw)
        T = y[1]
 
        # find freezing events
        rs = y[c.N_STATE_VARS:] # the radii
        r_drys=self.rhs_args[RDRY_IX]
        ix=np.where(rs<r_drys)
        if len(ix[0])>0:
            print('some rs less than r_dry!!!', ix)
            pdb.set_trace()
        kappas=self.rhs_args[KAPPA_IX]
        dt=0.1 # 10th of a second
        probs=np.zeros_like(rs)
        # find homogeneous freezing rate for not yet frozen particle bins
        ix=np.where((sw_array==False) & (self.het_ms==0))
        if len(ix[0])>0:
            probs[ix]=probability_hom_freezing(T, rs[ix], r_drys[ix], kappas[ix], dt)
        # find heterogeneous freezing rate for not yet frozen particle bins
        ix=np.where((sw_array==False) & (self.het_ms>0))
        if len(ix[0])>0:
            probs[ix]=probability_het_freezing(T, rs[ix], r_drys[ix], kappas[ix], self.het_m[ix], self.het_c[ic], dt)
        
        pthresh=0.99 # 99% chance of freezing in next 10th of a second
        events=pthresh-probs # if below threshold events will be positive. when it cross the threshold it will become negative

        return events

    # Event handling function
    def handle_event(self, solver, event_info):
        """Event handling. This function is called when Assimulo finds
        an event as specified by the event function."""
        events=np.asarray(event_info[0])
        # assume particles instantly freeze, water turns to ice and releases latent heat
        state=solver.y
        T = state[1]
        nr=self.rhs_args[NR_IX]
        r_drys=self.rhs_args[RDRY_IX]
        rs = state[c.N_STATE_VARS:] # the radii
        Nis=self.rhs_args[NIS_IX] # number per unit mass of dry air
        L_freeze=L_freezing(T) # pure water freezing
        dwi=0.0
        dT=0.0
        ix=np.where(events[:nr]==-1)
        for i in ix[0]:
            assert(solver.sw[i]==False)
            volume_water=4.0/3.0 * np.pi  * (rs[i]**3-r_drys[i]**3) * Nis[i] # this is per unit mass of dry air
            this_dwi =volume_water*c.rho_w # this mass of water is now ice
            dwi+=this_dwi
            # we've just switched from water to ice
            # ice has lower density so r will increase just by freezing but mass is same
            # rho_i(r_i**3 - r_dry**3)=rho_w*(r_w**3 - r_dry**3)
            # r_i**3=(rho_w/rho_i)*(r_w**3 - r_dry**3) + r_dry**3
            ri_cubed=(c.rho_w/c.rho_i)*(rs[i]**3 - r_drys[i]**3) +  r_drys[i]**3
            r_i=cbrt(ri_cubed)
            state[c.N_STATE_VARS:][i]=r_i
            self.freeze_time[i]=solver.t # set time when froze
            solver.sw[i]=True

        dT=L_freeze * dwi / c.Cp
        dP=state[0]*dT/state[1]

        print('handle_event:', ix[0], 'frozen at', solver.t, T, 'K, dwi', dwi)
        state[0]+=dP
        state[1]+=dT
        solver.y=state


