""" Main implementation of parcel model.
"""
import os
import numpy as np
from scipy.optimize import bisect
import time
timer= time.process_time

import xarray as xr
import pandas as pd

# Parcel model imports
import version as ver
import constants as c
from assimulo_ode_interface import *
import pdb

from aerosol import AerosolSpecies
from thermo import Seq, es, kohler_crit, rho_air, rho_air_dry, saturation_ratio, wv_from_saturation_ratio
from util import ParcelModelError

#__all__ = ["ParcelModel"]


class ParcelModel(object):
    """ Wrapper class for instantiating and running the parcel model.

    The parcel model has been implemented in an object-oriented format to facilitate
    easy extensibility to different aerosol and meteorological conditions. A
    typical use case would involve specifying the initial conditions such as:

    >>> P0 = 100000.
    >>> T0 = 243.15
    >>> S0 = 0.0
    >>> Chamber = dictionary of required constants describing chamber set up (flow rate, volume, area)
    >>> aerosol1 = pm.AerosolSpecies('sulfate',
    ...                              Lognorm(mu=0.025, sigma=1.3, N=2000.),
    ...                              bins=200, kappa=0.54)
    >>> initial_aerosols = [aerosol1, ]

    which initializes the model with initial T, P and S for the chamber and a simple
    sulfate aerosol distribution which will be discretized into 200 size bins to
    track.

    Running the model and saving the output can be accomplished by invoking::

    >>> model = ParcelModel(initial_aerosols, Chamber, T0, S0, P0)
    >>> model.run(t_end, dt)


    Attributes
    ----------
    Chamber, T0, S0, P0, aerosols : 
        Initial parcel settings (see **Parameters**).
    _r0s : array_like of floats
        Initial equilibrium droplet sizes.
    _r_drys : array_like of floats
        Dry radii of aerosol population.
    _kappas : array_like of floats
        Hygroscopicity of each aerosol size.
    _het_m: array_like of floats
    _het_c: array_like of floats
        parameters for heterogeneous nucleation
    _Nis : array_like of floats
        Number concentration of each aerosol size.
    _freeze_time: array of floats
        Time at which each particle bin froze (defaults to -1)
    _nr : int
        Number of aerosol sizes tracked in model.
    _model_set : boolean
        Flag indicating whether or not at any given time the model
        initialization/equilibration routine has been run with the current
        model settings.

    Methods
    -------
    run(t_end, dt, solver_args={})
        Execute model simulation.
    set_initial_conditions(T0=None, S0=None, P0=None, aerosols=None)
        Re-initialize a model simulation in order to run it.

    See Also
    --------
    _setup_run : companion routine which computes equilibrium droplet sizes
                 and sets the model's state vectors.
    _integrate : call the Assimilo simulate code for each set of times

    """

    def __init__(
        self,
        aerosols,
        Chamber,
        T0,
        S0,
        P0,
        console=False,
        accom=c.ac
    ):
        """Initialize the parcel model.

        Parameters
        ----------
        aerosols : array_like sequence of :class:`AerosolSpecies`
            The aerosols contained in the parcel.
        Chamber : dictionary
            The chamber constants (flow rate (m3/s), volume (m3), area (m2)) diameter (m),
        T0, S0, P0 : floats
            initial temperature (K), supersaturation (percent, with 0.0 = 100% RH), pressure (Pa).
        console : boolean, optional
            Enable some basic debugging output to print to the terminal.
        accom : float, optional (default=:const:`constants.ac`)
            Condensation coefficient

        """
        self._model_set = False
        self.console = console
        self.Chamber = Chamber
        self.T0 = T0
        self.S0 = S0
        self.P0 = P0
        self.aerosols = aerosols
        self.accom = accom

        self._setup_run()

    def set_initial_conditions(self, Chamber=None, T0=None, S0=None, P0=None, aerosols=None):
        """Set the initial conditions and parameters for a new parcel
        model run without having to create a new :class:`ParcelModel` instance.

        Based on the aerosol population which has been stored in the model, this
        method will finish initializing the model. This has three major parts:

        1. concatenate the aerosol population information (their dry radii,
           hygroscopicities, etc) into single arrays which can be placed into the
           rhs_args for use in the forward integration.
        2. Given the initial ambient water vapor concentration (computed from the
           temperature, pressure, and supersaturation), determine how much water
           must already be coated on the aerosol particles in order for their
           size to be in equilibrium.
        3. Set-up the state vector with these initial conditions.

        Once the state vector has been set up, the setup routine will record
        attributes in the parent instance of the :class:`ParcelModel`.

        Parameters
        ----------
        Chamber: dictionary describing chamber setup
        T0, S0, P0 : floats
            T0=initial temperature (K), P0=pressure (Pa), S0=supersaturation (percent, with 0.0 = 100% RH).
        aerosols : array_like sequence of :class:`AerosolSpecies`
            The aerosols contained in the parcel.

        Raises
        ------
        ParcelModelError
            If an equilibrium droplet size distribution could not be calculated.

        Notes
        -----
        The actual setup occurs in the private method `_setup_run()`; this
        method is simply an interface that can be used to modify an existing
        :class:`ParcelModel`.

        """

        if Chamber:
            self.Chamber = Chamber
        if T0:
            self.T0 = T0
        if P0:
            self.P0 = P0
        if S0:
            self.S0 = S0
        if aerosols:
            self.aerosols = aerosols

        if T0 or P0 or S0 or aerosols:
            self._setup_run()


    def _setup_run(self):
        """Perform equilibration and initialization calculations for the
        parcel model simulation.

        .. note:: See `set_initial_conditions` for full details.

        """

        if self.console:
            print("Setting up parcel model initial conditions...")

        # 1) Setup aerosols
        # a) grab all the initial aerosol size/concentrations
        # It is number of aerosol particles per unit mass of dry air that will stay the same
        # as air is being sucked out but the volume stays the same so use a Nis per unit mass of dry air.
        rho_a_d = rho_air_dry(self.P0, self.T0, self.S0)
        self.species = []
        self.r_drys, self.Nis, self.kappas, self.het_ms, self.het_cs = [], [], [], [], []
        for aerosol in self.aerosols:
            self.r_drys.extend(aerosol.r_drys)
            self.kappas.extend([aerosol.kappa] * aerosol.nr)
            self.het_ms.extend([aerosol.het_m] * aerosol.nr)
            self.het_cs.extend([aerosol.het_c] * aerosol.nr)
            self.Nis.extend(aerosol.Nis/rho_a_d)
            self.species.extend([aerosol.species] * aerosol.nr)

        self.r_drys = np.array(self.r_drys)
        self.kappas = np.array(self.kappas)
        self.het_ms = np.array(self.het_ms)
        self.het_cs = np.array(self.het_cs)
        self.Nis = np.array(self.Nis)
        self.nr=len(self.r_drys)
        self.freeze_time=np.zeros(self.nr)-1


        if self.console:
            # TODO: Fix print table formatting
            print("AEROSOL DISTRIBUTION")
            print("%8s %6s" % ("r", "N"))
            for sp, r, N in zip(self.species, self.r_drys, self.Nis):
                print("%10s %2.2e %4.1f" % (sp, r, N))
            print("\n" + "-" * 44)

        # 2) Setup parcel initial conditions
        # a) water vapor
        #        RH    * (Rd/Rv = epsilon) * ( es / P - es )
        wv0 = wv_from_saturation_ratio(self.P0,self.T0,self.S0 + 1.0) # Water Vapor mixing ratio, kg/kg

        # b) find equilibrium wet particle radius
        # wrapper function for quickly computing deviation from chosen
        # equilibrium supersaturation given current size and dry size
        f = lambda r, r_dry, kappa: Seq(r, r_dry, self.T0, kappa) - self.S0
        # Compute the equilibrium wet particle radii
        r0s = np.zeros(self.nr)
        raised = False
        for i in range(self.nr):
            # Locate the critical value (f(r_crit) > 0), and use bisection from it and
            # r_dry (f(r_dry) < 0) to find equilibrium wet particle radius for a given S0
            r_dry=self.r_drys[i]
            kappa=self.kappas[i]
            r_b, _ = kohler_crit(self.T0, r_dry, kappa)
            r0 = bisect(f, r_dry, r_b, args=(r_dry, kappa), xtol=1e-30, maxiter=500)
            if r0<r_dry:
                if self.console:
                    print("Found bad r0", r0, r_dry, self.species[i])
                raised = True

            r0s[i]=r0
            ss = Seq(r0, r_dry, self.T0, kappa)
            if np.abs(ss - self.S0) > 1e-4:
                if self.console:
                    print("Found S discrepancy", ss, self.S0, r_dry)
                raised = True
        if raised:
            raise ParcelModelError(
                "Couldn't calculate initial aerosol population wet sizes."
            )

        # c) compute equilibrium droplet liquid water and ice content
        # get mass per unit mass of the air
        print('r0s', r0s)
        print('Nis', self.Nis)
        wl0, wi0 = get_liquid_water_ice(self.r_drys, r0s, self.Nis, self.freeze_time>=0)
        print('initial wl, wi', wl0, wi0)
        self.water_tot=wv0+wl0+wi0

        # e) concatenate into initial conditions arrays and rhs args
        self.y0 = np.zeros(c.N_STATE_VARS+self.nr)
        self.y0[:c.N_STATE_VARS]=[self.P0, self.T0, wv0] # we only need to track wv not wl and wi
        self.y0[c.N_STATE_VARS:]=r0s
        self.rhs_args = [self.T0, self.nr, self.r_drys, self.Nis, self.kappas, self.accom, np.asarray([self.Chamber['chamber_vol'], self.Chamber['chamber_area'], self.Chamber['chamber_diameter'], self.Chamber['vol_flowrate']])]
        if self.console:
            print("PARCEL INITIAL CONDITIONS")
            print(
                ("    " + "{:>9} " * 6).format(
                    "P (hPa)",
                    "T (K)",
                    "wv (g/kg)",
                    "wl (g/kg)",
                    "wi (g/kg)",
                    "S",
                )
            )
            print(
                "    "
                + "{:9.1f} {:9.2f} {:9.1e} {:9.1e} {:9.1e} {:9.3f}".format(
                    self.P0 / 100.0, self.T0, wv0 * 1e3, wl0 * 1e3, wi0 * 1e3, self.S0
                )
            )
 
        self._model_set = True

        if self.console:
            print("Initial conditions set successfully.")
        
    def run(
        self,
        outfile_base,
        save_as_csv, 
        t_end=40.0,
        output_dt=1.0,
        solver_dt=None,
        **solver_args
    ):
        """Run the parcel model simulation and save results in outfile.

        Once the model has been instantiated, a simulation can immediately be
        performed by invoking this method. The numerical details underlying the
        simulation and the times over which to integrate can be flexibly set
        here.

        **Time** -- The user must specify two timesteps: `output_dt`, which is the
        timestep between output snapshots of the state of the parcel model, and
        `solver_dt`, which is the the interval of time before the ODE integrator
        is paused and re-started. It's usually okay to use a very large `solver_dt`,
        as `output_dt` can be interpolated from the simulation. In some cases though
        a small `solver_dt` could be useful to force the solver to use smaller
        internal timesteps.

        outfile_base is the base filename for the output. We will always write as netcdf
        to outfile_base.nc
        if save_as_csv is True we will also output in csv format in outfile_base.csv



        Parameters
        ----------
        t_end : float
            Total time over interval over which the model should be integrated
        output_dt : float
            Timestep intervals to report model output.
        solver_dt : float
            Timestep interval for calling solver integration routine.
        solver_args : 
            eg type of iteration, Maximum number of steps allowed by solver to satisfy error tolerances
            per timestep, etc.

        Raises
        ------
        ParcelModelError
            The parcel model failed to complete successfully or failed to initialize.

 
        """

        if solver_dt is None:
            solver_dt = 10.0 * output_dt

        if not self._model_set:
            self._setup_run()

        try:
            # Cython
            # from .parcel_aux import der as rhs_fcn
            # Numba - JIT
            from _parcel_aux_numba import parcel_ode_rhs as rhs_fcn

            # Numba - AOT
            # from .parcel_aux_numba import parcel_ode_rhs as rhs_fcn
        except ImportError:
            print("Could not load Cython derivative")

        if self.console:
            print()
            print("Integration control")
            print("----------------------------")
            print("        output dt: ", output_dt)
            print("    max solver dt: ", solver_dt)
            print(" solver int steps: ", int(solver_dt / output_dt))
 
        # Setup solver
        self.prob = ExtendedProblem(rhs_fcn, self.rhs_args, self.het_ms, self.het_cs, self.y0, self.freeze_time)

        # Create Assimulo interface
        self.sim = CVode(self.prob)
        self.sim.discr = "BDF"
        self.sim.maxord = 5

        # Setup some default arguments for the ODE solver, or override
        # if available. This is very hackish, but it's fine for now while
        # the number of anticipated tuning knobs is small.
 

        if "iter" in solver_args:
            self.sim.iter = solver_args["iter"]
        else:
            self.sim.iter = "Newton"

        if "linear_solver" in solver_args:
            self.sim.linear_solver = solver_args["linear_solver"]

        if "maxsteps" in solver_args:
            self.sim.maxsteps = solver_args["maxsteps"]
        else:
            self.sim.maxsteps = 1000

        if "time_limit" in solver_args:
            self.sim.time_limit = solver_args["time_limit"]
            self.sim.report_continuously = True
        else:
            self.sim.time_limit = 0.0

        # Don't save the [t_-, t_+] around events
        self.sim.store_event_points = False

        # Setup tolerances
        state_atol = [1e-4, 1e-4, 1e-10]
        state_rtol = 1e-7
        self.sim.rtol = state_rtol
        self.sim.atol = state_atol + [1e-12] * self.nr

        if not self.console:
            self.sim.verbosity = 50
        else:
            self.sim.verbosity = 40
            
        try:
            if self.console:
                print("\nBEGIN INTEGRATION ->\n")
            x, times = self._integrate(solver_dt, output_dt, t_end)
        except ValueError as e:
            raise ParcelModelError("Something failed during model integration: %r" % e)


        # Success if reached this point!
        if self.console:
            print("\nEND INTEGRATION <-\n")


        self._save_output(x, times, outfile_base, save_as_csv)

    
    def _integrate(self, solver_dt, output_dt, t_end):
        # Compute integration logic. We need to know:
        # 1) How are we iterating the solver loop?
        t_increment = solver_dt
        # 2) How many points do we want to interpolate for output?
        n_out = int(solver_dt / output_dt)
        t_current = 0.0

        if self.console:
            print()
            print("Integration Loop")
            print()
            print("  step     time  walltime  Δwalltime |     P       T       S       wv       wl       wi")
            print(" " "------------------------------------|----------------------")
            step_fmt = (
                " {:5d} {:7.2f}s  {:7.2f}s  {:8.2f}s |" " {:5.1f} {:7.2f} {:6.2f}% {:6.2e} {:6.2e} {:6.2e}"
               )

        txs, xxs = [], []
        n_steps = 1
        total_walltime = 0.0
        now = timer()
        while t_current < t_end:
            if self.console:
                # Update timing estimates
                delta_walltime = timer() - now
                total_walltime += delta_walltime

                # Grab state vars
                state = self.y0 if n_steps == 1 else xxs[-1][-1]
                _P = state[c.STATE_VAR_MAP["P"]]
                _T = state[c.STATE_VAR_MAP["T"]]
                _wv = state[c.STATE_VAR_MAP["wv"]]
                _S = (saturation_ratio(_P, _T, _wv)-1.0)*100
                _rwets=state[c.N_STATE_VARS:]
                _wl,_wi=get_liquid_water_ice(self.r_drys, _rwets, self.Nis, self.freeze_time>0)
                print(
                    step_fmt.format(
                        n_steps,
                        t_current,
                        total_walltime,
                        delta_walltime,
                        _P,
                        _T,
                        _S,
                        _wv,
                        _wl,
                        _wi
                    ))
                is_frozen=self.freeze_time>0
                print('frozen',is_frozen)
            try:
                now = timer()
                out_list = np.linspace(t_current, t_current + t_increment, n_out + 1)
                tx, xx = self.sim.simulate(t_current + t_increment, ncp_list=out_list)
 
            except CVodeError as e:
                 raise ValueError("Something broke in CVode: %r" % e)
            except TimeLimitExceeded:
                 raise ValueError("CVode took too long to complete")

            if n_out == 1:
                txs.append(tx[-1])
                xxs.append(xx[-1])
            else:
                txs.extend(tx[:-1])
                xxs.append(xx[:-1])
            t_current = tx[-1]

            n_steps += 1
        if self.console:
            print("---- end of integration loop ----")

        # Determine output information
        times = np.array(txs)
        if n_out == 1:  # can just merge the outputs
            x = np.array(xxs)
        else:  # Need to concatenate lists of outputs
            x = np.concatenate(xxs)

        return x, times

    def _save_output(self, x, times, filename, save_as_csv):
        nt=len(times)
        P=x[:,c.STATE_VAR_MAP["P"]]
        T=x[:,c.STATE_VAR_MAP["T"]]
        wv=x[:,c.STATE_VAR_MAP["wv"]]
        S=np.asarray([saturation_ratio(P[t], T[t], wv[t])-1.0 for t in range(nt)])
        # calculate split between ice and liquid based on radii of particles and their frozen state at each time
        ws = np.asarray([get_liquid_water_ice(self.r_drys, x[t,c.N_STATE_VARS:], self.Nis, (times[t]>=self.freeze_time)&(self.freeze_time>=0)) for t in range(len(times))])
        wl=ws[:,0]
        wi=ws[:,1]
        
        self._save_to_netcdf(x, times, S, wl, wi, filename+'.nc')
        if save_as_csv:
           self._save_to_csv(x, times, S, wl, wi, filename+'.csv')
           
    def _save_to_netcdf(self, x, times, S, wl, wi, filename):
        """Write model output to disk as netcdf file.

        Parameters
        ----------
        x - the state variables that were tracked for each time
        times - corresponding times of tracking
        S, wl, wi - calculated supersaturation, liquid water content and ice content
        filename : str
            Full filename to write output


        """

        if self.console:
            print()
            print(
                "Saving output to", filename
                 )
            print()

        P=x[:,c.STATE_VAR_MAP["P"]]
        T=x[:,c.STATE_VAR_MAP["T"]]
        wv=x[:,c.STATE_VAR_MAP["wv"]]

        ## Construct xarray datastructure to write to netCDF
        ds = xr.Dataset(attrs={"Conventions": "CF-1.0", "source": "pyrcel v%s" % ver.__version__})

        ds.coords["time"] = (
            "time",
            times,
            {"units": "seconds", "long_name": "simulation time"},
        )
        nt=len(times)
        ## Aerosol coordinates and basic data
        species_shift = 0  # increment by nr to select the next aerosol's radii
        for aerosol in self.aerosols:
            if self.console:
                print(aerosol)

            nr = aerosol.nr
            r_drys = aerosol.r_drys * 1e6
            kappas = [aerosol.kappa] * nr
            Nis = aerosol.Nis * 1e-6
            species = aerosol.species
            radii =  x[:, c.N_STATE_VARS + species_shift: c.N_STATE_VARS + species_shift+nr]
            freeze_time=self.freeze_time[species_shift: species_shift+nr]
            species_shift+=nr
            aer_coord = "%s_bins" % species
            ds.coords[aer_coord] = (
                aer_coord,
                np.array(list(range(1, aerosol.nr + 1)), dtype=np.int32),
                {"long_name": "%s size bin number" % species},
            )
            ds["%s_rdry" % species] = (
                (aer_coord,),
                r_drys,
                {"units": "micron", "long_name": "%s bin dry radii" % species},
            )
            ds["%s_kappas" % species] = (
                (aer_coord,),
                kappas,
                {"long_name": "%s bin kappa-kohler hygroscopicity" % species},
            )
            ds["%s_Nis" % species] = (
                (aer_coord,),
                Nis,
                {
                    "units": "cm-3",
                    "long_name": "%s bin number concentration" % species,
                },
            )
            ds["%s_freeze_time" % species] = (
                (aer_coord,),
                freeze_time,
                {
                    "units": "s",
                },
            )
            ds["%s_size" % species] = (
                ("time", aer_coord),
                radii * 1e6,
                {"units": "micron", "long_name": "%s bin wet radii" % species},
            )


        ## state variables
        ds["S"] = (
            ("time",),
            S * 100.0,
            {"units": "%", "long_name": "Supersaturation"},
        )
        ds["T"] = (
            ("time",),
            T,
            {"units": "K", "long_name": "Temperature"},
        )
        ds["P"] = (
            ("time",),
            P,
            {"units": "Pa", "long_name": "Pressure"},
        )
        ds["wv"] = (
            ("time",),
            wv,
            {"units": "kg/kg", "long_name": "Water vapor mixing ratio"},
        )
        ds["wl"] = (
            ("time",),
            wl,
            {"units": "kg/kg", "long_name": "Liquid water mixing ratio"},
        )
        ds["wi"] = (
            ("time",),
            wi,
            {"units": "kg/kg", "long_name": "Ice mixing ratio"},
        )

        ## Save to disk
        ds.to_netcdf(filename)

    def _save_to_csv(self, x, times, S, wl, wi, filename):
        """Write model output to disk as csv file to filename .

        Parameters
        ----------
        x - the state variables that were tracked for each time
        times - corresponding times of tracking
        filename : str
            Full filename to write output


        """

        if self.console:
            print()
            print(
                "Saving output to", filename
                 )
            print()
        parcel_dict={var: x[:, i] for i, var in enumerate(c.STATE_VARS)}
        parcel_df = pd.DataFrame(parcel_dict, index=times)

        ## Add S, wl, wi, radii and frozen state to the parcel model dataframe
        parcel_df["S"]=S*100 # save in %
        parcel_df["wl"]=wl
        parcel_df["wi"]=wi
    
        species_shift = 0  # increment by nr to select the next aerosol's radii
        radii_dict={}
        frozen_dict={}
        for aerosol in self.aerosols:
            nr = aerosol.nr
            species = aerosol.species
            for i in range(nr):
                label=species+"_r%03d" % i
                radii_dict[label] = x[:, c.N_STATE_VARS + species_shift + i]
                frozen=np.zeros(len(times), bool)
                ix=np.where(times>=self.freeze_time[species_shift+i])
                frozen[ix]=True
                frozen_dict[label+'_frozen']=frozen
            species_shift += nr
        radii_df=pd.DataFrame(radii_dict, index=times)
        frozen_df=pd.DataFrame(frozen_dict, index=times)
        parcel_df=pd.concat([parcel_df, radii_df, frozen_df], axis=0)
        parcel_df.to_csv(filename)
