# pyrcel_for_PINE
Model for the PINE chamber based partly on pyrcel which models a parcel of air being lifted by an updraught and handles condensation of water on the particles according to K-Kohler theory.

This is a model for an adiabatic expansion chamber with a volumetric flow rate V_rate which determines how fast the pressure and temperature decrease. It also models heat transfer from the walls. The initial temperature, pressure and supersaturation and the aerosols present are defined using a yml file. See pine_sulphate.yml and pine_oil_overboard.yml for examples.

The environment.yml file should be used to create a conda environment in which to run the model. To run the model use

python run_parcel.py <yml_file>

Firstly a ParcelModel is created with the aerosols and initial T, P and S. From this, the initial water vapor content, wv, and the initial wet radii, r_wets, of all the particles can be calculated. The ParcelModel creates an instance of the Assimulo CVODE solver and an instance of the ExtendedProblem (see assimulo_ode_interface.py). The ExtendedProblem class is used to handle freezing events. Currently the code only handles homogeneous freezing although there is some code for handling heterogeneous freezing.

The ParcelModel is then told to run which calls simulate() on the ODE solver. At each timestep the ODE solver calls the parcel_ode_rhs function (see _parcel_aux_numba.py) to calculate tendancies in P, T, wv and r_wets. The particle growth due to condensation or ice deposition assumes the particles are spherical. The model will be developed further to handle enhanced growth due to pores and irregular shaped particles.

Once the simulation is complete, the timeseries of P, T, S, wv, wl, wi and r_wets are output to a netcdf file, along with the time that each particle froze.

## Future enhancements:
To handle heterogeneous nucleation we will use the Knopf and Alpert 2013 model. For this we will need h and m for each aerosol type and the surface area of the particles. If the particles are spherical we can calculate the surface area, but for irregular shapes we will need some kind of shape factor. We may also need a shape factor or something similar for handling condensation on irregular shaped particles.
