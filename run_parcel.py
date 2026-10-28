#!/usr/bin/env python
"""
CLI interface to run parcel model simulation.

"""
import os
import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter

import yaml

from parcel import *
from aerosol import *
from thermo import p_ice, es
from util import *
from do_plots import *

parser = ArgumentParser(
    description=__doc__, formatter_class=RawDescriptionHelpFormatter
)
parser.add_argument(
    "namelist",
    type=str,
    metavar="config.yml",
    help="YAML namelist controlling simulation configuration",
)

DIST_MAP = {
    "lognormal": Lognorm,
    "multimode_lognormal": MultiModeLognorm
}


def run_parcel():
    # Read command-line arguments
    args = parser.parse_args()

    # Convert the namelist file into an in-memory dictionary
    try:
        print("Attempting to read simulation namelist {}".format(args.namelist))
        with open(args.namelist, "rb") as f:
            y = yaml.safe_load(f)
    except IOError:
        print("Couldn't read file {}".format(args.namelist))
        sys.exit(0)

    # Create the aerosol
    aerosol_modes = []
    smallest_rdry=1e-4 # microns
    print("Constructing aerosol modes")
    for i, aerosol_params in enumerate(y["initial_aerosol"], start=1):
        ap = aerosol_params
        dist = DIST_MAP[ap["distribution"]](**ap["distribution_args"])
        # 1st create without smallest_rdy specified then truncate if necessary
        if "het_m" in ap.keys() and "het_c" in ap.keys():
            aer = AerosolSpecies(ap["name"], dist, kappa=ap["kappa"], het_m=ap["het_m"], het_c=ap["het_c"], bins=ap["bins"])
        else:
            aer = AerosolSpecies(ap["name"], dist, kappa=ap["kappa"], bins=ap["bins"])

        N_old = np.sum(aer.Nis)
        bin_count = np.count_nonzero(aer.r_drys[aer.r_drys < smallest_rdry*1e-6])
        if bin_count > 0:
            aer_new = AerosolSpecies(
                    aer.species,
                    aer.distribution,
                    kappa=aer.kappa,
                    het_m=aer.het_m,
                    het_c=aer.het_c,
                    bins=aer.bins,
                    r_min=smallest_rdry,
                )
            N_new = np.sum(aer_new.Nis)
            print(
                "%s: Removed %03d bins, N change: %5.1f -> %5.1f (%2.1f%%)"
                 % (
                    aer.species,
                    bin_count,
                    N_old,
                    N_new,
                    (N_new - N_old) / N_new,
                   )
            )
            aer=aer_new
        aerosol_modes.append(aer)

    # Set up the model
    ic = y["initial_conditions"]
    print("Initializing model")
    try:
        pi=p_ice(ic["temperature"])
        pv=es(ic["temperature"])
        RH0=ic["relative_humidity_ice"]*pi/pv
        S0=RH0-1.0
        print('initial S', S0)

        model = ParcelModel(
            aerosol_modes,
            Chamber=y["chamber_setup"],
            T0=ic["temperature"],
            S0=S0,
            P0=ic["pressure"],
            console=True
        )
    except ParcelModelError:
        print("Something went wrong setting up the model")
        sys.exit(0)

    # set up Output directory
    ec = y["experiment_control"]

    # Make output directory if it doesn't exist
    if not os.path.exists(ec["output_dir"]):
        os.makedirs(ec["output_dir"])

    outfile_base = os.path.join(ec["output_dir"], ec["name"])
       
    # Run the model
    mc = y["model_control"]
    print("Beginning simulation")
    #try:
    model.run(outfile_base, ec['output_csv'], **mc)
    #except ParcelModelError:
    #    print("Something went wrong during model run")
    #    sys.exit(0)

    # Succesful completion
    print("Done!")
    do_plots(ec["output_dir"],  ec["name"])

if __name__ == "__main__":
    run_parcel()
