import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter

import yaml

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

# Read command-line arguments - yaml file defining experiment
args = parser.parse_args()

try:
    print("Attempting to read simulation namelist {}".format(args.namelist))
    with open(args.namelist, "rb") as f:
        y = yaml.safe_load(f)
except IOError:
    print("Couldn't read file {}".format(args.namelist))
    sys.exit(0)
    
ec = y["experiment_control"]
do_plots(ec["output_dir"],  ec["name"])
