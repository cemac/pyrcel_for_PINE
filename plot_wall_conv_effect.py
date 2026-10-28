import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from aerosol import *
import pdb

def read_traces(nc_filepath):
    
    aerosols_trace={}
    aerosols_species={}
    aero_kappas={}
    aero_r_drys={}
    aero_nis={}
    aero_cols={}
    colors=['magenta','blue', 'green']
    with xr.open_dataset(nc_filepath) as ds:
        i=0
        for k in ds.keys():
            if '_size' in k:
                aero_name=k.split('_')[0]
                aero_cols[aero_name]=colors[i]
                i+=1
                aerosols_trace[aero_name]=ds[k] * 1e-6 # back into m
            elif '_kappa' in k:
                aero_name=k.split('_')[0]
                aero_kappas[aero_name]=ds[k].values[0]
            elif '_rdry' in k:
                aero_name=k.split('_')[0]
                aero_r_drys[aero_name]=ds[k].values
            elif '_Nis' in k:
                aero_name=k.split('_')[0]
                aero_nis[aero_name]=ds[k].values

        for k in aero_r_drys.keys():
            # Nis and r_drys will be put into the correct units
            distrib={'r_drys': aero_r_drys[k], 'Nis': aero_nis[k]}
            aero=AerosolSpecies(k, 
                                distrib,
                                kappa=aero_kappas[k])
            print(k, len(aero_r_drys[k]), 'bins')
            aerosols_species[k]=aero

        
        parcel_trace={ 'time': ds['time'].values, 'S':ds['S'].values/100, 'T':ds['T'].values, 'P': ds['P'].values,
                       'wv':ds['wv'].values, 'wc':ds['wc'].values, 'wl':ds['wl'].values, 'wi':ds['wi'].values}


    return parcel_trace, aerosols_trace, aerosols_species


f1='output/pine_sulphate.nc'
parcel_trace1, aerosols_trace1, aerosols_species1=read_traces(f1)
f2='output/pine_sulphate_no_wall_conv.nc'
parcel_trace2, aerosols_trace2, aerosols_species2=read_traces(f2)

naero=len(list(aerosols_trace1.keys()))
fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharex=True)

axs[0].plot(parcel_trace1['time'], parcel_trace1['S']*100, label='wall conv')
axs[0].plot(parcel_trace2['time'], parcel_trace2['S']*100, label='no wall conv')
axs[1].plot(parcel_trace1['time'], parcel_trace1['T'])
axs[1].plot(parcel_trace2['time'], parcel_trace2['T'])
axs[2].plot(parcel_trace1['time'], parcel_trace1['P']/100)
axs[2].plot(parcel_trace2['time'], parcel_trace2['P']/100)

axs[0].set_ylabel("%")
axs[1].set_ylabel("K")
axs[2].set_ylabel("hPa")
axs[0].set_xlabel("Time (s)")
axs[1].set_xlabel("Time (s)")
axs[2].set_xlabel("Time (s)")
axs[0].grid(False, 'both', 'both')
axs[1].grid(False, 'both', 'both')
axs[2].grid(False, 'both', 'both')
axs[0].set_title("Supersaturation")
axs[1].set_title("Temperature")
axs[2].set_title("Pressure")
axs[0].legend()
plt.suptitle('impact of convection of heat from walls')
plt.savefig("output/pine_sulphate_effect_of_wall_conv.png")

plt.close()
