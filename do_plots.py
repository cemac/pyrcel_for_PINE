import xarray as xr
from vis import *
from aerosol import *
from thermo import saturation_ratio, rho_air_dry
from activation import *
import constants as c

def do_plots(outdir, exp_name):

    nc_filepath = outdir+exp_name+ ".nc"
    with xr.open_dataset(nc_filepath) as ds:
        aerosols_trace={}
        aerosols_freeze_time={}
        aerosols_species={}
        aero_kappas={}
        aero_r_drys={}
        aero_nis={}
        aero_cols={}
        colors=['magenta','blue', 'green']
        i=0
        for k in ds.keys():
            if '_size' in k:
                aero_name=k.split('_')[0]
                print('found aerosol', aero_name)
                aero_cols[aero_name]=colors[i]
                i+=1
                aerosols_trace[aero_name]=ds[k] * 1e-6 # back into m
            elif '_freeze_time' in k:
                aerosols_freeze_time[aero_name]=ds[k].values
            elif '_kappa' in k:
                aero_name=k.split('_')[0]
                print('found kappa for', aero_name)
                aero_kappas[aero_name]=ds[k].values[0]
            elif '_rdry' in k:
                aero_name=k.split('_')[0]
                print('found rdry for', aero_name)
                aero_r_drys[aero_name]=ds[k].values
            elif '_Nis' in k:
                aero_name=k.split('_')[0]
                print('found Nis for', aero_name)
                aero_nis[aero_name]=ds[k].values

        for k in aero_r_drys.keys():
            # Nis and r_drys will be put into the correct units
            distrib={'r_drys': aero_r_drys[k], 'Nis': aero_nis[k]}
            aero=AerosolSpecies(k, 
                                distrib,
                                kappa=aero_kappas[k])
            print(k, len(aero_r_drys[k]), 'bins')
            aerosols_species[k]=aero

        plot_aerosol_distributions(aerosols_species, aero_cols, outdir+exp_name+'_aero_distribution.png')

        parcel_trace={ var: ds[var].values for var in ds.keys()}
        parcel_trace['time']=ds['time'].values # time is a coordinate in ds but needs to be copied as well as variables

        pine_P=[]
        pine_T=[]
        pine_time=[]
        #pine_T=np.load('../df_PINE-1B_TERC2_opid-13_instrument_T.npy', allow_pickle=True) # in K
        #pine_T=pine_T[:,0]
        #pine_P=np.load('../df_PINE-1B_TERC2_opid-13_instrument_P.npy', allow_pickle=True) # in hPa
        #pine_time=np.load('../df_PINE-1B_TERC2_opid-13_instrument_time.npy', allow_pickle=True)# in s
        
        # put everything on one plot as we only have a few aero species
        naero=len(list(aerosols_trace.keys()))
        nrows=1+naero
        ncols=2
        fig, axs = plt.subplots(nrows, ncols, figsize=(12, 4*nrows))
        plot_STP_timeseries(parcel_trace, axs[0,0],'', pine_P=pine_P, pine_T=pine_T, pine_time=pine_time)
        plot_water_timeseries(parcel_trace, axs[0,1], '')
        r=1
        for k in aerosols_trace.keys():
            activated_timeseries=integrated_binned_activation(parcel_trace['T'], aerosols_trace[k].values,  aerosols_species[k])
            plot_activation_timeseries(parcel_trace, aerosols_trace[k], aerosols_species[k], activated_timeseries, aerosols_freeze_time[k], axs[r,0], '')
            plot_particle_state(parcel_trace, aerosols_trace[k], aerosols_species[k], activated_timeseries, aerosols_freeze_time[k], axs[r,1],'')
            r+=1
        fout=outdir+exp_name+'_results.png'
        plt.subplots_adjust(wspace=0.6, hspace=0.3)
        plt.savefig(fout)
        plt.close()

        # plot all radii growth
        nrows=naero
        ncols=1
        fig, axs = plt.subplots(nrows, ncols, figsize=(8, 4*nrows))
        if naero==1:
            axs=[axs]
        r=0
        for k in aerosols_trace.keys():
            im=plot_particle_growth(parcel_trace, aerosols_trace[k], aerosols_species[k], axs[r])
            fig.colorbar(im, ax=axs[r], orientation='horizontal', label='r_wet ($\mu$m)', shrink=0.5)
            r+=1
        fout=outdir+exp_name+'_particle_growth.png'
        plt.subplots_adjust(hspace=0.1)
        plt.savefig(fout)
        plt.close()
        
        fout=outdir+exp_name+'_nice.png'
        rho_a_d = rho_air_dry(parcel_trace['P'][0], parcel_trace['T'][0], parcel_trace['S'][0]/100)
        plot_nice_timeseries(parcel_trace, aerosols_species, aerosols_freeze_time, rho_a_d, fout)
