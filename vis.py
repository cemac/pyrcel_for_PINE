import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import constants as c
import pdb

def plot_distribution(aer, aer_kwargs={}, ax=None, **kwargs):
    """Generate a comparison plot of a given aerosol or
    droplet distribution

    Parameters
    ----------
    aer : `AerosolSpecies`
        The container class for the aerosol
    aer_kwargs : dict
        A dictionary of arguments to pass to the matplotlib
        function which plots the binned distribution
    ax : Axis
        The axes object to plot on

    """

    if ax is None:
        raise ValueError("Must provide axes instance for plotting.")

    ## Add some basic aer_kwargs if not provided
    if not "color" in aer_kwargs:
        aer_kwargs["color"] = "g"
    if not "alpha" in aer_kwargs:
        aer_kwargs["alpha"] = 0.5

    r_left = aer.rs[:-1]
    bin_height = aer.Nis / 1e6
    r_width=np.diff(aer.rs)
    print(aer.species, 'max bin height', np.amax(bin_height), 'min max width', np.amin(r_width), np.amax(r_width))
    bars = ax.bar(r_left, bin_height, width=r_width, **aer_kwargs)

    return bars
    
def plot_aerosol_distributions(aero_species, aero_cols, fout):
    legend_objects=[]
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    for k in aero_species.keys():
        aer_kwargs={'color':aero_cols[k]}
        aer=aero_species[k]
        bars=plot_distribution(aer, aer_kwargs=aer_kwargs, ax=ax)
        legend_objects.append((bars[0], "%s bins" % aer.species))

    handles, labels = list(zip(*legend_objects))
    ax.legend(handles, labels, loc="upper right")

    ax.semilogx()
    ax.set_xlabel("$r$ ($\mu$m)")
    ax.set_ylabel("Number Concentration (cm$^{-3}$)")
    plt.savefig(fout)
    plt.close()

def plot_mass_flow_rate(parcel_trace, Vrate, fout):
    # plot the dn_dt
    dn_dt=Vrate*parcel_trace['P']/(c.R*parcel_trace['T'])
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(parcel_trace['time'], dn_dt*c.Ma*1e3)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('rate of mass of dry air extracted(g/s)')
    ax.set_title('extraction of air')
    plt.savefig(fout)
    plt.close()
    
# plots S, T and P vs time
def plot_STP_timeseries(parcel_trace, axS, fout, pine_P=[], pine_T=[], pine_time=[]):
    ind_final=len(parcel_trace['S'])-1
    if axS==None:
        fig, axS = plt.subplots(1, 1, figsize=(6, 4))

    axS.plot(parcel_trace['time'], parcel_trace['S'], color='blue', lw=2)
    axT = axS.twinx()
    axT.plot(parcel_trace['time'], parcel_trace['T'], color='r', lw=1.5)
    if len(pine_T)>0:
        axT.plot(pine_time, pine_T, color='magenta', lw=1.5)
    axP = axS.twinx()
    axP.spines['right'].set_position(('outward', 60))
    axP.plot(parcel_trace['time'], parcel_trace['P']/100, color='k', lw=1.5)
    if len(pine_P)>0:
        axP.plot(pine_time, pine_P, color='grey', lw=1.5)

    axS.yaxis.label.set_color('blue')
    axS.tick_params(axis='y', colors='blue')

    axT.yaxis.label.set_color('red')
    axT.tick_params(axis='y', colors='red')
    
    axP.yaxis.label.set_color('black')
    axP.tick_params(axis='y', colors='black')

    axS.set_ylabel("Supersaturation (%)")
    axT.set_ylabel("Temperature (K)")
    axP.set_ylabel("Pressure (hPa)")
    axS.set_xlabel("Time (s)")

    axS.grid(False, 'both', 'both')

    if fout!='':
        plt.savefig(fout)
        plt.close()

def plot_water_timeseries(parcel_trace, ax, fout):

    if ax==None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    ax.plot(parcel_trace['time'], parcel_trace['wv']*1e3, color='k', label='water vapour')
    ax.plot(parcel_trace['time'], parcel_trace['wl']*1e3, color='r', label='liquid water')
    ax.plot(parcel_trace['time'], parcel_trace['wi']*1e3, color='b', label='ice')

    ax.set_ylabel("Mixing Ratio (g/kg)")
    ax.set_xlabel("Time (s)")
    ax.legend()
    if fout!='':
        plt.savefig(fout)
        plt.close()
        

def plot_particle_state(parcel_trace, aerosol_trace, aero_species, activated_timeseries, freezing_times, ax, fout):
    # plots the final wet radius of each particle and the time at which that particle activated and froze


    if ax==None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    n=0
    rs = aerosol_trace.values
    #ax.plot(aero_species.r_drys*1e6, aero_species.r_drys*1e6, color='grey') # plot the 1:1 line
    ax.plot(aero_species.r_drys*1e6, rs[-1,:]*1e6, color='black', label='final')
    ax.plot(aero_species.r_drys*1e6, rs[0,:]*1e6, color='black', linestyle='dashed', label='start')
    ax.plot(aero_species.r_drys*1e6, np.amax(rs, axis=0)*1e6, color='black', linestyle='dotted', label='max')
    times=parcel_trace['time']
    # get time when particles activated
    tact=np.zeros(rs.shape[1])+np.nan
    for rx in range(rs.shape[1]):
        ix=np.where((activated_timeseries[1:,rx]==True)&(activated_timeseries[:-1,rx]==False))
        if len(ix[0])>0:
            tact[rx]=times[ix[0][0]+1]
        if activated_timeseries[0,rx]:
            tact[rx]=times[0] # it was activated at the start

    ix=np.where(np.isfinite(tact)==False)
    if len(ix[0])>0:
        print('bins',ix[0], 'never activated')
        
    ax2=ax.twinx()
    ax2.scatter(aero_species.r_drys*1e6, tact, color='red', marker='x', s=2, label='activates') 

    ix=np.where(freezing_times<0)
    if len(ix[0])>0:
        print('bins',ix[0], 'never froze')
    ix=np.where(freezing_times>=0)
    ax2.scatter(aero_species.r_drys[ix]*1e6, freezing_times[ix], color='blue', marker='+', s=2, label='freezes') 
    ax.set_title(aero_species.species)
    ax.set_xlabel('dry radius ($\mu$m)')
    ax.set_ylabel('wet radius ($\mu$m)')
    ax.legend(bbox_to_anchor=(0.75, -0.2),ncol=3)
    ax2.set_ylabel('Time at activation / freezing (s)')
    ax2.legend(loc='lower right')
    ax.semilogx()
    ax.semilogy()

    if fout!='':
        plt.savefig(fout)
        plt.close()
   
def plot_activation_timeseries(parcel_trace, aerosol_trace, aero_species, activated_timeseries, freeze_times, ax, fout):
    # for a selected few dry radii, plot growth of particles
    if ax==None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    n=0
    times=parcel_trace['time']
    nt=len(times)
    rs = aerosol_trace.values 
    # just keep state of every 10th bin
    rs2=rs[:,::10]*1e6
    max_rs2=np.amax(rs2)
    min_rs2=np.amin(rs2)
    rdry=aero_species.r_drys[::10]*1e6
    act_timeseries2=activated_timeseries[:,::10]
    freeze_times2=freeze_times[::10]
    colors=['black', 'blue', 'magenta', 'green', 'orange', 'purple']
    label_inact='unactivated'
    label_act='activated'
    label_nucl='nucleating'
    print('freeze_times', freeze_times)
    for b in range(rs2.shape[1]):
        label='r={r:1.2e}'.format(r=rdry[b])
        before_freeze=(times<freeze_times2[b])|(freeze_times2[b]<0)
        after_freeze=(times>=freeze_times2[b])&(freeze_times2[b]>=0)
        activated=act_timeseries2[:,b]
        # as sometimes droplets can grow and shrink again they don't just get activated
        # and stay activated. They will however not unfreeze if they freeze
        # so its best to plot the whole timeseries up to the point they freeze in solid
        # and plot the timeseries after they freeze in dashed and use scatter to mark
        # where changes to activated and non-activated happened
        ix=np.where(before_freeze)
        ax.plot(times[ix[0]], rs2[ix[0],b], color=colors[b], linestyle='solid', label=label)
        ix=np.where(after_freeze)
        if len(ix[0])>0:
            ax.plot(times[ix[0]], rs2[ix[0],b], color=colors[b], linestyle='dashed')
            
        ix=np.where((activated[:-1]==False)&(activated[1:]==True)) # switch to activated
        if len(ix[0])>0:
            ax.scatter(times[ix[0]+1], rs2[ix[0]+1,b], color=colors[b], marker='+')
        ix=np.where((activated[:-1]==True)&(activated[1:]==False)) # switch to unactivated
        if len(ix[0])>0:
            ax.scatter(times[ix[0]+1], rs2[ix[0]+1,b], color=colors[b], marker='x')

    ix=np.where(rs[-1,:]-aero_species.r_drys<0.00001e-6)
    if len(ix[0])>0:
        print(len(ix[0]), 'particle bins did not grow overall')
        print(ix[0])

    # create scatter points outside plot area so we get details in the legend
    ax.scatter(times[:2], [max_rs2*1e3]*2, color='black', marker='x', label=label_inact)
    ax.scatter(times[:2], [max_rs2*1e3]*2, color='black', marker='+', label=label_act)
    ax.plot(times[:2], [max_rs2*1e3]*2, color='black', linestyle='dashed', label=label_nucl)
    ax.set_ylim([min_rs2*0.9, max_rs2*1.1])
    ax.set_title(aero_species.species)
    ax.set_ylabel('wet radius ($\mu$m)')
    ax.set_xlabel('Time (s)')
    ax.legend()
    ax.semilogy()

    if fout!='':
        plt.savefig(fout)
        plt.close()

def plot_particle_growth(parcel_trace, aerosol_trace, aero_species, ax):
    times=parcel_trace['time']
    nt=len(times)
    rs = np.transpose(aerosol_trace.values*1e6) # need x and y other way round
    rdrys=aero_species.r_drys*1e6
    im=ax.imshow(rs, interpolation=None, norm=colors.LogNorm(), origin='lower')
    maxt=int(times[-1])
    time_ticks=[np.argmin(abs(times-i)) for i in range(0,maxt,5)]
    time_labels=['{t:1.1f}'.format(t=times[i]) for i in time_ticks]
    ax.set_xticks(time_ticks, labels=time_labels)
    ax.set_xlabel('time (s)')
    # take every bins closest to 0.01,0.1,0.5,1.0 microns
    rbins=[0.01,0.1,0.5,1.0]
    rticks=[np.argmin(abs(rdrys-i)) for i in rbins]
    rlabels=['{r:1.2e}'.format(r=rdrys[i]) for i in rticks]
    ax.set_title(aero_species.species)
    ax.set_ylabel('dry radius ($\mu$m)')
    ax.set_yticks(rticks, labels=rlabels)
    #ax.semilogy()
    return im
    
def plot_nice_timeseries(parcel_trace, aero_species, freeze_times, rho_a_d, fout):  
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    times=parcel_trace['time']
    nt=len(times)
    for k in aero_species.keys():
        nice=np.zeros(nt)
        Nis=aero_species[k].Nis*1e6/rho_a_d  # We need in kg-1
        for t in range(nt):
            after_freeze=(times[t]>=freeze_times[k])&(freeze_times[k]>=0)
            ix=np.where(after_freeze)
            if len(ix[0])>0:
                nice[t]=np.sum(Nis[ix[0]])
        ax.plot(times, nice*1e-12, label=k)
        print(nice, 'ice particles per kg of air')
    ax.set_ylabel('NICE (10$^{-12}$kg-1)')
    ax.set_xlabel('time (s)')
    ax.legend()
    plt.savefig(fout)
    plt.close()
