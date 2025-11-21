import numpy as np, xarray as xr
import matplotlib.pylab as plt
from scipy.signal import welch
import pickle, random, re, io
from eofs.xarray import Eof
from statsmodels.tsa.api import VAR
import pandas as pd, seaborn as sns
from contextlib import redirect_stdout
import dask, timeit, os, shutil, datetime
from distributed import Client
from scipy import stats
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings; warnings.filterwarnings('ignore')
import matplotlib.gridspec as gridspec;
import regionmask; import matplotlib as mpl
# Figure formatting
mpl.rcParams['font.family'] = 'sans-serif' # Change default font family
tickfontsize = 14; axislabelfontsize=16
titlefontsize=18; legendfontsize=14
color1 = 'goldenrod'; color2='teal'

def calc_weights(ds):
    '''
    Construct area-weighting matrix in shape of eof function input data
    '''
    calc_weights = list(np.cos(np.deg2rad(ds.lat).values))
    shape_weights = np.tile(calc_weights, (len(ds.lon), 1))
    weights = shape_weights.T
    
    weights = xr.DataArray(weights, coords=[ds['lat'], ds['lon']], 
                           dims=['lat', 'lon'])
    return weights

def concat_with_monthids(ds):
    
    t1 = ds.time[0]; tn = ds.time[-1]
    ds = ds.sel(time=slice(t1, tn)) 

    arrays = []
    n = 0
    month_ids = [] # integer identifiers for month associated with each timepoint (e.g. 1=Jan, 2=Feb, etc)

    for ens in ds.ensemble:

        vals = ds.sel(ensemble=ens)

        months = vals['time.month'].values.tolist()

        month_ids.extend(months)

        tstart = n
        n = n + len(vals.time)
        tfinish = n

        vals['time'] = np.arange(tstart, tfinish)

        arrays.append(vals)

    concatted = xr.concat(arrays, dim='time') 
    concatted = concatted.to_dataset(name='anoms').drop('ensemble')
    concatted['month'] = month_ids
    
    return concatted

def calc_EOFs(ds, path, filename):
    
    from pathlib import Path
    
    # if the file already exists
    if (Path(path + filename)).is_file():
        print('Loading EOF solver')
        
        # Load EOF solver object
        with open(path + filename, "rb") as fp:
            solver = pickle.load(fp)
    else:
        print('calculating EOFs')
 
        weights = calc_weights(ds)
        solver = Eof(ds, weights=weights) #, center=False)

        # Save solver as pickled object
        with open(path + filename, "wb") as fp: 
            pickle.dump(solver, fp)
        print('done')
    return solver 

def areaweighted_mean(ds):    
    
    if hasattr(ds, 'lat'): # if latitude dimension has label 'lat'
        weights = np.cos(np.deg2rad(ds.lat)) 
        weights.name = "weights"
        array_weighted = ds.weighted(weights)
        weighted_mean = array_weighted.mean(("lon", "lat"))
        
    elif hasattr(ds, 'latitude'): # if latitude dimension has label 'latitude'
        weights = np.cos(np.deg2rad(ds.latitude)) 
        weights.name = "weights"
        array_weighted = ds.weighted(weights)
        weighted_mean = array_weighted.mean(("longitude", "latitude"))
        
    return weighted_mean

def plot_spectrum(data, label, color, linestyle=None):
    
    f, Pxx = welch(data, fs=12, nperseg=128)
    plt.loglog(f, Pxx, label=label, color=color, linestyle=linestyle)
    plt.legend()
    
def autocorr(x):
    '''
    Test autocorrelations
    '''
    result = np.correlate(x, x, mode='full')
    
    return result[result.size//2:] / max(result)

def add_gmean(data_array, var_name):
    '''
    For a gridded data array, calculate the area-weighted mean of the gridded data and save it 
    under variable name 'gmean'
    
    convert the data array into a dataset with the gridded variable name as var_name
    and gmean as the variable for the global mean
    '''
    ds = data_array.to_dataset(name=var_name)
    
    ds['gmean'] = areaweighted_mean(ds[var_name])
    
    return ds

def unstack_time(ds, esize):
    '''
    Group T anomalies from time-concattenated data into separate ensemble members
    esize is the number of ensemble members that have been stacked in time
    '''
    # Change time coordinate to integer values
    ds['time'] = np.arange(0, len(ds.time))

    if 'ensemble' in ds.coords:
        ds = ds.drop_vars('ensemble')    
        
    tseries = [] # list to hold time series for each ensemble member
    elen = len(ds.time) / esize  # length of each ensemble member = total length / number of members
    
    for ens in range(esize):
        tseries_n = (ds.sel(time=slice(ens * elen, (ens+1) * elen -1)))
        tseries_n['time'] = np.arange(0, elen)
        tseries.append(tseries_n)
        
    new_ds = xr.concat(tseries, dim='ensemble')
    new_ds = new_ds.assign_coords({'ensemble':new_ds.ensemble})
    
    return new_ds

def stack_time(ds):
    '''
    Concatenate separate ensemble members in time
    '''
    ens_list = []
    
    for i, ens in enumerate(ds.ensemble.values):
        ens_member = ds.sel(ensemble=ens)
        elen = len(ens_member.time)
        ens_member['time'] = np.arange(i * elen, (i+1) * elen)
        ens_list.append(ens_member)
        
    ds_concatted = xr.concat(ens_list, dim='time')

    return ds_concatted

def createRandomSortedList(num_entries, start = 0, end = 49):
    
    arr = []
    tmp = random.randint(start, end)
    
    for x in range(num_entries):
        
        while tmp in arr:
            tmp = random.randint(start, end)
            
        arr.append(tmp)
        
    arr.sort()
    
    return arr


def custom_sum_along_mode(x, axis):
    """
    Custom summation function to sum along the 'mode' dimension while preserving other dimensions.
    """
    return np.sum(x, axis=axis, keepdims=True)


def compare_T_pdfs(t1, t2, name1, name2, color1, color2, title, 
                   tickfontsize = 14, axislabelfontsize=16,
                   titlefontsize=18, legendfontsize=14, ax=None):
    '''Global mean
    '''
    sns.kdeplot(t1, color=color1, label=name1, ax=ax)
    sns.kdeplot(t2, color=color2, label=name2, ax=ax)
    
    ax.set_xlabel('T anomaly [K]', fontsize=axislabelfontsize)
    ax.set_ylabel('density [K$^{-1}$]', fontsize=axislabelfontsize)

    # Calculate standard deviation ranges
    t1_std_range = np.array([t1.mean().values-t1.std().values, 
                             t1.mean().values+t1.std().values])
    t2_std_range = np.array([t2.mean().values-t2.std().values, 
                             t2.mean().values+t2.std().values])
    # Get y-axis limits
    ymin, ymax = ax.get_ylim()  # Get current y-axis limits
    
    ax.axvline(t1.mean(), color=color1, linestyle='dashdot', linewidth=1,
               label=name1 + ' $\mu$ +/- 1 $\sigma$')
    ax.axvline(t2.mean(), color=color2, linestyle='--', linewidth=1,
               label=name2 + ' $\mu$ +/- 1 $\sigma$')
    
    ax.axvline(t1.mean()+t1.std(), color=color1, linestyle='dashdot', linewidth=1)
    ax.axvline(t2.mean()+t2.std(), color=color2, linestyle='--', linewidth=1)
    
    ax.axvline(t1.mean()-t1.std(), color=color1, linestyle='dashdot', linewidth=1)
    ax.axvline(t2.mean()-t2.std(), color=color2, linestyle='--', linewidth=1)
    
    xticks = [-.5, -.25, 0, .25, .5]
    ax.set_xticks(xticks, labels=xticks, fontsize=tickfontsize)
    
    yticks = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    ax.set_yticks(yticks, labels=yticks, fontsize=tickfontsize)
    
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('dimgrey'); ax.spines['left'].set_color('dimgrey') 
    
    ax.set_ylim(ymin, ymax)
    ax.legend(fontsize=legendfontsize)
    
def calc_psd_stats(data, fs=12, nperseg=128):
    """
    Calculates the mean and standard deviation of the PSD at each frequency
    for several ensemble members in an xarray data array.
    Parameters:
        - data: The xarray data array containing the time series.
        - fs: The sampling frequency of the time series.
    Returns:
        - frequencies: Array of frequencies.
        - mean_psd, std_psd: Arrays of mean and stds of PSD values at each frequency.
    """
    # Get the time series data from the dataset
    tseries_list = [data.sel(ensemble=ens).values for ens in data['ensemble'].values]

    # Calculate PSD for each time series
    psds = []
    for tseries in tseries_list:
        f, psd = welch(tseries, fs=fs, nperseg=128)
        psds.append(psd)

    # Calculate mean and standard deviation of PSD at each frequency
    psds_array = np.array(psds)
    mean_psd = np.mean(psds_array, axis=0)
    std_psd = np.std(psds_array, axis=0)

    return f, mean_psd, std_psd
    
def plot_GMST_psd_spread(ax, original_T, InVERT_T, nmodes, title, color1, color2,
                         name1, name2, tickfontsize = 14, axislabelfontsize=16,
                         titlefontsize=18, legendfontsize=14,):
    
    f, mean_psd, std_psd = calc_psd_stats(original_T)
    invert_f, invert_mean_psd, invert_std_psd = calc_psd_stats(InVERT_T)

    ax.loglog(f, mean_psd, color=color1, linestyle='--', label=name1+' $\mu_{PSD}$')
    ax.loglog(f, mean_psd - std_psd, color=color1)
    ax.loglog(f, mean_psd + std_psd, color=color1)
    
    ax.fill_between(f, mean_psd - std_psd, mean_psd + std_psd, 
                     color=color1, alpha=0.2, label=name1+' $\mu_{PSD}$ +/- 1 $\sigma_{PSD}$')

    ax.loglog(invert_f, invert_mean_psd, color=color2, linestyle='dotted', label=name2+' $\mu_{PSD}$')
    ax.loglog(invert_f, invert_mean_psd - invert_std_psd, color=color2)
    ax.loglog(invert_f, invert_mean_psd + invert_std_psd, color=color2)

    ax.fill_between(invert_f, invert_mean_psd - invert_std_psd, 
                               invert_mean_psd + invert_std_psd, 
                     color=color2, alpha=0.2, label=name2+' $\mu_{PSD}$  +/- 1 $\sigma_{PSD}$')
    ax.set_ylim(1e-4, 1e-1)
    yticks = [1e-4, 1e-3, 1e-2, 1e-1]
    yticklabels = [f'$10^{{{np.log10(y):.0f}}}$' for y in yticks]
    ax.set_yticks(yticks, labels=yticklabels, fontsize=tickfontsize)
    
    ax.set_xlim(np.min(f), np.max(f))
    xticks = [1e-1, 5e-1, 1e0, 5e0]

    xticklabels = []
    for x in xticks:
        base = 10
        exponent = int(np.floor(np.log10(abs(x))))  # Get the exponent
        coefficient = x / (base ** exponent)  # Calculate the coefficient

        if coefficient == 1:  # Handle cases where coefficient is 1
            label = f'$10^{{{exponent}}}$'  # Just show 10^exponent
        elif coefficient.is_integer():  # Handle integer coefficients
            label = f'${int(coefficient)} \\times 10^{{{exponent}}}$'  # Show coefficient * 10^exponent
        else:  # Handle decimal coefficients
            label = f'${coefficient:.1f} \\times 10^{{{exponent}}}$'  # Show coefficient with 1 decimal * 10^exponent
        xticklabels.append(label)

    ax.set_xticks(xticks, labels=xticklabels, fontsize=tickfontsize)
    
    ax.set_ylabel('power spectral density [K$^{2}$ Hz$^{-1}$]', fontsize=axislabelfontsize)
    ax.set_xlabel('frequency [year$^{-1}$]', fontsize=axislabelfontsize)
    
    ax.legend(fontsize=legendfontsize)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('dimgrey'); ax.spines['left'].set_color('dimgrey')  


def calc_emean_autocorrs(Tanoms_unstacked):
    
    autocorrs = np.zeros((len(Tanoms_unstacked.ensemble), 13))
    for i, ens in enumerate(Tanoms_unstacked.ensemble):
        ac = autocorr(Tanoms_unstacked.sel(ensemble=ens))
        autocorrs[i] = (ac[0:13])
    return np.mean(autocorrs, axis=0)

def calc_ensemble_std_autocorrs(Tanoms_unstacked):

    autocorrs = np.zeros((len(Tanoms_unstacked.ensemble), 13))
    for i, ens in enumerate(Tanoms_unstacked.ensemble):
        ac = autocorr(Tanoms_unstacked.sel(ensemble=ens))
        autocorrs[i] = (ac[0:13])
    return np.std(autocorrs, axis=0)

def calc_efold_time(data):
    """Calculates the e-folding time of the autocorrelation of a time series.
    Args:
        data: The input time series data as a 1D NumPy array.
    Returns:
        The e-folding time (in units of the time series' sampling interval).
    """
    # Calculate the autocorrelation function
    autocorr_func = np.correlate(data, data, mode='full')

    # Normalize and keep only positive lags
    autocorr_func = autocorr_func[autocorr_func.size // 2:] / autocorr_func.max()

    # Find the index where autocorrelation drops to 1/e
    e_folding_index = np.where(autocorr_func < 1 / np.e)[0][0]

    # Return the e-folding time (index corresponds to time units)
    return e_folding_index  # Return as a scalar

def calc_eft_stats(ds):
    '''
    Calculate ensemble mean and standard deviations of autocorrelation e-folding times 
    '''
    efold_times = []
    
    for ens in ds.ensemble.values:
        
        efold_times.append(calc_efold_time(ds.sel(ensemble=ens)))
        
    return np.mean(efold_times), np.std(efold_times)

def compare_autocorrs_emean(T1, name1, T2, name2, title, color1, color2, 
                            tickfontsize = 14, axislabelfontsize=16,
                            titlefontsize=18, legendfontsize=14,
                            markersize=10, capsize=6, elinewidth=1.5, 
                            markeredgewidth=1.5, ax=None):  # Add ax argument
    '''
    Compare scenarios for one data type (either original or emulated)
    '''
    if ax is None:
        ax = plt.gca()  # Get current axes if ax is not provided

    means = calc_emean_autocorrs(T1)
    stds = calc_ensemble_std_autocorrs(T1)

    # Plot error bars with desired color
    ax.errorbar(np.arange(0, 13), means, yerr=stds,
                fmt='none', markersize=markersize, capsize=capsize, color=color1, 
                label='$\mu_{'+name1+'}$'+' +/- 1$\sigma_{ '+name1+'}$',  # Use fmt='none' to hide default marker
                ecolor=color1, elinewidth=elinewidth,
                markeredgewidth=markeredgewidth)  

    # Plot markers separately with unfilled style
    ax.plot(np.arange(0, 13), means, 'o', markersize=markersize, color=color1,  # Plot markers with desired color
             markerfacecolor=color1, markeredgecolor=color1, label=name1)  # Set marker properties

    means = calc_emean_autocorrs(T2)
    stds = calc_ensemble_std_autocorrs(T2)

    # Plot error bars with desired color
    ax.errorbar(np.arange(0, 13), means, yerr=stds,
                fmt='none', markersize=markersize, capsize=capsize, color=color2, 
                label='$\mu_{'+name2+'}$'+' +/- 1$\sigma_{ '+name2+'}$',  # Use fmt='none' to hide default marker
                ecolor=color2, elinewidth=elinewidth, 
                markeredgewidth=markeredgewidth)  

    # Plot markers separately with unfilled style
    ax.plot(np.arange(0, 13), means, 'o', markersize=markersize, color=color2,  # Plot markers with desired color
             markerfacecolor='none', markeredgecolor=color2, label=name2,
             markeredgewidth=markeredgewidth)  # Set marker properties

    ax.set_xlim([-0.25, 12.25]); ax.set_ylim([0, 1.05])  # Use ax.set_ylim
    yticks = [0, 0.5, 1.0]; ax.set_yticks(yticks, labels=yticks, fontsize=tickfontsize)
    xticks = [0,2,4,6,8,10,12]; ax.set_xticks(xticks, labels=xticks, fontsize=tickfontsize)

    ax.set_ylabel('autocorrelation', fontsize=axislabelfontsize)  # Use ax.set_ylabel
    ax.set_xlabel('lag [months]', fontsize=axislabelfontsize)  # Use ax.set_xlabel
    
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('gray'); ax.spines['left'].set_color('gray')  
    
    ax.axvline(calc_eft_stats(T1)[0], color=color1, label=r'$\tau_{'+name1+'}$', linestyle='--')
    ax.axvline(calc_eft_stats(T2)[0], color=color2, label=r'$\tau_{'+name2+'}$', linestyle='--')
    
    ax.legend(fontsize=legendfontsize)  


def plot_GMST_comparisons(T_lens, T_invert,
                          T_lens_stacked, T_invert_stacked,
                          name1, name2,
                          color1, color2, path, save_name,
                          tickfontsize = 14, axislabelfontsize=16,
                          titlefontsize=18, legendfontsize=14):

    fig, axes = plt.subplot_mosaic("a;b;c", figsize=(8, 16));  # Create figure and axes
    
    for n, (key, ax) in enumerate(axes.items()):
        ax.text(-0.1, 1.075, key, transform=ax.transAxes, 
                size=20, weight='bold')
        
        if n == 0:
            ## PDF
            compare_T_pdfs(T_lens_stacked.gmean, T_invert_stacked.gmean, 
                           name1, name2, color1, color2, 'Probability distribution functions', ax=ax)
        if n == 1:
            ## PSD
            plot_GMST_psd_spread(ax=ax, original_T=T_lens.gmean, 
                                 InVERT_T=T_invert.gmean, nmodes=100, 
                                 title='Power spectral density curves', color1=color1, color2=color2,
                                     name1=name1, name2=name2)
        if n == 2:
            ## Autocorrelations
            compare_autocorrs_emean(T_lens.gmean, name1, T_invert.gmean, name2, 
                                    'Autocorrelations', color1, color2, ax=ax)
    fig.tight_layout(pad=1)
    plt.show()
    
    fig.savefig(path + save_name + '.pdf'); fig.savefig(path + save_name + '.png')


def save_region_means(ds, name, path):
    
    import regionmask
    
    # Determine temperature variable name
    temp_var_name = 'T' if 'T' in ds.variables else 'anoms' if 'anoms' in ds.variables else None

    if temp_var_name is None:
        raise ValueError("Temperature variable not found ('T' or 'anoms')")
    
    # Get AR6 land region mask
    mask = regionmask.defined_regions.ar6.land.mask(ds)
    
    # Calculate weighted mean over the regions
    mask = mask.fillna(0)

    region_mean = ds[temp_var_name].groupby(mask).mean()
    
    region_mean.to_netcdf(path + name + '_AR6_region_mean_Tanoms.nc')


def plot_regional_psd_spread(ax, original_T, InVERT_T, title, color1, color2,
                             tickfontsize = 14, axislabelfontsize=16,
                             titlefontsize=18, legendfontsize=14):

    f, mean_psd, std_psd = calc_psd_stats(original_T)
    invert_f, invert_mean_psd, invert_std_psd = calc_psd_stats(InVERT_T)

    ax.loglog(f, mean_psd, color=color1, linestyle='--', label='LENS2 mean PSD')
    ax.loglog(f, mean_psd - std_psd, color=color1)
    ax.loglog(f, mean_psd + std_psd, color=color1)
    
    ax.fill_between(f, mean_psd - std_psd, mean_psd + std_psd, 
                    color=color1, alpha=0.2, label='LENS2 $\mu$ +/- 1 $\sigma$')

    ax.loglog(invert_f, invert_mean_psd, color=color2, linestyle='dotted', label='InVERT mean PSD')
    ax.loglog(invert_f, invert_mean_psd - invert_std_psd, color=color2)
    ax.loglog(invert_f, invert_mean_psd + invert_std_psd, color=color2)

    ax.fill_between(invert_f, invert_mean_psd - invert_std_psd, 
                              invert_mean_psd + invert_std_psd, 
                     color=color2, alpha=0.2, label='InVERT $\mu$ +/- 1 $\sigma$')

    ymin,ymax = ax.get_ylim(); xmin,xmax = ax.get_xlim()  # Get current axis limits  
    
    yticks = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]
    yticklabels = [f'$10^{{{np.log10(y):.0f}}}$' for y in yticks]
    
    ax.set_yticks(yticks, labels=yticklabels, fontsize=tickfontsize)
    ax.set_ylim(ymin, ymax*1.5)
    ax.set_xlim(np.min(f), np.max(f))
    xticks = [1e-1, 5e-1, 1e0, 5e0]

    xticklabels = []
    for x in xticks:
        base = 10
        exponent = int(np.floor(np.log10(abs(x))))  # Get the exponent
        coefficient = x / (base ** exponent)  # Calculate the coefficient

        if coefficient == 1:  # Handle cases where coefficient is 1
            label = f'$10^{{{exponent}}}$'  # Just show 10^exponent
        elif coefficient.is_integer():  # Handle integer coefficients
            label = f'${int(coefficient)} \\times 10^{{{exponent}}}$'  # Show coefficient * 10^exponent
        else:  # Handle decimal coefficients
            label = f'${coefficient:.1f} \\times 10^{{{exponent}}}$'  # Show coefficient with 1 decimal * 10^exponent
        xticklabels.append(label)
        
    ax.set_xticks(xticks, labels=xticklabels, fontsize=tickfontsize)
    
    ax.set_ylabel('power spectral density [K$^{2}$ Hz$^{-1}$]', fontsize=axislabelfontsize)
    ax.set_xlabel('frequency [year$^{-1}$]', fontsize=axislabelfontsize)
    ax.legend(fontsize=legendfontsize)

    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('gray'); ax.spines['left'].set_color('gray')  
    
    ax.set_title(title, fontsize=titlefontsize) 


def plot_regional_T_pdfs(t1, t2, name1, name2, color1, color2, title, ax=None,
                        tickfontsize = 14, axislabelfontsize=16,
                          titlefontsize=18, legendfontsize=14):

    sns.kdeplot(t1, color=color1, label=name1, ax=ax)
    sns.kdeplot(t2, color=color2, label=name2, ax=ax)
    
    ax.set_title(title, fontsize=titlefontsize)
    ax.set_xlabel('T anomaly [K]', fontsize=axislabelfontsize)
    ax.set_ylabel('density [K$^{-1}$]', fontsize=axislabelfontsize)
    
    ymin, ymax = ax.get_ylim(); xmin, xmax = ax.get_xlim()  # Get current axis limits
    
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('dimgrey'); ax.spines['left'].set_color('dimgrey') 
    
    xrange = min(abs(xmin), abs(xmax)) # Calculate the range for x-axis ticks
    ax.set_xlim(-xrange, xrange); ax.set_ylim(ymin, ymax)
    
    # Generate x-axis ticks centered around zero
    xticks = np.round(np.linspace(-xrange*.75, xrange*.75, 5), 1)
    ax.set_xticks(xticks, labels=xticks, fontsize=tickfontsize)
    
    yticks = np.round(np.linspace(ymin, ymax, 2),1)
    ax.set_yticks(yticks, labels=yticks, fontsize=tickfontsize)
    
    ax.legend(fontsize=legendfontsize)

def plot_regional_emean_autocorrs(T1, name1, T2, name2, title, color1, color2, nlags,
                                  markersize=10, capsize=6, elinewidth=1.5, 
                                  markeredgewidth=1.5, ax=None,
                                 tickfontsize = 14, axislabelfontsize=16,
                          titlefontsize=18, legendfontsize=14):  
    ''''''
    means = calc_emean_autocorrs(T1)
    stds = calc_ensemble_std_autocorrs(T1)
    # Plot error bars with desired color
    ax.errorbar(np.arange(0, nlags+1), means[0:nlags+1], yerr=stds[0:nlags+1],
                fmt='none', markersize=markersize, capsize=capsize, color=color1, 
                label=name1+' $\mu$ +/- 1$\sigma$',  # Use fmt='none' to hide default marker
                ecolor=color1, elinewidth=elinewidth,
                markeredgewidth=markeredgewidth)  
    # Plot markers separately with unfilled style
    ax.plot(np.arange(0, nlags+1), means[0:nlags+1], 'o', markersize=markersize, color=color1,  # Plot markers with desired color
             markerfacecolor=color1, markeredgecolor=color1, label=name1)  # Set marker properties

    means = calc_emean_autocorrs(T2)
    stds = calc_ensemble_std_autocorrs(T2)
    # Plot error bars with desired color
    ax.errorbar(np.arange(0, nlags+1), means[0:nlags+1], yerr=stds[0:nlags+1],
                fmt='none', markersize=markersize, capsize=capsize, color=color2, 
                label=name2+' $\mu$ +/- 1$\sigma$',  # Use fmt='none' to hide default marker
                ecolor=color2, elinewidth=elinewidth, 
                markeredgewidth=markeredgewidth)  
    # Plot markers separately with unfilled style
    ax.plot(np.arange(0, nlags+1), means[0:nlags+1], 'o', 
             markersize=markersize, color=color2,  # Plot markers with desired color
             markerfacecolor='none', markeredgecolor=color2, label=name2,
             markeredgewidth=markeredgewidth)  # Set marker properties

    ax.set_xlim([-0.25, nlags+.25]); ax.set_ylim([-0.05, 1.05]) 
    yticks = [0, 0.5, 1.0]
    ax.set_yticks(yticks, labels=yticks, fontsize=tickfontsize)
    
    xticks = [int(n) for n in np.linspace(0, nlags, (nlags//2)+1)]
    ax.set_xticks(xticks, labels=xticks, fontsize=tickfontsize)

    ax.set_title(title, fontsize=titlefontsize) 
    ax.set_ylabel('autocorrelation', fontsize=axislabelfontsize) 
    ax.set_xlabel('lag [months]', fontsize=axislabelfontsize)  
    
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('gray'); ax.spines['left'].set_color('gray') 
    ax.legend(fontsize=legendfontsize)  


def plot_regional_comparison(T_og, T_invert, T_og_stacked, T_invert_stacked, 
                             region, name1, name2, color1, color2,
                             tickfontsize = 14, axislabelfontsize=16,
                             titlefontsize=18, legendfontsize=14):
    import regionmask
    
    og = find_var_name(T_og).sel(mask=region)
    invert = find_var_name(T_invert).sel(mask=region)
    
    og_stacked = find_var_name(T_og_stacked).sel(mask=region)
    invert_stacked = find_var_name(T_invert_stacked.sel(mask=region))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5));  # Create figure and axes
    
    ## PDF
    plot_regional_T_pdfs(t1=og_stacked, t2=invert_stacked, 
                   name1=name1, name2=name2,
                   color1=color1, color2=color2, 
                   title='PDF', ax=axes[0])
    ## PSD
    plot_regional_psd_spread(axes[1], og, invert, title='PSD', 
                             color1=color1, color2=color2)  # Call plot_psd_spread() with axes[1]
    axes[1].set_title('PSD', fontsize=titlefontsize) #set title for plot_psd_spread
    
    ## Autocorrelations
    axes[2].set_title('Autocorrelations', fontsize=titlefontsize) # Set title for the third subplot
    
    plot_regional_emean_autocorrs(og, name1, invert, name2, title='Autocorrelations', ax = axes[2],
                            color1=color1, color2=color2, nlags=12)  #plot on third subplot
    plt.tight_layout()  
    
    fig.suptitle('AR6 region '+str(region) + \
                 ': ' + str(regionmask.defined_regions.ar6.land[region]).split(': ')[1].split('(')[0], 
                 fontsize=20, y=1.1)
    plt.show()


def find_var_name(ds):
    
    temp_var_name = 'T' if 'T' in ds.variables else 'anoms' if 'anoms' in ds.variables else None
    if temp_var_name is None: raise ValueError("Temperature variable not found ('T' or 'anoms')")
        
    return ds[temp_var_name]


def compare_MSE_to_emean_PSD(tseries_1, tseries_2, fs=12, nperseg=128):
    """
    Compares the shapes of two PSD curves on a log-log scale using log10.
    Args:
        tseries_1: NumPy array representing the first time series.
        time_series_2: NumPy array representing the second time series.
    Returns:
        The Mean Squared Error (MSE) between the log10-transformed PSD curves.
        
    This code primarily focuses on differences in the shapes of the curves, 
        rather than absolute power differences. The logarithmic transformations applied to
        both the frequency and PSD values before calculating the MSE emphasizes the 
        relative power distribution across frequencies
    """
    # Calculate ensemble mean PSD of first time series
    f1, Pxx1, std_pxx1 = calc_psd_stats(tseries_1)
    
    # Exclude zero frequency component to avoid infinities when taking log
    # Exclude highest frequency component to avoid sudden psd "drop-off" 
    f1 = f1[1:-1];  Pxx1 = Pxx1[1:-1]
    
    # Take logarithm of PSD values and frequencies
    log_f1 = np.log10(f1);  log_Pxx1 = np.log10(Pxx1)
    
    # Get the time series data from the second time series by ensemble
    tseries_list = [tseries_2.sel(ensemble=ens).values for ens in tseries_2['ensemble'].values]

    # Initialize arrays to store PSDs
    MSEs = []
    
    # Calculate PSD for each time series
    for tseries in tseries_list:
        
        f2, Pxx2 = welch(tseries, fs=fs, nperseg=nperseg)  # Use Welch's method
        f2 = f2[1:-1];  Pxx2 = Pxx2[1:-1]

        # Take logarithm of PSD values and frequencies
        log_f2 = np.log10(f2);  log_Pxx2 = np.log10(Pxx2)

        ## if log_f1 and log_f2 are identical:
        if np.all(log_f1 == log_f2):
            # Calculate MSE on log-transformed PSD values
            mse_log = np.mean((log_Pxx2 - log_Pxx1)**2)
        else: print('error: frequencies not identical')
    
        MSEs.append(mse_log)
        
    return MSEs


def welch_psd(x, fs=12, **kwargs):
    """
    Calculates the power spectral density using Welch's method.
        x (array-like): Time series data for a single grid cell.
        fs (float, optional): Sampling frequency. Default to 12.
        **kwargs: Additional keyword arguments to pass to `scipy.signal.welch`.
    Returns:
        array-like: Reshaped PSD array with correct shape for apply_ufunc.
    """
    f, Pxx = welch(x, fs=fs, **kwargs)

    return Pxx  


def calc_emean_gridcell_MSE(ref_grid_psd, test_grid_psd):
    '''
    # Calculate MSE between each InVERT ensemble members' gridcell PSDs
    # and the ensemble mean gridcell PSDs of LENS_outsample
    '''
    test_grid_psd_members = np.zeros((len(test_grid_psd.ensemble.values), 
                                      len(test_grid_psd.lat.values),
                                      len(test_grid_psd.lon.values)))
    
    for i, ens in enumerate(test_grid_psd.ensemble.values):

        # Reference PSD: LENS ensemble mean grid PSD
        f1, Pxx1 = ref_grid_psd.frequency, ref_grid_psd.emean
        f2 = test_grid_psd.frequency

        # PSD to compare: each InVERT gridcell PSD
        Pxx2 = test_grid_psd.sel(ensemble=ens).psd

        # Calculate and plot gridcell ensemble mean MSE
        f1 = f1.isel(frequency=slice(1,-1));  Pxx1 = Pxx1.isel(frequency=slice(1,-1))
        f2 = f2.isel(frequency=slice(1,-1));  Pxx2 = Pxx2.isel(frequency=slice(1,-1))

        # Take logarithm of PSD values and frequencies
        log_f1 = np.log10(f1);  log_Pxx1 = np.log10(Pxx1)
        log_f2 = np.log10(f2);  log_Pxx2 = np.log10(Pxx2)

        ## if log_f1 and log_f2 are identical:
        if np.all(log_f1 == log_f2):
            # Calculate MSE on log-transformed PSD values
            mse_log = ((log_Pxx2 - log_Pxx1)**2).mean('frequency')
        else: print('frequencies not the same')

        test_grid_psd_members[i] = mse_log

    mean_grid_mse = (test_grid_psd_members.mean(axis=0))
    mean_grid_mse = xr.DataArray(mean_grid_mse, 
                                 coords={'lat':test_grid_psd.lat, 
                                         'lon':test_grid_psd.lon})
    return mean_grid_mse  


def get_ensemble_variance(data, stat):
    
    if stat == 'std':
        variances = find_var_name(data).std('time')
    if stat == 'var':
        variances = find_var_name(data).var('time')
    if stat == 'mean':
        variances = find_var_name(data).mean('time')
    
    return variances

def plot_regional_variance_stats(T_LENS_regional, T_InVERT_regional, 
                                 color1, color2, stat, path,
                                 name1, name2,
                                 tickfontsize = 14, axislabelfontsize=16,
                                 titlefontsize=18, legendfontsize=14):

    vs = []; vs_invert = []; pvals = np.zeros((46))
    
    if stat == 'std':
        symbol = '{\sigma}'
        stat_long_name = 'standard deviation'
    if stat =='var':
        symbol = '{\sigma^{2}}'
        stat_long_name = 'variance'
    if stat == 'mean':
        symbol = '{\mu}'
        stat_long_name = 'mean'

    for region in range(46):
        v = get_ensemble_variance(T_LENS_regional.sel(mask=region), stat)
        v_invert = get_ensemble_variance(T_InVERT_regional.sel(mask=region), stat)

        vs.append(v); vs_invert.append(v_invert)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.errorbar(np.arange(0,46), np.mean(vs, axis=1), yerr=np.std(vs, axis=1), 
                fmt='o', markersize=6.5, capsize=4, color=color1, 
                label=name1 + '$\mu_'+symbol+'$ +/- 1 $\sigma_'+symbol+'$')

    ax.errorbar(np.arange(0.2,46.2), np.mean(vs_invert, axis=1), yerr=np.std(vs_invert, axis=1), 
                fmt='o', markersize=6.5, capsize=4, color=color2, 
                label=name2 + '$\mu_'+symbol+'$ +/- 1 $\sigma_'+symbol+'$')

    ax.set_xlabel('AR6 region', fontsize=axislabelfontsize);
    if stat == 'std':
        ax.set_ylabel(r'ensemble mean $\sigma$ [K]', fontsize=axislabelfontsize);
    if stat == 'var':
        ax.set_ylabel(r'ensemble mean $\sigma^{2}$ [K$^{2}$]', fontsize=axislabelfontsize)
    if stat == 'mean':
        ax.set_ylabel(r'ensemble mean $\mu$ [K]', fontsize=axislabelfontsize)
        
    ax.set_title('Ensemble mean '+stat_long_name+'s of regional mean T anomalies', fontsize=titlefontsize);
    ax.legend(fontsize=axislabelfontsize);
    
    yticks = [0,1,2,3,4,5]
    ax.set_yticks(yticks, labels=yticks, fontsize=tickfontsize)
    
    xticks = [int(n) for n in list(range(0, 46, 5))]
    ax.set_xticks(xticks, labels=xticks, fontsize=tickfontsize)
    
    ax.set_xlim(-0.5, 46.5)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('dimgrey'); ax.spines['left'].set_color('dimgrey')
    
    if stat == 'std': ax.set_ylim(0, 2.5)
    if stat == 'var': ax.set_ylim(0, 5.5)       
    if stat == 'mean':
        ax.set_ylim(-0.2,0.2)
        ax.axhline(0, color='gray', linewidth=0.5)
    
    fig.savefig(path + 'Ensemble mean ' + stat_long_name + 's of regional mean T anomalies.pdf')
    fig.savefig(path + 'Ensemble mean ' + stat_long_name + 's of regional mean T anomalies.png')
    
    return pvals

def plot_regional_eft_stats(T_LENS_regional, T_InVERT_regional, path, 
                            color1, color2, name1, name2,
                            tickfontsize = 14, axislabelfontsize=16,
                            titlefontsize=18, legendfontsize=14):
    
    regional = find_var_name(T_LENS_regional)
    invert_regional = find_var_name(T_InVERT_regional)

    means_lens = {}; stds_lens = []
    means_invert = {}; stds_invert = []

    for region in range(46):

        mean_eft_LENS, std_eft_LENS = calc_eft_stats(regional.sel(mask=region))
        means_lens[region] = mean_eft_LENS; stds_lens.append(std_eft_LENS)

        mean_eft_InVERT, std_eft_InVERT = calc_eft_stats(invert_regional.sel(mask=region))
        means_invert[region] = mean_eft_InVERT; stds_invert.append(std_eft_InVERT)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.errorbar(np.arange(0,46), [means_lens[key] for key in np.arange(46)], yerr=stds_lens, 
                 fmt='o', markersize=6.5, capsize=4, color=color1, label='LENS2 $\mu$ +/- 1 $\sigma$')

    ax.errorbar(np.arange(0.2,46.2), [means_invert[key] for key in np.arange(46)], yerr=stds_invert, 
                 fmt='o', markersize=6.5, capsize=4, color=color2, label='InVERT $\mu$ +/- 1 $\sigma$')

    ax.set_xlabel('AR6 Region', fontsize=axislabelfontsize);
    ax.set_ylabel('e-folding time [months]', fontsize=axislabelfontsize);
    ax.set_title('Ensemble mean e-folding times of autocorrelations \n of regional mean T anomalies', 
                 fontsize=titlefontsize);
    ax.legend(fontsize=legendfontsize);

    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('dimgrey'); ax.spines['left'].set_color('dimgrey')  

    yticks = [0,2,4,6,8]
    ax.set_yticks(yticks, labels=yticks, fontsize=tickfontsize)

    xticks = [int(n) for n in list(range(0, 46, 5))]
    ax.set_xticks(xticks, labels=xticks, fontsize=tickfontsize)

    ax.set_xlim(-0.5, 46.5)
    
    return means_lens, means_invert


def gridcell_map_plot(ds, cbar_label, title, find_vlims=True, vmin=None, 
                      vmax=None, cmap='RdBu', ax=None, fig=None, shrink=0.67,
                      colorbar=True, tickfontsize = 14, axislabelfontsize=16,
                      titlefontsize=18, legendfontsize=14):
    if ax == None:
        #  Create figure and axes with cartopy projection
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    
    if find_vlims==True:
        vmax = max(np.abs(ds.values.flatten())); vmin = - vmax

    # plot data_array using pcolormesh
    im = ax.pcolormesh(ds.lon, ds.lat, ds, 
                       cmap=cmap, vmin=vmin, vmax=vmax,
                       transform = ccrs.PlateCarree())
    
    if colorbar == True:
        cbar = fig.colorbar(im, ax=ax, shrink=shrink, pad=0.02)
        cbar.ax.set_ylabel(cbar_label, fontsize=axislabelfontsize)
        cbar.ax.tick_params(labelsize=tickfontsize)
    
    # Add stock image to mask ocean
    ax.stock_img()
    ax.add_feature(cfeature.OCEAN, zorder=100, edgecolor='black', 
                   facecolor='gainsboro')

    ax.set_title(title, fontsize=titlefontsize)
    
    xticks = [-180, -120, -60, 0, 60, 120, 180]
    xticklabels = [str(tick) for tick in xticks]

    yticks = [-90, -60, -30, 0, 30, 60, 90]
    yticklabels = [str(tick) for tick in yticks]
    
    return im, fig, ax, vmin, vmax, ds


def plot_gridcell_diff(D1, D2, stat, difference, find_vlims=True, 
                       vmin=None, vmax=None, ax=None, fig=None, shrink=0.67,
                       colorbar=True, im=None, cmap='RdBu', tickfontsize = 14, axislabelfontsize=16,
                          titlefontsize=18, legendfontsize=14):
    """
    Plots gridcell percent difference between two xarray datasets.
    Args:
        D1: Reference xarray dataset. D2: xarray dataset to compare.
        stat: 'variance' or 'standard deviation'
        difference: string to specify which difference, 'percent' or 'absolute'
    """
    if difference == 'percent':
        # Calculate percent difference
        diff = ((D2 - D1) / D1) * 100
        
        if stat == 'standard deviation':
            cbar_label = r'($\sigma_{InVERT}-\sigma_{LENS2}$)' + ' $\sigma_{LENS} * 100$ [%]'
        if stat == 'autocorrelation e-folding time':
            cbar_label = r'$(\tau_{InVERT} - \tau_{LENS2})\ / \tau_{LENS2} * 100$ [%]'
        
    if difference == 'absolute':
        # Calculate absolute difference
        diff = D2 - D1
        
        if stat == 'standard deviation':
            cbar_label = r'$\sigma_{InVERT} - \sigma_{LENS2}$ [K]'
        if stat == 'autocorrelation e-folding time':
            cbar_label = r'$\tau_{InVERT} - \tau_{LENS2}$ [months]'
        
    title = 'Gridcell ' + difference + ' difference: \n ensemble mean ' + stat
    
    if find_vlims == True:
        im, fig, ax, vmin, vmax, _ = gridcell_map_plot(diff, cbar_label, title, ax=ax, fig=fig, shrink=shrink,
                                                colorbar=colorbar, cmap=cmap)
    else:
        im, fig, ax, _, _, _ = gridcell_map_plot(diff, cbar_label, title,
                                                 find_vlims=False, vmin=vmin,
                                                 vmax=vmax, ax=ax, fig=fig, shrink=shrink,
                                                 colorbar=colorbar, cmap=cmap)
    return vmin, vmax, im, diff


def plot_regional_diff_map(lens_regional, invert_regional, ds_gridded, 
                           difference, stat, vmin, vmax, ax=None, fig=None,
                           subplot_label=None, shrink=0.67, colorbar=True, 
                           im=None, cmap='RdBu', tickfontsize = 14, axislabelfontsize=16,
                          titlefontsize=18, legendfontsize=14):  
    """
    Plots percent difference in ensemble means of regional mean of a quantity ('stat') 
        between two regionally-aggregated mean T anomaly datasets
    ds_gridded is an original gridded dataset with dims lat lon (e.g. not aggregated into regional means)
    """
    import regionmask
    
    # Calculate stat over time for each ensemble member then ensemble mean
    if stat == 'std':
        lens_regional = find_var_name(lens_regional).std(dim='time').mean('ensemble')
        invert_regional = find_var_name(invert_regional).std(dim='time').mean('ensemble')

        ### Calculate ensemble mean stat in each region and save in dictionary
        reg_means_lens = {}; reg_means_invert = {}

        for region in range(46):
            reg_means_lens[region] = lens_regional.sel(mask=region).values.tolist()
            reg_means_invert[region] = invert_regional.sel(mask=region).values.tolist()
            
    if stat == 'eft': # input is already a dict of regional means
        reg_means_lens = lens_regional
        reg_means_invert = invert_regional
        
    regional_pct_diff = {}; regional_abs_diff = {}

    for region in range(46):
    
        # Calculate percent difference in ensemble mean stat in each region and add to dictionary
        pct_diff = ((reg_means_invert[region] - reg_means_lens[region]) / reg_means_lens[region]) * 100  
        regional_pct_diff[region] = pct_diff
        # Calculate absolute difference and add to dictionary
        regional_abs_diff[region] = reg_means_invert[region] - reg_means_lens[region]

    if difference == 'percent':
        diff_dict = regional_pct_diff
        values = regional_pct_diff.values()
    if difference == 'absolute':
        diff_dict = regional_abs_diff
        values = regional_abs_diff.values()

    # Get AR6 land regions
    regions = regionmask.defined_regions.ar6.land

    # Create a data array with regional means filled in
    # Use ds_gridded coordinates (an original gridded dataset, not regional) to create the DataArray
    data_array = xr.DataArray(
        np.nan,  # Initialize with NaN values
        coords=[ds_gridded.coords["lat"], ds_gridded.coords["lon"]],  
        dims=["lat", "lon"])

    for region_name, region_diff in diff_dict.items():
        region_mask = regions.mask(ds_gridded, wrap_lon=True) == regions.map_keys(region_name)
        data_array = data_array.where(~region_mask, region_diff)
    
    if stat == 'std':
        stat_long_name = 'standard deviation'
        if difference == 'percent':
            cbar_label = r"$(\sigma_{InVERT}-\sigma_{LENS2})\  / \  \sigma_{LENS2} * 100$ [%]"
        if difference == 'absolute':
            cbar_label = '$\sigma_{InVERT} - \sigma_{LENS2}$ [K]'
    
    if stat == 'eft':
        stat_long_name = 'autocorrelation e-folding time'
        if difference == 'percent':
            cbar_label = r'$(\tau_{InVERT} - \tau_{LENS2})\ / \ \tau_{LENS2} * 100$ [%]'
    
        if difference == 'absolute':
            cbar_label = r'$\tau_{InVERT} - \tau_{LENS2}$ [months]'
    
    title = 'AR6 region ' + difference + " difference: \n ensemble mean " + \
            stat_long_name 

    im, fig, ax, vmin, vmax, _ = gridcell_map_plot(data_array, cbar_label, title, find_vlims=False,
                                                vmin=vmin, vmax=vmax, ax=ax, fig=fig, shrink=shrink,
                                                colorbar=colorbar, cmap=cmap)

    # Plot AR6 region boundaries with custom edgecolor and linewidth
    regions.plot(ax=ax,add_ocean=True,
                 ocean_kws={'facecolor': 'lightgray'},
                 label='number', text_kws={'visible':False},
                 line_kws={'lw':1}) # linewidth of region boundaries
    
    return data_array


def calc_gridcell_psd(da, fs=12, nperseg=128):

    # Pre-compute frequency values (since they're the same for all grid cells)
    f, _ = welch(da.isel(ensemble=0, lat=0, lon=0), fs=12, nperseg=128)
    freq_size = f.size  # Get the size of the frequency dimension

    # Create an empty DataArray to store PSD values for all ensemble members
    psd_all_members = xr.DataArray(
        np.zeros((da.lat.size, da.lon.size, da.ensemble.size, f.size)),
        dims=['lat', 'lon', 'ensemble', 'frequency'],
        coords={'lat': da.lat, 'lon': da.lon, 'ensemble': da.ensemble, 'frequency': f})

    # Loop through ensemble members and calculate PSD
    for i in range(da.ensemble.size):
        da_single_member = da.isel(ensemble=i)

        # Apply welch_psd using apply_ufunc with correct core dimensions
        psd_da = xr.apply_ufunc(
            welch_psd, da_single_member,
            input_core_dims=[['time']],
            output_core_dims=[['frequency']],
            exclude_dims=set(('time',)),
            dask='parallelized',
            kwargs={'fs': 12, 'nperseg': 128},
            output_sizes={'frequency': freq_size}, # Add this line
            vectorize=True)  # Enable vectorization for grid cells

        # Assign PSD values to the corresponding ensemble member in psd_all_members
        psd_all_members[:, :, i, :] = psd_da

    # convert gridcell psd (dims time, ensemble) to dataset
    ds_psd = psd_all_members.to_dataset(name='psd')
    # Calculate ensemble mean psd at each gridcell and add to dataset
    ds_psd['emean'] = ds_psd['psd'].mean('ensemble')

    # Data array of PSD values with dims lat, lon, ensemble, and frequency
    return ds_psd


def plot_MSE_by_region(regional_val_dict, cbar_label, title, ds_gridded, vmin, vmax, 
                       ax=None, fig=None, subplot_label=None, shrink=0.67, 
                       colorbar=True, im=None, cmap='Reds', tickfontsize = 14, axislabelfontsize=16,
                       titlefontsize=18, legendfontsize=14):
    '''
    Plot gridcell absolute MSE between InVERT PSD and LENS2 ensemble mean PSD
    '''
    import regionmask

    # Get AR6 land regions
    regions = regionmask.defined_regions.ar6.land
    
    # Create a data array with regional means filled in
    # Use ds_gridded coordinates (an original gridded dataset, not regional) to create the DataArray
    data_array = xr.DataArray(
        np.nan,  # Initialize with NaN values
        coords=[ds_gridded.coords["lat"], ds_gridded.coords["lon"]],  # Use ds_gridded coordinates
        dims=["lat", "lon"])

    for region_name, vals in regional_val_dict.items():
        region_mask = regions.mask(ds_gridded, wrap_lon=True) == regions.map_keys(region_name)
        data_array = data_array.where(~region_mask, np.mean(vals))

    im, fig, ax, vmin, vmax, _ = gridcell_map_plot(data_array, cbar_label, title, find_vlims=False,
                                                vmin=vmin, vmax=vmax, cmap=cmap, ax=ax, fig=fig, 
                                                shrink=shrink, colorbar=colorbar)

    # Plot AR6 region boundaries with custom edgecolor and linewidth
    regions.plot(ax=ax, add_ocean=True,
                 ocean_kws={'facecolor': 'lightgray'},
                 label='number', text_kws={'visible':False},
                 line_kws={'lw':1}) # linewidth of region boundaries

    return data_array


def calc_efold_time_dataset(ds):
    """
    Calculates e-folding time for each grid cell and ensemble member in an xarray dataset.
    Args:
        ds: xarray dataset with dimensions 'time', 'lat', 'lon', and 'ensemble'.
    Returns:
        xarray DataArray with e-folding times for each grid cell and ensemble member.
    """
    # Create an empty DataArray to store the e-folding times
    efold_times = xr.DataArray(
        np.zeros((ds.lat.size, ds.lon.size, ds.ensemble.size)),
        dims=['lat', 'lon', 'ensemble'],
        coords={'lat': ds.lat, 'lon': ds.lon, 'ensemble': ds.ensemble})

    # Loop through lat, lon, and ensemble dimensions
    for lat_idx in range(ds.lat.size):
        for lon_idx in range(ds.lon.size):
            for ens_idx in range(ds.ensemble.size):
                # Extract the time series for the current grid cell and ensemble member
                timeseries = ds.isel(lat=lat_idx, lon=lon_idx, ensemble=ens_idx)

                # Calculate the e-folding time and store it in the DataArray
                efold_times.loc[dict(lat=ds.lat[lat_idx], lon=ds.lon[lon_idx], 
                        ensemble=ds.ensemble[ens_idx])] = calc_efold_time(timeseries.values)
    return efold_times
    
    
def plot_var_coeffs(lagged_coeffs_dataset, months_per_row, savepath, 
                    xaxis_lag=12, output_modes_per_row=4):

    from matplotlib.colors import SymLogNorm, Normalize

    nrows_coeffs = len(months_per_row) # Number of rows for coefficient plots
    nplots_per_row = output_modes_per_row # Number of plots per row

    # Calculate total rows and columns
    total_rows = nrows_coeffs; total_cols = nplots_per_row

    # Adjust fig height based on no. of coefficient rows
    fig = plt.figure(figsize=(18, 3.2*total_rows)) 
    gs = gridspec.GridSpec(total_rows, total_cols, figure=fig, 
                           wspace=0.2, hspace=0.6, 
                           height_ratios = [2] * total_rows)

    # Create axes for coefficient plots and EOF plots
    axes_coeffs = [fig.add_subplot(gs[r, c]) for r in range(nrows_coeffs) for c in range(total_cols)]

    # create axes for colorbars (# Left, bottom, width, height)
    # Adjust colorbar position based on the number of coefficient rows
    cbar_coeffs_ax = fig.add_axes([0.92, 0.15, 0.012, 0.7]) 

    # Define subplot labels
    subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', 
                      '(i)', '(j)', '(k)', '(l)', '(m)', '(n)', '(o)', '(p)', 
                      '(q)', '(r)', '(s)', '(t)', '(u)', '(v)', '(w)', '(x)',
                      '(y)', '(z)'] # Extend labels

    # Select the first `output_modes_per_row` output modes for the coefficient plots
    selected_output_modes_coeffs = lagged_coeffs_dataset.sel(output_mode=slice(0, output_modes_per_row - 1))

    coeffs = selected_output_modes_coeffs['lagged_coefficients']
    vmax_coeffs = np.abs(coeffs).max().values; vmin_coeffs = -vmax_coeffs
    
    normalized_coeffs = coeffs / vmax_coeffs

    # Iterate through each row of coefficient plots
    for r, months in enumerate(months_per_row):
        # Iterate through each output mode and create a pcolor plot
        
        for c, output_mode in enumerate(selected_output_modes_coeffs.output_mode.values):

            # Select subplot axis (corresponds to row and column)
            ax = axes_coeffs[r * total_cols + c]

            coeff_data = normalized_coeffs.sel(output_mode=output_mode, 
                                               month=months)
            month_label = str(months)

            cmap_coeffs = 'seismic'
            norm_coeffs = SymLogNorm(linthresh=0.1, linscale=0.03, vmin=-1, vmax=1) 

            contour = ax.pcolor(-lagged_coeffs_dataset['lag'], lagged_coeffs_dataset['input_mode'],
                                coeff_data.T.values, 
                                cmap=cmap_coeffs, norm=norm_coeffs)

            # Add subplot label
            ax.text(-0.08, 1.15, subplot_labels[r * total_cols + c], transform=ax.transAxes,
                    fontsize=16, fontweight='bold', va='top', ha='left', zorder=10)

            # Set the color of top and right spines to 'none'
            ax.spines['top'].set_color('none'); ax.spines['right'].set_color('none')

            # Explicitly set spine visibility
            ax.spines['left'].set_visible(True); ax.spines['bottom'].set_visible(True)

            # Set spine color and width for bottom and left spines
            ax.spines['bottom'].set_color('darkgrey'); ax.spines['bottom'].set_linewidth(1)
            ax.spines['left'].set_color('darkgrey'); ax.spines['left'].set_linewidth(1)

            # Ensure ticks are visible
            ax.tick_params(axis='x', bottom=True, labelbottom=True)
            ax.tick_params(axis='y', left=True, labelleft=True)

            # Set labels -- Explicitly set labels and ensure they are drawn
            ax.xaxis.set_visible(True) # Ensure x-axis is visible

            if c == 0: # Only set ylabel for the first column
                ax.set_ylabel('predictor mode \n month '+str(months), fontsize=axislabelfontsize)
                ax.yaxis.set_visible(True) # Ensure y-axis is visible
            else: ax.set_ylabel('')

            if r == 0:
                ax.set_title('mode ' + str(output_mode+1), fontsize=titlefontsize)

            ax.set_xlim(-12.5, -0.5); ax.set_ylim(0.5, 4 + 0.45)
            ax.tick_params(axis='both', which='major', labelsize=tickfontsize)
            ax.set_yticks([1,2,3,4]); ax.set_xticks([-12, -10, -8, -6, -4, -2])

            if r == 1:            
                ax.set_xlabel('lag [months]', fontsize=axislabelfontsize)
            else: ax.set_xlabel('')

    ######## Add colorbars
    cbar_coeffs = fig.colorbar(contour, cax=cbar_coeffs_ax, # Use the created axis
                               orientation='vertical')
    cbar_coeffs.set_label(label='normalized coefficient', size=axislabelfontsize)
    cbar_coeffs.ax.tick_params(labelsize=tickfontsize)

    # Define custom tick locations
    cbarticks = [-1, -.9, -.8, -.7, -.6, -.5, -.4, -.3, -.2, -.1, 0,
             .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
    cbarticklabels = [-1, -0.5, 0, 0.5, 1]
    cbar_coeffs.set_ticks(cbarticks)

    # Create a list of tick labels with empty strings for unwanted ticks
    labels = [str(val) if val in cbarticklabels else '' for val in cbarticks]
    cbar_coeffs.set_ticklabels(labels, fontsize=tickfontsize)

    plt.suptitle('VAR coefficients', fontsize=titlefontsize+4, y=0.98)
    
        # Adjust tight_layout to make space for colorbars and move everything to the left
    plt.tight_layout()#rect=[0, 0, 0.9, 1])

    plt.savefig(savepath + 'VAR_coeffs_months_' + str(months_per_row)+ '.png')
    plt.savefig(savepath + 'VAR_coeffs_months_' + str(months_per_row)+ '.pdf')
    
    
def convert_lon(ds, lon_name):
    '''
    Converts a 0-360 degree longitude grid dataset to a -180 to 180 dataset
    lon_name is input string either 'longitude' or 'lon'
    '''
    ds_copy = ds.copy()
    ds_copy.coords[lon_name] = (ds_copy.coords[lon_name] + 180) % 360 - 180
    ds_copy = ds_copy.sortby(ds_copy[lon_name])
    return ds_copy

def emulate_pcs(training_pcs, monthly_var_models, n_training_members, 
                optimal_lag, n_samples, n_steps, nmodes, M, savepath):
    '''
    Use monthly VAR model components to generate synthetic PC time series
    
    Parameters:
        - training_pcs (xr.DataArray): Time-sorted training PCs. Used for initial conditions.
        - monthly_var_models (dict): Dictionary containing trained monthly VAR model components 
                                     (intercepts and lagged coefficients) for training PCs
        - n_training_members (int): The number of original training data ensemble members' PCs 
                                    used to train the VAR model.
        - optimal_lag (int): The number of timesteps (months) used as lag.
        - n_samples (int): The number of emulated ensemble members to produce.
        - n_steps (int): The desired length of each emulated ensemble member after truncation (months).
        - nmodes (int): The number of modes included.
        - M (int): The number of extra months of PCs to simulate for burn-in
        - path (str): Path to save results.
    '''
    ## Get the required number of past values for VAR initialization
    lag_order = optimal_lag

    ## Generate more steps than desired length to allow for burn-in and to ensure
    ## truncated time series can start in January
    ## Generate at least lag_order + M (burn-in, 120 months) + 12 (to ensure a January start) steps
    total_steps_to_generate = n_steps + lag_order + M + 12

    ## Prepare training PCs for initial conditions selection
    training_pcs_values = training_pcs.pcs.values

    ## Save month values from training PCs sorted by time
    training_months_values = training_pcs.month.values

    ## Select any time index besides the last lag_order indices for initial conditions
    len_pcs = len(training_pcs.time)
    valid_start_indices = np.arange(len_pcs - lag_order)
    
    ## Avoid indices where the initial condition sequence spans across different ensemble members 
    ## in the training data: record indices in stacked training PCs where a new ensemble member starts
    ## This assumes training_pcs_sorted was created by stacking ensemble members of equal length
    len_training_member = len(training_pcs.time) // n_training_members 
    indices_btwn_ens_members = [len_training_member * i for i in range(1, n_training_members)]
    
    ## Create a set of 'forbidden indices' to avoid selecting as a start index for initial conditions
    ## in the stacked training_pcs; aka the last 'optimal_lag' indices from each training ensemble member
    forbidden_indices = []
    for idx in indices_btwn_ens_members:
        ## Prevent choosing an initial condition index within the last (lag_order - 1) indices
        ## of each ensemble member, so that initial conditions stay within the same ensemble member
        for i in range(-lag_order + 1, 1): # Use 1 here to include the index itself
            forbidden_indices.append(idx + i)
    
    emulated_samples = [] ## List to store time series of emulated PCs
    
    ## Remove the forbidden indices from valid_start_indices
    valid_start_indices = [idx for idx in valid_start_indices if idx not in forbidden_indices]

    for i in range(n_samples): ## Start emulating a new ensemble member of PCs
        
        ## Randomly select a starting point for initial conditions from valid indices
        start_index = np.random.choice(valid_start_indices)

        ## Select last `lag_order` time steps of training PCs as initial conditions, for each mode
        initial_conditions = training_pcs_values[start_index : start_index + lag_order, :]
        
        ## Initialize the synthetic PC time series for this sample as a list (start w initial conditions)
        ## synthetic_series is identical to initial_conditions in values, but in list form
        synthetic_series = [initial_conditions[j, :] for j in range(lag_order)]
            
        ## Determine the month of the last initial condition: first, find its index (as a 0-index)
        last_initial_condition_month_index = start_index + lag_order - 1
        
        ## use last initial condition month index to find its corresponding month in training_months_values
        current_month = training_months_values[last_initial_condition_month_index] 
        
        ## Generate the rest of the synthetic PC time series (len = total_steps_to_generate - lag_order)
        for step in range(total_steps_to_generate - lag_order):

            ## Determine the month being predicted
            month_to_predict = (current_month % 12) + 1

            ## Get the trained model components for the month to predict
            model_components = monthly_var_models[month_to_predict]
            intercept = model_components['intercept']
            lagged_coeffs = model_components['lagged_coeffs'] # Shape: (lag, input_mode, output_mode)
            residuals = model_components['residuals'] # Shape: ( (len(training_pcs) // 12 -1), nmodes )
            
            ## Get the input features: the last 'lag_order' steps from the current synthetic series
                ## at first, this will be the randomly chosen 12 initial conditions for each mode
                ## As new synthetic PCs are added, this will draw the last lag_order synthetic PCs
            input_features = np.array(synthetic_series[-lag_order:]) ## Shape: (lag, mode)

            ## Reshape and flatten the input features to match the training format
            input_features_flat = input_features.flatten() ## Shape: (lag * mode,)

            ## Add the intercept term # Shape: (lag * mode + 1,)
            input_features_with_intercept = np.hstack([np.ones(1), input_features_flat]) 

            ## Reshape lagged_coeffs for matrix multiplication: (lag*input_mode, output_mode)
            lagged_coeffs_reshaped = lagged_coeffs.reshape((lag_order * nmodes, nmodes))

            ## Predict the next step (PCs for the month_to_predict)   # Shape: (nmodes,)
            predicted_pcs = intercept + np.dot(input_features_flat, lagged_coeffs_reshaped) 
            
            ## Randomly sample a residual vector from the residuals for this month
            ## Start by getting a random index from the shape of the residuals
            random_residual_index = np.random.choice(residuals.shape[0])

            ## Grab a residual vector from that index # Shape: (nmodes,)
            sampled_residual = residuals[random_residual_index, :]

            ## Add the sampled residual to the predicted PCs
            predicted_pcs_with_residual = predicted_pcs + sampled_residual

            ## Append the predicted PCs (with residual) to the synthetic series
            synthetic_series.append(predicted_pcs_with_residual)

            ## Update the current month for the next step
            current_month = month_to_predict           

        ### After generating 'total_steps_to_generate' steps of new PCs, find the first January and truncate

        ## Convert list to array for easier slicing
        synthetic_series = np.array(synthetic_series) 
        
        ## Determine the month sequence for the synthetic PC time series just generated
        ## The month of the first generated step (at index lag_order) is (month of last initial condition % 12) + 1
        start_month_generated = (training_months_values[start_index + lag_order - 1] % 12) + 1
        
        ## Create array of the full sequence of months, starting at start_month_generated
        full_months_sequence = np.tile(np.arange(1, 13), 
                                       total_steps_to_generate // 12 + 2)[start_month_generated - \
                                       1 : start_month_generated - 1 + total_steps_to_generate - lag_order]   
        
        ## Prepend the months of the initial conditions
        initial_condition_months = training_months_values[start_index : start_index + lag_order]
        full_months_sequence = np.concatenate((initial_condition_months, full_months_sequence))
        
        ## Find the index of the first January *after* the initial conditions AND burn-in period
        ## Search within the generated part of the series (from index lag_order onwards)
        ## Find the index of the first January in the full generated sequence
        first_january_index = -1

        for idx in range(lag_order + M, total_steps_to_generate): ## Start searching after burn-in
            if full_months_sequence[idx] == 1: ## January
                first_january_index = idx
                break

        ## if an index for the first january is found after burn-in and ICs:
        if first_january_index != -1: 
            
            ## Truncate the series: remove everything before the first January after burn-in and ICs
            truncated_sample = synthetic_series[first_january_index : first_january_index + n_steps, :]
            ## Adjust months
            months = full_months_sequence[first_january_index : first_january_index + n_steps]
            
        ## Store the truncated sample and its corresponding months
        emulated_samples.append(truncated_sample)
        
    ## Convert list of synthetic PC time series to NumPy array
    ## Shape: (num_samples, n_steps, num_modes)
    emulated_samples = np.array(emulated_samples)   
    
    ## Convert the list of month arrays into a single array
    months = np.array(months) ## Assuming all truncated samples have the same month sequence
    
    ## Convert emulated PCs to an xarray DataArray
    emulated_da = xr.DataArray(emulated_samples,
                                dims=("ensemble", "time", "mode"),
                                coords={"ensemble": np.arange(n_samples),
                                        "time": np.arange(n_steps),
                                        "mode": np.arange(nmodes)})
    
    ## Assign month coordinate to emulated PCs DataArray
    emulated_da = emulated_da.assign_coords({'month': ('time', months)})

    ## Convert to Dataset with variable name 'pcs'
    new_pcs_final = emulated_da.to_dataset(name='pcs')

    ## Assign the month coordinate again, just to be sure, although it should be carried over
    new_pcs_final = new_pcs_final.assign_coords({'month': ('time', months)})
    
    new_pcs_final.to_netcdf(savepath + 'InVERT_PCs.nc')
    
    return new_pcs_final

def plot_local_monthly_T_stds(LENS_emean_monthly_stds, InVERT_emean_monthly_stds,
                              LENS_estd_monthly_stds, InVERT_estd_monthly_stds,
                              locations_lat_lon, savepath, ylim, save_name, 
                              name1, name2,
                              markersize=5, capsize=4, elinewidth=1.5, 
                              markeredgewidth=1.5,):
    """
    Plots monthly T anomaly standard deviations for 4 locations in a 2x2 grid 
     - locations_lat_lon (list of tuples): A list of tuples, each containing (lat, lon, location_name)
     - ylim (float): The upper limit for the y-axis of the plots.
    """
    n_locations = len(locations_lat_lon)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10), squeeze=False) # 2x2 grid
    axes = axes.flatten() # Flatten axes array for easy iteration

    ## Add subplot labels
    subplot_labels = ['(a)', '(b)', '(c)', '(d)']

    for i, (lat, lon, location) in enumerate(locations_lat_lon):
        ax = axes[i] # Get current subplot axis

        invert_emean_local = InVERT_emean_monthly_stds.sel(lat=lat, lon=lon, method='nearest')
        lens_emean_local = LENS_emean_monthly_stds.sel(lat=lat, lon=lon, method='nearest')
        
        invert_estd_local = InVERT_estd_monthly_stds.sel(lat=lat, lon=lon, method='nearest')
        lens_estd_local = LENS_estd_monthly_stds.sel(lat=lat, lon=lon, method='nearest')

        lat_selected = np.round(invert_emean_local.lat.values, 2)
        lon_selected = np.round(invert_emean_local.lon.values, 2)

        print(f"{location}: {invert_emean_local.lat.values}, {invert_emean_local.lon.values}")

        if lat_selected >= 0: NS_hemisphere = 'N'
        else: NS_hemisphere = 'S'

        if lon_selected >= 0: EW_hemisphere = 'E'
        else: EW_hemisphere = 'W'

        # Plot error bars with desired color (LENS)
        ax.errorbar(np.arange(1, 13), lens_emean_local, yerr=lens_estd_local,
                    fmt='none', markersize=markersize, capsize=capsize, color=color1, 
                    label= name1, ecolor=color1, elinewidth=elinewidth,
                    markeredgewidth=markeredgewidth)  
        # Plot markers separately with unfilled style
        ax.plot(np.arange(1, 13), lens_emean_local, 'o', markersize=markersize, color=color1,  
                 markerfacecolor=color1, markeredgecolor=color1, label=name1)  

        # Plot error bars with desired color (InVERT)
        ax.errorbar(np.arange(1, 13), invert_emean_local, yerr=invert_estd_local,
                    fmt='none', markersize=markersize, capsize=capsize, color=color2, 
                    label= name2, ecolor=color2, elinewidth=elinewidth,
                    markeredgewidth=markeredgewidth)  
        # Plot markers separately with unfilled style
        ax.plot(np.arange(1, 13), invert_emean_local, 'o', markersize=markersize, color=color2,  
                 markerfacecolor=color2, markeredgecolor=color2, label=name2)  

        ax.set_ylim(0,ylim[i]); ax.set_xlim(0.5,12.5)
        ax.set_xlabel('month', fontsize=axislabelfontsize)
        ax.set_xticks([2,4,6,8,10,12])
        ax.set_xticklabels(['2', '4', '6', '8', '10', '12'], fontsize=tickfontsize)
        
        if ylim[i] == 2:
            ax.set_yticks([0,1,2])
            ax.set_yticklabels(['0', '1', '2'])
        if ylim[i] == 3:
            ax.set_yticks([0,1,2,3])
            ax.set_yticklabels(['0', '1', '2', '3'])

        ax.set_title(location, fontsize=titlefontsize)
        
        if i == 0: # Add legend only to top-left subplot
            ax.legend(fontsize=legendfontsize)

        ax.set_ylabel('standard deviation [K]', fontsize=axislabelfontsize)
        ax.text(-0.05, 1.1, subplot_labels[i], transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top', ha='left', zorder=10)
    plt.tight_layout()

    fig.savefig(savepath + save_name + '.png')
    fig.savefig(savepath + save_name + '.pdf')
    
def calc_gridcell_monthly_autocorrs(data_array):
    """
    Calculates month-to-month autocorrelation for each grid cell and month pair
    - data_array (xr.DataArray): input DataArray with dims(ensemble, time, lat, lon)
                                 and 'month' coordinate. Assumes 'time' is integer or float.
    Returns
    - xr.DataArray: DataArray with dims (month_pair, lat, lon) containing the autocorrelation coefficients.
                    month_pair 1 corresponds to Jan-Feb, 2 to Feb-Mar, ..., 12 to Dec-Jan.
    """
    monthly_autocorrelations = []
    month_pairs = []

    ## Group by ensemble to process each ensemble member separately
    ensemble_groups = data_array.groupby('ensemble')

    for month in range(1, 13):
        next_month = month + 1 if month < 12 else 1
        month_pairs.append(f'{month}-{next_month}')

        ## List to store correlations for the current month pair across ensembles
        ensemble_correlations = []

        for ens_id, ens_data in ensemble_groups:
            # Select data for the current and next month within this ensemble member
            current_month_data = ens_data.sel(time=(ens_data['month'] == month))
            next_month_data = ens_data.sel(time=(ens_data['month'] == next_month))

            # find time indices where month N is followed by month N+1
            current_month_times = current_month_data['time'].values
            next_month_times = next_month_data['time'].values

            aligned_current_times = []; aligned_next_times = []

            # Find pairs of consecutive time indices where the month transitions from 'month' to 'next_month'
            original_ensemble_time_indices = ens_data['time'].values
            original_ensemble_months = ens_data['month'].values

            for i in range(len(original_ensemble_time_indices) - 1):
                if original_ensemble_months[i] == month and original_ensemble_months[i+1] == next_month:
                    aligned_current_times.append(original_ensemble_time_indices[i])
                    aligned_next_times.append(original_ensemble_time_indices[i+1])

            if len(aligned_current_times) > 1: # Need at least two pairs to calculate correlation
                current_aligned_data = ens_data.sel(time=aligned_current_times).squeeze('ensemble')
                next_aligned_data = ens_data.sel(time=aligned_next_times).squeeze('ensemble')

                # Calculate correlation across the time dimension for each lat/lon for this ensemble
                # Reshape to (lat, lon, time_pairs)
                current_reshaped = current_aligned_data.transpose('lat', 'lon', 'time')
                next_reshaped = next_aligned_data.transpose('lat', 'lon', 'time')

                # Calculate correlation along the time dimension
                correlations_ensemble = xr.DataArray(
                    np.full((data_array['lat'].size, data_array['lon'].size), np.nan),
                    coords={'lat': data_array['lat'], 'lon': data_array['lon']},
                    dims=['lat', 'lon'])

                for i in range(data_array['lat'].size):
                    for j in range(data_array['lon'].size):
                        cur_data = current_reshaped[i, j, :].values
                        next_data = next_reshaped[i, j, :].values

                        valid_indices = ~np.isnan(cur_data) & ~np.isnan(next_data)

                        if np.sum(valid_indices) > 1:
                             correlations_ensemble[i, j] = np.corrcoef(cur_data[valid_indices], next_data[valid_indices])[0, 1]
                ensemble_correlations.append(correlations_ensemble)
                
            else:
                # Append NaN dataarray if not enough data points for this ensemble
                nan_dataarray = xr.DataArray(
                    np.full((data_array['lat'].size, data_array['lon'].size), np.nan),
                    coords={'lat': data_array['lat'], 'lon': data_array['lon']},
                    dims=['lat', 'lon'])
                ensemble_correlations.append(nan_dataarray)

        # Average the correlations across ensemble members for the current month pair
        if ensemble_correlations: # Check if the list is not empty
            mean_correlations = xr.concat(ensemble_correlations, dim='ensemble').mean('ensemble')
            monthly_autocorrelations.append(mean_correlations)
        else: # Append NaN dataarray if no valid ensemble correlations were calculated
            nan_dataarray = xr.DataArray(
                np.full((data_array['lat'].size, data_array['lon'].size), np.nan),
                coords={'lat': data_array['lat'], 'lon': data_array['lon']},
                dims=['lat', 'lon'])
            monthly_autocorrelations.append(nan_dataarray)

    # Concatenate results into a single DataArray
    autocorr_da = xr.concat(monthly_autocorrelations, dim='month_pair')
    autocorr_da = autocorr_da.assign_coords(month_pair=[f'{m}-{m+1 if m<12 else 1}' for m in range(1,13)])

    return autocorr_da