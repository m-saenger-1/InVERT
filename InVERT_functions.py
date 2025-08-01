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
import random

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


def make_summary_strings(VAR_result, nmodes):
    '''
    Separate VAR model results summary into strings for each equation
    '''
    # Capture the results.summary() output as a string
    with io.StringIO() as buf, redirect_stdout(buf):
        print(VAR_result.summary())
        
        summary_str = buf.getvalue()
        
    summary_strings = []

    for eq in range(nmodes):

        idx0 = summary_str.find('Results for equation y'+str(eq+1))

        if eq < (nmodes - 1):
            idx1 = summary_str.find('Results for equation y'+str(eq+2))
        else:
            idx1 = summary_str.find('Correlation matrix of residuals')       

        summary_strings.append(summary_str[idx0:idx1])

    return summary_strings  


def make_coeff_dataset(summary_strings):
    '''
    Save coefficient data from VAR model in xarray dataset for lag coefficient
         contour plot
    
    summary_strings is a list of strings containing VAR model result output
    
    y1 is the predictor mode: the PC mode whose current value is being predicted
        by the lagged values of other PCs, including itself
        
    y2 is the response variable or mode: the PC mode whose current value is being
        predicted by the lagged values of other PCs, including itself
    '''
    datasets = []

    for i, summary_str in enumerate(summary_strings):
        # Extract coefficients and intercept using regex
        coefficients = {}
        intercept_value = np.nan 
        for line in summary_str.split('\n'):
            if line.startswith('coefficient') or line.startswith('---'):
                continue

            # Match 'const' row first
            match = re.search(r'const\s+([\d\.\-]+)', line)
            if match:
                intercept_value = float(match.group(1))
                continue

            # If not 'const', match other coefficient rows
            match = re.search(r'([a-zA-Z]+\d*\.\w+)\s+([\d\.\-]+)', line)
            if match:
                var_name, coef = match.groups()
                lag_str, eq_str = var_name[1:].split('.')
                lag = int(lag_str)
                eq1 = int(eq_str[1:])
                eq2 = i + 1
                coefficients[(lag, eq1, eq2)] = float(coef)

        # Create xarray Dataset
        lags = list(set(lag for lag, _, _ in coefficients))
        eq1s = list(set(eq1 for _, eq1, _ in coefficients))
        eq2s = list(set(eq2 for _, _, eq2 in coefficients))

        coef_array = np.empty((len(lags), len(eq1s), len(eq2s)))
        coef_array[:] = np.nan

        for (lag, eq1, eq2), coef in coefficients.items():
            lag_idx = lags.index(lag)
            eq1_idx = eq1s.index(eq1)
            eq2_idx = eq2s.index(eq2)
            coef_array[lag_idx, eq1_idx, eq2_idx] = coef

        # Add intercept as a separate variable in the Dataset
        ds = xr.Dataset(
            {'Coefficient': (['lag', 'equation1', 'equation2'], coef_array),
             'intercept': (['equation2'], [intercept_value])},  # Assign intercept here
            coords={'lag': lags, 'equation1': eq1s, 'equation2': eq2s}
        )
        datasets.append(ds)

    # Concatenate datasets along 'equation2'
    datasets = xr.concat(datasets, dim='equation2')

    datasets = datasets.rename({'equation1':'y1'})
    datasets = datasets.rename({'equation2':'y2'})
    
    return datasets


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
                            tickfontsize = 14, axislabelfontsize=16,
                            titlefontsize=18, legendfontsize=14,):
    
    f, mean_psd, std_psd = calc_psd_stats(original_T)
    invert_f, invert_mean_psd, invert_std_psd = calc_psd_stats(InVERT_T)

    ax.loglog(f, mean_psd, color=color1, linestyle='--', label='LENS2 $\mu_{PSD}$')
    ax.loglog(f, mean_psd - std_psd, color=color1)
    ax.loglog(f, mean_psd + std_psd, color=color1)
    
    ax.fill_between(f, mean_psd - std_psd, mean_psd + std_psd, 
                     color=color1, alpha=0.2, label='LENS2 $\mu_{PSD}$ +/- 1 $\sigma_{PSD}$')

    ax.loglog(invert_f, invert_mean_psd, color=color2, linestyle='dotted', label='InVERT $\mu_{PSD}$')
    ax.loglog(invert_f, invert_mean_psd - invert_std_psd, color=color2)
    ax.loglog(invert_f, invert_mean_psd + invert_std_psd, color=color2)

    ax.fill_between(invert_f, invert_mean_psd - invert_std_psd, 
                               invert_mean_psd + invert_std_psd, 
                     color=color2, alpha=0.2, label='InVERT $\mu_{PSD}$  +/- 1 $\sigma_{PSD}$')
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
                label='$\mu_{'+name1+'}$'+' +/- 1$\sigma_{'+name1+'}$',  # Use fmt='none' to hide default marker
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
                label='$\mu_{'+name2+'}$'+' +/- 1$\sigma_{'+name2+'}$',  # Use fmt='none' to hide default marker
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
                          color1, color2, path,
                          tickfontsize = 14, axislabelfontsize=16,
                          titlefontsize=18, legendfontsize=14):

    fig, axes = plt.subplot_mosaic("a;b;c", figsize=(8, 16));  # Create figure and axes
    
    for n, (key, ax) in enumerate(axes.items()):
        ax.text(-0.1, 1.075, key, transform=ax.transAxes, 
                size=20, weight='bold')
        
        if n == 0:
            ## PDF
            compare_T_pdfs(T_lens_stacked.gmean, T_invert_stacked.gmean, 
                           'LENS2', 'InVERT', color1, color2, 'Probability distribution functions', ax=ax)
        if n == 1:
            ## PSD
            plot_GMST_psd_spread(ax, T_lens.gmean, 
                                 T_invert.gmean, 
                                 100, 'Power spectral density curves', color1, color2) 
        if n == 2:
            ## Autocorrelations
            compare_autocorrs_emean(T_lens.gmean, 'LENS2', T_invert.gmean, 'InVERT', 
                                    'Autocorrelations', color1, color2, ax=ax)
    fig.tight_layout(pad=1)
    plt.show()
    
    fig.savefig(path + 'Figure_2_subplots.pdf'); fig.savefig(path + 'Figure_2_subplots.png')


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
                             region, name, color1, color2,
                             tickfontsize = 14, axislabelfontsize=16,
                             titlefontsize=18, legendfontsize=14):
    ''''''
    import regionmask
    
    og = find_var_name(T_og).sel(mask=region)
    invert = find_var_name(T_invert).sel(mask=region)
    
    og_stacked = find_var_name(T_og_stacked).sel(mask=region)
    invert_stacked = find_var_name(T_invert_stacked.sel(mask=region))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5));  # Create figure and axes
    
    ## PDF
    plot_regional_T_pdfs(t1=og_stacked, t2=invert_stacked, 
                   name1='LENS2', name2='InVERT',
                   color1=color1, color2=color2, 
                   title='PDF', ax=axes[0])
    ## PSD
    plot_regional_psd_spread(axes[1], og, invert, title='PSD', 
                             color1=color1, color2=color2)  # Call plot_psd_spread() with axes[1]
    axes[1].set_title('PSD', fontsize=titlefontsize) #set title for plot_psd_spread
    
    ## Autocorrelations
    axes[2].set_title('Autocorrelations', fontsize=titlefontsize) # Set title for the third subplot
    
    plot_regional_emean_autocorrs(og, 'LENS2', invert, 'InVERT', title='Autocorrelations', ax = axes[2],
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
                                 tickfontsize = 14, axislabelfontsize=16,
                                 titlefontsize=18, legendfontsize=14):

    vs = []; vs_invert = []
    pvals = np.zeros((46))
    
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
                label='LENS2 $\mu_'+symbol+'$ +/- 1 $\sigma_'+symbol+'$')

    ax.errorbar(np.arange(0.2,46.2), np.mean(vs_invert, axis=1), yerr=np.std(vs_invert, axis=1), 
                fmt='o', markersize=6.5, capsize=4, color=color2, 
                label='InVERT $\mu_'+symbol+'$ +/- 1 $\sigma_'+symbol+'$')

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
                            color1, color2,
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


def plot_combined_coeffs_and_eofs(datasets, eofs, varFracs, path,
                                  tickfontsize = 14, axislabelfontsize=16,
                                  titlefontsize=18, legendfontsize=14):
    
    from matplotlib.colors import SymLogNorm, Normalize
    import matplotlib.gridspec as gridspec
    
    nplots = 4
    vmin_coeffs = -1; vmax_coeffs = 1
    cmap_coeffs = 'seismic'
    norm_coeffs = SymLogNorm(linthresh=0.1, linscale=0.03, vmin=vmin_coeffs, vmax=vmax_coeffs)

    vmin_eofs = -.04; vmax_eofs = .04
    cmap_eofs = 'RdBu_r'

    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(2, nplots, figure=fig, height_ratios=[1, 1.3], wspace=0.2)

    axes_coeffs = [fig.add_subplot(gs[0, i]) for i in range(nplots)]
    axes_eofs = [fig.add_subplot(gs[1, i], projection=ccrs.Robinson()) for i in range(nplots)]

    # Create axes for colorbars
    cbar_coeffs_ax = fig.add_axes([0.92, 0.57, 0.01, 0.32]) # Left, bottom, width, height
    cbar_eofs_ax = fig.add_axes([0.92, 0.20, 0.01, 0.23])  # Adjust these values to position
    
    # Define subplot labels
    subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']

    # Top row: VAR coefficients
    for i in np.arange(1, nplots + 1):
        
        ax = axes_coeffs[i-1]

        coefficient_data = (datasets['Coefficient'].sel(y2=i)) / \
                            np.max(datasets['Coefficient'].sel(y2=i))

        contour = ax.pcolor(-datasets['lag'], datasets['y1'], coefficient_data.T,
                            cmap=cmap_coeffs, norm=norm_coeffs)
        
        # Add subplot label
        ax.text(-0.08, 1.15, subplot_labels[i-1], transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top', ha='left', zorder=10)

        # Set the color of top and right spines to 'none'
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        
        # Explicitly set spine visibility
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        
        # Set spine color and width for bottom and left spines
        ax.spines['bottom'].set_color('darkgrey')
        ax.spines['bottom'].set_linewidth(1)
        ax.spines['left'].set_color('darkgrey')
        ax.spines['left'].set_linewidth(1)

        # Ensure ticks are visible
        ax.tick_params(axis='x', bottom=True, labelbottom=True)
        ax.tick_params(axis='y', left=True, labelleft=True)

        # Set labels -- Explicitly set labels and ensure they are drawn
        ax.set_xlabel('lag [months]', fontsize=axislabelfontsize)
        ax.xaxis.set_visible(True) # Ensure x-axis is visible

        if i == 1:
            ax.set_ylabel('predictor mode', fontsize=axislabelfontsize)
            ax.yaxis.set_visible(True) # Ensure y-axis is visible

        ax.set_title('mode ' + str(i), fontsize=titlefontsize)
        ax.set_xlim(-10.5, -0.5)
        ax.set_ylim(0.5, 4 + 0.5)
        ax.tick_params(axis='both', which='major', labelsize=tickfontsize)
        ax.set_yticks([1,2,3,4])

    # Bottom row: EOF mode patterns
    for mode in range(nplots):

        ax = axes_eofs[mode]
        ds = eofs.sel(mode=mode)
        temp_data = ds.values
        lon = ds['lon']; lat = ds['lat']

        im = ax.pcolormesh(lon, lat, temp_data, transform=ccrs.PlateCarree(),
                           cmap=cmap_eofs, vmin=vmin_eofs, vmax=vmax_eofs)
        ax.add_feature(cfeature.COASTLINE)
        
        # Add subplot label (continuing from the top row)
        ax.text(-0.08, 1.15, subplot_labels[nplots + mode], transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top', ha='left', zorder=10)

        if mode == 0:
            ax.set_ylabel('\n EOF pattern \n', fontsize=axislabelfontsize)
            ax.yaxis.set_visible(True) # Ensure y-axis is visible
            ax.tick_params(axis='both', which='major', labelsize=tickfontsize)
            ax.set_yticks([])

        varFrac = np.round(varFracs.sel(mode=mode).values.tolist()*100, 1)

        ax.set_title(str(varFrac) + ' %', fontsize=titlefontsize)

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

    cbar_eofs = fig.colorbar(im, cax=cbar_eofs_ax, # Use the created axis
                             cmap='RdBu_R', orientation='vertical')
    cbar_eofs.set_label(label='T anomaly [K]', size=axislabelfontsize)

    # Define custom tick locations
    cbarticks = [-.04,-.02,  0, .02, .04]
    cbarticklabels = [-.04,-.02,  0, .02, .04]
    cbar_eofs.set_ticks(cbarticks)
    cbar_eofs.ax.tick_params(labelsize=tickfontsize)

    plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust tight_layout to make space for colorbars and move everything to the left

    plt.savefig(path + 'fig4_lag_coeff_and_eof_mode_plot.png')
    plt.savefig(path + 'fig4_lag_coeff_and_eof_mode_plot.pdf')



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
#         print(lat_idx)
        for lon_idx in range(ds.lon.size):
            for ens_idx in range(ds.ensemble.size):
                # Extract the time series for the current grid cell and ensemble member
                timeseries = ds.isel(lat=lat_idx, lon=lon_idx, ensemble=ens_idx)

                # Calculate the e-folding time and store it in the DataArray
                efold_times.loc[dict(lat=ds.lat[lat_idx], lon=ds.lon[lon_idx], 
                        ensemble=ds.ensemble[ens_idx])] = calc_efold_time(timeseries.values)
    return efold_times

def convert_longitude_range(input_ds):
    """
    Converts the longitude range in a NetCDF file from 0-360 to -180-180.

    Args:
        input_filepath (str): The path to the input NetCDF file with 0-360 longitude.
        output_filepath (str): The path to save the output NetCDF file with -180-180 longitude.
    """
    if input_ds.lon.max().values > 181:
    
        # Convert longitude from 0-360 to -180-180
        input_ds['lon'] = (input_ds['lon'] + 180) % 360 - 180

        # Sort the dataset by the new longitude values
        output_ds = input_ds.sortby('lon')
        
    else:
        print('input dataset longitude not 0-360')
        print(input_ds.lon.values)
        
    return output_ds

def plot_gridcell(example_ds, test_lat, test_lon_input, location):
    '''
    Input: lat and lon on -180/180 longitude map
    Converts to correct gridcell on 0-360 longitude map
    Output: plots on -180/180 longitude map
    '''
    # Convert the input longitude from -180/180 to 0/360 range for selection in the original dataset
    test_lon_original = (test_lon_input + 360) % 360

    # Find the nearest grid cell in your original data (assuming Tanoms_lens has 0-360 longitude)
    nearest_gridcell_original = example_ds.sel(lat=test_lat, 
                                            lon=test_lon_original, method='nearest')

    # Create a boolean mask for the selected grid cell in the original dataset
    gridcell_mask_original = (example_ds.lat == nearest_gridcell_original.lat) & \
                             (example_ds.lon == nearest_gridcell_original.lon)

    # Create a DataArray with the selected grid cell values, and NaNs elsewhere, 
    # using the original dataset structure
    highlighted_gridcell_original = example_ds.where(gridcell_mask_original)

    # Convert the highlighted grid cell dataset to -180 to 180 longitude range for plotting
    highlighted_gridcell_converted = convert_longitude_range(highlighted_gridcell_original.to_dataset(name='anoms'))['anoms']

    # Create the plot
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # Add land and ocean features
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.coastlines()
    ax.set_xlabel('longitude [º]')
    ax.set_ylabel('latitude [º]')

    # Plot the highlighted grid cell using the converted dataset
    highlighted_gridcell_converted.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='Reds', add_colorbar=False)

    # Set extent to focus on the area around the grid cell using the original test_lon (which is now in -180 to 180 range after conversion)
    # Convert the original test_lon to -180 to 180 range for setting extent
    test_lon_converted_for_extent = (nearest_gridcell_original.lon.values[()] + 180) % 360 - 180
    ax.set_extent([test_lon_converted_for_extent - 15, test_lon_converted_for_extent + 15, 
                   test_lat - 15, test_lat + 15], crs=ccrs.PlateCarree())
    ax.set_title(location)
    
    
def plot_monthly_T_pdfs(T_invert, T_lens, location, color1, color2, path,
                        ylim=1, lat=None, lon=None, tickfontsize = 14, 
                        axislabelfontsize=16, titlefontsize=18, legendfontsize=14):
    '''
    Create a figure and a grid of 3x4 subplots (1 for each month)
    Convert longitude from input lon on -180/180 map to lon on 0/360 map
    '''
    month_dict = {}
    month_dict[1]='jan'; month_dict[2]='feb'; month_dict[3]='mar'
    month_dict[4]='apr'; month_dict[5]='may'; month_dict[6]='june'
    month_dict[7]='july'; month_dict[8]='aug'; month_dict[9]='sept'
    month_dict[10]='oct'; month_dict[11]='nov'; month_dict[12]='dec'
    
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 10))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    if location == 'Global mean':
        lens_vals = T_lens['gmean']
        invert_vals = T_invert['gmean']
        suptitle_text = f'Monthly T anomaly distributions: \n {location}'
        
    else:
        # Convert the input longitude from -180/180 to 0/360 range for selection
        lon_0_360 = (lon + 360) % 360

        lens_vals = T_lens['anoms'].sel(lat=lat, lon=lon_0_360, method='nearest')
        invert_vals = T_invert['T'].sel(lat=lat, lon=lon_0_360, method='nearest')
                # Determine N/S for latitude and E/W for longitude
        lat_hemisphere = 'N' if lens_vals.lat.values >= 0 else 'S'
        lon_hemisphere = 'E' if lens_vals.lon.values <= 180 else 'W' # Assuming 0-360 E representation

        # Construct the suptitle string with degree symbols and hemispheres
        suptitle_text = f'Monthly T anomaly distributions: \n {location} ({abs(lens_vals.lat.values):.1f}º {lat_hemisphere}, {abs(lens_vals.lon.values if lens_vals.lon.values <= 180 else 360 - lens_vals.lon.values):.1f}º {lon_hemisphere})'
        
    # Find max T anomaly value
    vals_max = np.max([np.abs(lens_vals).max().values, np.abs(invert_vals).max().values])

    for month_id in range(1, 12 + 1):
        # Calculate the index for the current subplot
        ax_index = month_id - 1
        ax = axes[ax_index]

        sns.kdeplot(lens_vals.where(lens_vals.month == month_id, drop=True).values.flatten(),
                    label='LENS', color=color1, ax=ax)

        sns.kdeplot(invert_vals.where(invert_vals.month == month_id, drop=True).values.flatten(),
                    label='InVERT', color=color2, ax=ax)

        ax.set_title(month_dict[month_id], fontsize=titlefontsize) 
        
        if month_id == 1: ax.legend(fontsize=legendfontsize)

        # Adjust xlim based on the maximum absolute value found
        ax.set_xlim(-vals_max, vals_max)
        ax.set_ylim(0, ylim)
        
        xticks = np.round([-2/3*vals_max, 0, 2/3*vals_max], 1)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks, size=tickfontsize)
        
        yticks = np.round([0, ylim/2, ylim], 2)
        
        if month_id in [1, 5, 9]:
            ax.set_ylabel('density', fontsize=axislabelfontsize)
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticks, size=tickfontsize)
        else: ax.set_ylabel(''); ax.set_yticks([]); ax.set_yticklabels('')
            
        if month_id in [9, 10, 11, 12]:
            ax.set_xlabel('T anomaly [K]', fontsize=axislabelfontsize)
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticks, size=tickfontsize)
        else: ax.set_xlabel(''); 

    # Adjust layout to prevent overlapping titles and labels
    plt.tight_layout()
    plt.suptitle(suptitle_text, y=1.06, fontsize=20)
    plt.show()
    plt.savefig(path + 'monthly_T_pdfs_'+location+'.png')
    plt.savefig(path + 'monthly_T_pdfs_'+location+'.pdf')