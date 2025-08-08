#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np, xarray as xr, matplotlib.pylab as plt, pandas as pd, seaborn as sns
import pickle, random, re, io, regionmask, dask, timeit, os, shutil, datetime
from scipy.signal import welch
from eofs.xarray import Eof
from scipy.stats import genextreme # Import GEV distribution from scipy

from statsmodels.tsa.api import VAR
from contextlib import redirect_stdout
from distributed import Client
from scipy import stats
import cartopy.crs as ccrs, cartopy.feature as cfeature
import warnings; warnings.filterwarnings('ignore')

from InVERT_functions import (calc_weights, calc_EOFs, areaweighted_mean, autocorr, 
unstack_time, stack_time, createRandomSortedList, make_summary_strings, make_coeff_dataset,  
custom_sum_along_mode, compare_T_pdfs, calc_psd_stats, plot_GMST_psd_spread, calc_emean_autocorrs,  
calc_ensemble_std_autocorrs, calc_efold_time, calc_eft_stats, compare_autocorrs_emean, plot_GMST_comparisons,
save_region_means, plot_regional_psd_spread, plot_regional_T_pdfs, plot_regional_emean_autocorrs, 
plot_regional_comparison, find_var_name, compare_MSE_to_emean_PSD, welch_psd, get_ensemble_variance, 
calc_emean_gridcell_MSE, plot_regional_variance_stats, plot_regional_eft_stats, plot_combined_coeffs_and_eofs,
gridcell_map_plot, plot_gridcell_diff, plot_regional_diff_map, calc_gridcell_psd, plot_MSE_by_region,
calc_efold_time_dataset, convert_longitude_range, plot_monthly_T_pdfs, plot_gridcell)


# In[2]:


path = '/home/msaenger/InVERT/Vector_autoregression/'

scenario = 'Historical'

savepath = '/home/msaenger/InVERT/Vector_autoregression/08.01.2025_monthly_EOFs_SSP370/'

n_samples = 50
n_steps = 1032
optimal_lag = 12
nmodes = 100
M = 120


import regionmask; import matplotlib as mpl

# Figure formatting
mpl.rcParams['font.family'] = 'sans-serif' # Change the default font family
tickfontsize = 14; axislabelfontsize=16
titlefontsize=18; legendfontsize=14
color1 = 'goldenrod'; color2='teal'


if scenario == 'Historical':
    scenario_name = 'HIST'
else: scenario_name = scenario
    
nmodes = 100
optimal_lag = 12
    
# Path where LENS2 Historical data is stored
lpath = '/home/msaenger/InVERT/Vector_autoregression/LENS2_' + scenario + '_info/'



# Load EOF solver objects by month

solvers_bymonth = {}

for month in range(1,13):

    solvers_bymonth[month] = calc_EOFs(0, path=lpath + 'EOFs_by_month/',
                                       filename='LENS2_' + scenario_name + \
                                       '_regridded_monthly_deseasoned_Tanom_EOFs_month='+str(month))


## Extract PCs, EOFs, and variance fractions from the number of specified modes 

eofs_dict = {}

for month in range(1, 13):

    eofs_dict[month] = {}
    
    eofs_dict[month]['eofs'] = solvers_bymonth[month].eofs().sel(mode=slice(0, nmodes-1))
    eofs_dict[month]['pcs'] = solvers_bymonth[month].pcs().sel(mode=slice(0, nmodes-1))
    eofs_dict[month]['varfracs'] = solvers_bymonth[month].varianceFraction().sel(mode=slice(0, nmodes-1))



# Extract cos(lat) weights 
weights = solvers_bymonth[1].getWeights()
weights = xr.DataArray(weights, coords=[eofs_dict[1]['eofs']['lat'], 
                                        eofs_dict[1]['eofs']['lon']], 
                       dims=['lat', 'lon'])


# Remove EOF solvers from active memory
print('removing EOF solvers from active memory')
del solvers_bymonth


### Load LENS2 T anomalies (pre-processed)
Tanoms_lens = xr.open_dataset(lpath + scenario_name + \
                              '_regridded_monthly_TREFHT_anoms_deseasoned_concatted_with_gmean.nc')

# Save month IDs from original T anomaly time series
month_da = xr.DataArray(Tanoms_lens.month.values,
                        coords={'time': Tanoms_lens.time.values, 
                                'month': ('time', Tanoms_lens.month.values)},
                        dims=['time'])


# ### Compile DataArrays of PCs from the EOF solvers for each month


# Save PC data array (separated by ensemble member) for each month and store in dictionary

pcs_unstacked = {}

for month in range(1, 13):

    pcs_unstacked[month] = unstack_time(eofs_dict[month]['pcs'].drop('month'), esize = 50) 


# Convert dictionary to a dataset. Variable names will be the month (1-12)
pcs_by_month_dataset = xr.Dataset(pcs_unstacked)



## Randomly select a subset ensemble members from which to use the month-specific 
# PCs to train the VAR model

n_training_members = 25 # Number of training ensemble members to use

rand_ens_list = (createRandomSortedList(n_training_members))

training_pcs_bymonth_unstacked = pcs_by_month_dataset.sel(ensemble = rand_ens_list)



# Re-stack the 25 chosen training ensemble members over time

training_pcs_bymonth = stack_time(training_pcs_bymonth_unstacked)


# Extract each month's PC data array and adjust the 'time' values so as to put them 
# back together in time order (e.g. month 1 year 1, month 2 year 1, ... month 12 year 1, 
# month 1 year 2, month 2 year 2, ... etc)

month_pc_da_list = []

for month in range(1, 13):

    training_pcs_da_month = training_pcs_bymonth[month].drop('ensemble')
    training_pcs_da_month['time'] = training_pcs_da_month.time * 12 + month - 1
    training_pcs_da_month = training_pcs_da_month.to_dataset(name='pcs')
    
    month_pc_da_list.append(training_pcs_da_month)


# Merge into one dataset, sorted by time and re-apply a month coordinate

training_pcs = xr.merge(month_pc_da_list).sortby('time')
training_pcs['month'] = month_da.sel(time=slice(0,len(training_pcs['time'])))


training_pcs = training_pcs.assign_coords({'month': training_pcs.month})


### TRAIN 12 VAR models (one for each month of the year) using input-output pairs and standardization

    # monthly_var_models will contain the regression coefficients and residuals 
    # for each monthly VAR model.

monthly_var_models = {}

# Iterate through each target month (1 for January to 12 for December)

for target_month in range(1, 13): # Loop through all 12 months

    # Lists to store input features and output targets for the current month 
    input_features = []
    output_targets = []

    # Iterate through the time dimension of the original training data to create input-output pairs
    # Start the loop from optimal_lag so we have enough preceding data
    
    for i in range(optimal_lag, len(training_pcs.time)):
        
        # Check if the current month is the target month
        if training_pcs.month.values[i] == target_month:
            
            # Extract the preceding optimal_lag consecutive months of standardized PC data
            # Flatten the data from (lag, mode) to a 1D array (lag * mode)
            features = training_pcs.pcs.values[i - optimal_lag : i, :].flatten()
            input_features.append(features)

            # Extract the current month's PCs as the output target
            targets = training_pcs.pcs.values[i, :]
            output_targets.append(targets)

    # Convert input and output feature lists to numpy arrays
    input_features = np.array(input_features)
    output_targets = np.array(output_targets)

    # Check if enough data points to train the model
    if len(input_features) > 0:
        
        # Fit a linear regression model using the input-output pairs
        # Add a column of ones to input_features to account for the intercept (constant term)
        X = np.hstack([np.ones((input_features.shape[0],1)),
                       input_features])
        y = output_targets

        # Solve for the coefficients using least squares regression
        coefficients, residuals_info, rank, s = np.linalg.lstsq(X, y, rcond=None)

        # Calculate the predicted values for the training data
        # aka the linear combination (@) of the input features (lagged PCs and the intercept) 
            # using the learned coefficients to produce the model's predicted PC values 
            # for each instance of the target month in the training data.
        predicted_targets = X @ coefficients

        # Calculate the residuals (actual - predicted)
        # shape is the number of target_months in the training_pc time series minus 1
            # because the first instance of target_month won't have a full 12 months before it
        residuals = y - predicted_targets

        # The first row of coefficients is the intercept of the lstsq calculation,
            # the rest are for the lagged coefficients
        intercept = coefficients[0, :]
        lagged_coeffs = coefficients[1:, :]

        # Store the trained model components (coefficients, intercept, and residuals) for the target month
        # Reshape lagged_coeffs back to (lag, input_mode, output_mode)
        monthly_var_models[target_month] = {
            'intercept': intercept,
            'lagged_coeffs': lagged_coeffs.reshape((optimal_lag, nmodes, nmodes)), 
            'residuals': residuals} 


# Create a list to store DataArrays for each month
monthly_coeffs_da_list = []

# Define coordinates
## Corrected lags_coord to match the 'months ago' meaning of the indices of the NumPy array
lags_coord = np.arange(optimal_lag, 0, -1) # Lags from 12 down to 1
input_modes_coord = np.arange(nmodes)
output_modes_coord = np.arange(nmodes)

# Iterate through each month in the dictionary
for month, components in monthly_var_models.items():
    lagged_coeffs = components['lagged_coeffs']

    # Create a DataArray for the lagged coefficients of the current month
    coeffs_da = xr.DataArray(lagged_coeffs,
                             coords={'lag': lags_coord,
                                     'input_mode': input_modes_coord,
                                     'output_mode': output_modes_coord},
                             dims=['lag', 'input_mode', 'output_mode'])

    # Add the month as a coordinate
    coeffs_da = coeffs_da.expand_dims(month=[month])

    monthly_coeffs_da_list.append(coeffs_da)

# Concatenate the DataArrays along the new 'month' dimension
lagged_coeffs_dataset = xr.concat(monthly_coeffs_da_list, dim='month')

lagged_coeffs_dataset = lagged_coeffs_dataset.to_dataset(name='lagged_coefficients')

lagged_coeffs_dataset.to_netcdf(savepath + 'lagged_coeff_dataset.nc')


# ## Plot VAR coefficients

def plot_var_coeffs(lagged_coeffs_dataset, months_per_row, xaxis_lag=12, 
                    output_modes_per_row=4):

    from matplotlib.colors import SymLogNorm, Normalize
    import matplotlib.gridspec as gridspec

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

    # create axes for colorbars
    # Adjust colorbar position based on the number of coefficient rows
    cbar_coeffs_ax = fig.add_axes([0.92, 0.15, 
                                   0.012, 0.7]) # Left, bottom, width, height

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

            ax.set_xlim(-12.5, -0.5)
            ax.set_ylim(0.5, 4 + 0.45)
            ax.tick_params(axis='both', which='major', labelsize=tickfontsize)
            ax.set_yticks([1,2,3,4])
            ax.set_xticks([-12, -10, -8, -6, -4, -2])

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


plot_var_coeffs(lagged_coeffs_dataset, months_per_row=[1,2,3])

plot_var_coeffs(lagged_coeffs_dataset, months_per_row=[7,8,9])

plot_var_coeffs(lagged_coeffs_dataset, months_per_row=[10,11,12])


# # EMULATE PCs AND TEMPERATURE


def emulate_pcs(training_pcs, monthly_var_models, n_training_members, 
                optimal_lag, n_samples, n_steps, nmodes, M, path):
    '''
    Use monthly VAR model components to generate synthetic PC time series
    
    Parameters:
        - training_pcs (xr.DataArray): The original time-sorted training PCs. Used for initial conditions.
        - monthly_var_models (dict): Dictionary containing trained monthly VAR model components 
                                     (intercepts and lagged coefficients) for training PCs
        - n_training_members (int): The number of original training data ensemble members' PCs 
                                    used to train the VAR model.
        - optimal_lag (int): The number of timesteps (months) used as lag.
        - n_samples (int): The number of emulated ensemble members to produce.
        - n_steps (int): The desired length of each emulated ensemble member after truncation (months).
        - nmodes (int): The number of modes.
        - M (int): The number of extra months of PCs to simulate for burn-in
        - path (str): Path to save results.
    '''
    # Get the required number of past values for VAR initialization
    lag_order = optimal_lag

    # We need to generate more steps than desired length to allow for burn-in and to ensure
    # we can start the truncated series in January.
    # Generate at least lag_order + M (burn-in, 120 months) + 12 (to ensure a January start) steps
    # A bit of extra buffer doesn't hurt for initial condition influence decay.
    total_steps_to_generate = n_steps + lag_order + M + 12

    # Prepare the original training PCs for initial conditions selection
    training_pcs_values = training_pcs.pcs.values

    # Save month values from training PCs sorted by time
    training_months_values = training_pcs.month.values

    # Select any time index besides the last lag_order indices for initial conditions
    len_pcs = len(training_pcs.time)
    valid_start_indices = np.arange(len_pcs - lag_order)

    # Avoid indices where the initial condition sequence spans across different original ensemble members 
        # in the training data: record indices in stacked training PCs where a new ensemble member starts
    # This assumes training_pcs_sorted was created by stacking ensemble members of equal length
    len_training_member = len(training_pcs.time) // n_training_members 
    indices_btwn_ens_members = [len_training_member * i for i in range(1, n_training_members)]
    
    # Create a set of 'forbidden indices' to avoid selecting as a start index for initial conditions
        # in the stacked training_pcs; aka the last 'optimal_lag' indices from each training ensemble member
    forbidden_indices = []
    for idx in indices_btwn_ens_members:
        # Prevent choosing an initial condition index within the last (lag_order - 1) indices
        # of each ensemble member, so that initial conditions stay within one ensemble member
        for i in range(-lag_order + 1, 1): # Use 1 here to include the index itself
            forbidden_indices.append(idx + i)

    # Filter out the forbidden indices from valid_start_indices
    valid_start_indices = [idx for idx in valid_start_indices if idx not in forbidden_indices]

    emulated_samples = []
    
    for i in range(n_samples):
        
        # Randomly select a starting point for initial conditions from valid indices
        start_index = np.random.choice(valid_start_indices)

        # Select last `lag_order` time steps of training PCs as initial conditions, for each mode
        initial_conditions = training_pcs_values[start_index : \
                                                 start_index + lag_order, :]
        
        # Initialize the synthetic PC time series for this sample as a list 
            # (start with the initial conditions)
        # synthetic_series is identical to initial_conditions in values, 
            # but in list form
        synthetic_series = [initial_conditions[j, :] for j in range(lag_order)]
            
        # Determine the month of the last initial condition: first, find its index (as a 0-index)
        last_initial_condition_month_index = start_index + lag_order - 1
        
        # use last initial condition month index to find its corresponding month in training_months_values
        current_month = training_months_values[last_initial_condition_month_index] 
        
        # Generate the rest of the synthetic series for the given ensemble members 
        # (length = total_steps_to_generate - lag_order)
        for step in range(total_steps_to_generate - lag_order):

            # Determine the month we are predicting
            # The month sequence wraps around from 12 to 1
            month_to_predict = (current_month % 12) + 1

            # Get the trained model components for the month to predict
            model_components = monthly_var_models[month_to_predict]
            intercept = model_components['intercept']
            lagged_coeffs = model_components['lagged_coeffs'] # Shape: (lag, input_mode, output_mode)
            residuals = model_components['residuals'] # Shape: ( (len(training_pcs) // 12 -1), nmodes )

            # Get the input features: the last 'lag_order' steps from the current synthetic series
                # at first, this will be the randomly chosen 12 initial conditions for each mode
                # Then, as we add new synthetic PCs, this will draw the last lag_order synthetic PCs
            input_features = np.array(synthetic_series[-lag_order:]) # Shape: (lag, mode)

            # Reshape and flatten the input features to match the training format
            input_features_flat = input_features.flatten() # Shape: (lag * mode,)

            # Add the intercept term # Shape: (lag * mode + 1,)
            input_features_with_intercept = np.hstack([np.ones(1), input_features_flat]) 

            # Reshape lagged_coeffs for matrix multiplication: (lag*input_mode, output_mode)
            lagged_coeffs_reshaped = lagged_coeffs.reshape((lag_order * nmodes, nmodes))

            # Predict the next step (PCs for the month_to_predict)   # Shape: (nmodes,)
            predicted_pcs = intercept + np.dot(input_features_flat, 
                                               lagged_coeffs_reshaped) 

            # Randomly sample a residual vector from the residuals for this month 
                # calculated in VAR model training
            # Start by getting a random index from the shape of the residuals
            random_residual_index = np.random.choice(residuals.shape[0])

            # Grab a residual vector from that index # Shape: (nmodes,)
            sampled_residual = residuals[random_residual_index, :]

            # Add the sampled residual to the predicted PCs
            predicted_pcs_with_residual = predicted_pcs + sampled_residual

            # Append the predicted PCs (with residual) to the synthetic series
            synthetic_series.append(predicted_pcs_with_residual)

            # Update the current month for the next step
            current_month = month_to_predict
        
        ### After generating 'total_steps_to_generate' steps of new PCs, find the first January and truncate

        # Convert list to array for easier slicing
        synthetic_series = np.array(synthetic_series) 
        
        # Determine the month sequence for the synthetic PC time series just generated
        # The month of the first generated step (at index lag_order) is (month of last initial condition % 12) + 1
        start_month_generated = (training_months_values[start_index + lag_order - 1] % 12) + 1
        
        # Create array of the full sequence of months, starting at start_month_generated
        full_months_sequence = np.tile(np.arange(1, 13), 
                                       total_steps_to_generate // 12 + 2)[start_month_generated - \
                                       1 : start_month_generated - 1 + total_steps_to_generate - lag_order]

        # Prepend the months of the initial conditions
        initial_condition_months = training_months_values[start_index : start_index + lag_order]
        full_months_sequence = np.concatenate((initial_condition_months, full_months_sequence))
        
        # Find the index of the first January *after* the initial conditions AND burn-in period
        # Search within the generated part of the series (from index lag_order onwards)
        # Find the index of the first January in the full generated sequence
        first_january_index = -1

        for idx in range(lag_order + M, total_steps_to_generate): # Start searching after burn-in
            if full_months_sequence[idx] == 1: # January
                first_january_index = idx
                break

        # if you found an index for the first january after burn-in and ICs:
        if first_january_index != -1: 
            
            # Truncate the series: remove everything before the first January after burn-in and ICs
            truncated_sample = synthetic_series[first_january_index : first_january_index + n_steps, :]
            # Adjust months
            months = full_months_sequence[first_january_index : first_january_index + n_steps]
        else:
            # This case should ideally not happen if total_steps_to_generate is large enough,
            # but as a fallback, take the last n_steps and assign a repeating month sequence.
            # This would mean the series might not start in January.
            print("Warning: Could not find a January after burn-in. Truncating from the end.")
            truncated_sampled = synthetic_series[-n_steps:, :]
            months = np.tile(np.arange(1, 13), n_steps // 12 + 1)[-n_steps:] # Assign repeating months
            
        # Store the truncated sample and its corresponding months
        emulated_samples.append(truncated_sample)
        
    # Convert list of synthetic PC time series to NumPy array
    # Shape: (num_samples, n_steps, num_modes)
    emulated_samples = np.array(emulated_samples)   
    
    # Convert the list of month arrays into a single array
    months = np.array(months) # Assuming all truncated samples have the same month sequence
    
    # Convert the emulated PCs to an xarray DataArray
    emulated_da = xr.DataArray(emulated_samples,
                                dims=("ensemble", "time", "mode"),
                                coords={"ensemble": np.arange(n_samples),
                                         "time": np.arange(n_steps),
                                        "mode": np.arange(nmodes)})

    # Assign the month coordinate to the emulated PCs DataArray
    # Ensure the month coordinate is broadcast correctly across ensemble members
    emulated_da = emulated_da.assign_coords({'month': ('time', months)})

    # Convert to Dataset with variable name 'PCs'
    new_pcs_final = emulated_da.to_dataset(name='pcs')

    # Assign the month coordinate again, just to be sure, although it should be carried over
    new_pcs_final = new_pcs_final.assign_coords({'month': ('time', months)})
    
    new_pcs_final.to_netcdf(savepath + 'InVERT_PCs_final_monthly_VAR.nc')
    
    return new_pcs_final


InVERT_pcs = emulate_pcs(training_pcs, monthly_var_models, n_training_members,
                         optimal_lag, n_samples, n_steps, nmodes, M, savepath)


# Separate PCs back into separate months, multiply by EOFs independently, and re-merge sorted by time

# Step 1: separate PCs back out into separate months 
# Compute PCs * EOFs for each mode and divide by weights for every month. Save in dict.

print('Multiplying PCs * EOFs and dividing by weights')

products_by_month = {}

for month in range(1, 13):
    print(month)
    
    products_by_month[month] = InVERT_pcs.groupby('month')[month] * eofs_dict[month]['eofs'] / weights


# Sum T anomalies over modes then merge

print('Summing over modes')

products_by_month_summed = {}

for month in range(1, 13):
    
    print(month)
    
    products_by_month_summed[month] =  products_by_month[month].pcs.sum(dim='mode')


# Re-stack ensemble members over time (for easier re-sorting of months by time) and save in new dict

print('Stacking ensemble members over time')

Tanoms_bymonth = {}

for month in range(1, 13):
    print(month)
    
    Tanoms_bymonth[month] = (stack_time(products_by_month_summed[month]))


# Extract each month's T anomaly data array and adjust the 'time' values so as to put them 
# back together in time order (e.g. month 1 year 1, month 2 year 1, ... month 12 year 1, 
# month 1 year 2, month 2 year 2, ... etc)

print('Updating time indices')

Tanom_da_list = []

for month in range(1, 13):

    Tanoms_month = Tanoms_bymonth[month]
    Tanoms_month['time'] = Tanoms_month.time * 12 + month - 1
    Tanoms_month = Tanoms_month.to_dataset(name='T')
    Tanom_da_list.append(Tanoms_month)


# Concatenate over time dimension and then sort by time 
print('Merging and sorting by time')

InVERT_stacked = xr.concat(Tanom_da_list, dim='time').sortby('time')
InVERT_stacked['gmean'] = areaweighted_mean(InVERT_stacked.T)

InVERT_T = unstack_time(InVERT_stacked, esize=n_samples)

print('Saving final InVERT dataset')

InVERT_T.to_netcdf(savepath + 'InVERT_'+str(nmodes)+'modes_lag='+str(optimal_lag)+'.nc')

print('saved')


InVERT_T = xr.open_dataset(savepath + 'InVERT_'+str(nmodes)+'modes_lag='+str(optimal_lag)+'.nc')


# # Diagnostics


print('Diagnostics')

# Separate LENS temperature anomaly data back into ensemble members
# T_unstacked = unstack_time(Tanoms_lens, esize=50)
T_unstacked = xr.open_dataset(lpath + 'LENS_Tanoms_unstacked.nc')

# # Concatenate emulated ensemble members in time
InVERT_stacked = stack_time(InVERT_T)


# Load monthly climatology from LENS2 T anomalies (for re-introducing seasonal cycle)
monthly_means = xr.open_dataarray(lpath + 'monthly_means.nc')


# Add monthly climatologies to InVERT e.g. impose 'original' seasonal cycle
InVERT_seasonal = InVERT_stacked['T'].groupby('month') + monthly_means

LENS_seasonal = xr.open_dataarray(lpath + 'LENS_'+scenario+'_Tanoms_with_seasonal_cycle_reintroduced.nc')


def convert_lon(ds, lon_name):
    '''
    Converts a 0-360 degree longitude grid dataset to a -180 to 180 dataset
    lon_name is input string either 'longitude' or 'lon'
    '''
    ds_copy = ds.copy()
    ds_copy.coords[lon_name] = (ds_copy.coords[lon_name] + 180) % 360 - 180
    ds_copy = ds_copy.sortby(ds_copy[lon_name])
    return ds_copy


InVERT_seasonal = convert_lon(InVERT_seasonal, 'lon')
LENS_seasonal = convert_lon(LENS_seasonal, 'lon')


def plot_local_pdfs(ds_invert, ds_lens, lat, lon, location):
    
    plt.figure(figsize=(6,4))
    
    local_pdf_invert = ds_invert.sel(lat=lat, lon=lon, method='nearest')
    local_pdf_lens = ds_lens.sel(lat=lat, lon=lon, method='nearest')
    
    sns.kdeplot(local_pdf_invert, label='InVERT')
    sns.kdeplot(local_pdf_lens, label='LENS2')
    
    plt.legend()
    plt.title(location)
    
    plt.ylabel('Density')
    plt.xlabel('T anomaly [K]')
    
    plt.savefig(savepath + 'local_pdf_'+location+'.png')



# ### Calculate return periods using Generalized Extreme Value distributions
# 
# #### Step 1: find max T anomaly value at every grid cell in each year

# Calculate the year for each time step
LENS_seasonal['year'] = (LENS_seasonal.time // 12) + 1

# Reshape the LENS data to include an explicit 'year' dimension using groupby
LENS_anoms_by_year = LENS_seasonal.groupby('year')

LENS_annual_maxima = LENS_anoms_by_year.max(dim='time')


# Calculate the year for each time step
InVERT_seasonal['year'] = (InVERT_seasonal.time // 12) + 1

# Reshape the LENS data to include an explicit 'year' dimension using groupby
InVERT_anoms_by_year = InVERT_seasonal.groupby('year')

InVERT_annual_maxima = InVERT_anoms_by_year.max(dim='time')


# #### Step 2: fit a Generalized Extreme Value (GEV) distribution to the annual maxima


# Function to fit GEV distribution to a time series
def fit_gev(data):
    if data.size < 10: # GEV fitting needs a reasonable number of data points
        return np.nan, np.nan, np.nan # Return NaN if not enough data
    
    try: 
        # Fit GEV distribution to the data
        # Returns shape (c), location (loc), and scale (scale) parameters
        params = genextreme.fit(data, f0=0) # f0=0 fixes shape param to 0 - optional
        return params
    except Exception as e:
        print(f"Could not fit GEV: {e}")
        return np.nan, np.nan, np.nan # Return NaN if fitting fails
    
# # Initialize empty Data Arrays to store the GEV parameters 
# # Parameters are shape (c), location (loc), scale (scale)
# gev_params_lens = xr.DataArray(
#     np.empty((len(LENS_annual_maxima['lat']), len(LENS_annual_maxima['lon']), 3)),
#     coords={'lat': LENS_annual_maxima['lat'], 'lon': LENS_annual_maxima['lon'], 
#             'parameter': ['shape', 'location', 'scale']},
#     dims = ['lat', 'lon', 'parameter'])

gev_params_invert = xr.DataArray(
    np.empty((len(InVERT_annual_maxima['lat']), len(InVERT_annual_maxima['lon']), 3)),
    coords={'lat': InVERT_annual_maxima['lat'], 'lon': InVERT_annual_maxima['lon'], 
            'parameter': ['shape', 'location', 'scale']},
    dims=['lat', 'lon', 'parameter'])

# Iterate through each grid cell to fit the GEV distribution
for i in range(len(LENS_annual_maxima['lat'])):
    for j in range(len(LENS_annual_maxima['lon'])):
        # Extract the time series of annual maxima for the current grid cell
#         lens_data = LENS_annual_maxima.isel(lat=i, lon=j).values
        invert_data = InVERT_annual_maxima.isel(lat=i, lon=j).values

#         # Fit GEV to LENS data
#         c_lens, loc_lens, scale_lens = fit_gev(lens_data)
#         gev_params_lens.loc[dict(lat=LENS_annual_maxima['lat'][i], 
#                                  lon=LENS_annual_maxima['lon'][j])] = [c_lens, loc_lens, scale_lens]

        # Fit GEV to InVERT data
        c_invert, loc_invert, scale_invert = fit_gev(invert_data)
        gev_params_invert.loc[dict(lat=InVERT_annual_maxima['lat'][i], 
                                   lon=InVERT_annual_maxima['lon'][j])] = [c_invert, loc_invert, scale_invert]

print(savepath)

# Save the fitted parameters
# gev_params_lens.to_netcdf(lpath + 'LENS_annual_maxima_gev_params.nc')
gev_params_invert.to_netcdf(savepath + 'InVERT_annual_maxima_gev_params.nc')

gev_params_lens = xr.open_dataarray(lpath + 'LENS_annual_maxima_gev_params.nc')
gev_params_invert = xr.open_dataarray(savepath + 'InVERT_annual_maxima_gev_params.nc')


# #### Step 3: calculate return periods 

### Define the return periods (in years) for which you want to calculate return levels
return_periods = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

# Calculate the non-exceedance probabilities corresponding to these return periods
non_exceedance_probabilities = [1 - (1 / T) for T in return_periods]

### Initialize empty DataArrays to store the return levels
### Dimensions will be lat, lon, and return_period

return_levels_lens = xr.DataArray(
                        np.empty((len(gev_params_lens['lat']), 
                                  len(gev_params_lens['lon']), 
                                  len(return_periods))),
                        coords = {'lat': gev_params_lens['lat'], 
                                  'lon': gev_params_lens['lon'],
                                  'return_period' : return_periods},
                        dims = ['lat', 'lon', 'return_period'])

print('finished return levels for LENS')

return_levels_invert = xr.DataArray(
                                np.empty((len(gev_params_invert['lat']), 
                                          len(gev_params_invert['lon']), 
                                          len(return_periods))),
                                coords={'lat': gev_params_invert['lat'], 
                                        'lon': gev_params_invert['lon'], 
                                        'return_period': return_periods},
                                dims=['lat', 'lon', 'return_period'])

print('finished return levels for InVERT')


### Iterate thru each grid cell and each return period to calculate the return level

for i in range(len(gev_params_lens['lat'])):
    for j in range(len(gev_params_lens['lon'])):
        
        ## Extract the fitted GEV parameters for the current grid cell
#         c_lens, loc_lens, scale_lens = gev_params_lens.isel(lat=i, lon=j).values
        c_invert, loc_invert, scale_invert = gev_params_invert.isel(lat=i, lon=j).values
        
#         ## Check if parameters are valid (not NaN, which could happen if fitting failed)
#         if not np.isnan([c_lens, loc_lens, scale_lens]).any():
#             # Calculate return levels for LENS using the fitted parameters and probabilities
#             lens_rls = genextreme.ppf(non_exceedance_probabilities, c_lens, 
#                                       loc=loc_lens, scale=scale_lens)
#             return_levels_lens.loc[dict(lat=gev_params_lens['lat'][i],
#                                         lon=gev_params_lens['lon'][j])] = lens_rls
#         else:
#             return_levels_lens.loc[dict(lat=gev_params_lens['lat'][i], 
#                                         lon=gev_params_lens['lon'][j])] = np.nan

        ## Same as above for InVERT
        if not np.isnan([c_invert, loc_invert, scale_invert]).any():
             # Calculate return levels for InVERT using the fitted parameters and probabilities
            invert_rls = genextreme.ppf(non_exceedance_probabilities, c_invert, 
                                        loc=loc_invert, scale=scale_invert)
            return_levels_invert.loc[dict(lat=gev_params_invert['lat'][i], 
                                          lon=gev_params_invert['lon'][j])] = invert_rls
        else:
             return_levels_invert.loc[dict(lat=gev_params_invert['lat'][i], 
                                           lon=gev_params_invert['lon'][j])] = np.nan


# Save the calculated return levels
# return_levels_lens.to_netcdf(lpath + 'LENS_overall_return_levels.nc')
return_levels_invert.to_netcdf(savepath + 'InVERT_overall_return_levels.nc')

return_levels_lens = xr.open_dataarray(lpath + 'LENS_overall_return_levels.nc')


def plot_return_levels(return_levels_lens, return_levels_invert, 
                       lat, lon, location):

    return_levels_lens.sel(lat=lat, lon=lon, 
                           method='nearest').plot(label='LENS2', color=color1)
    return_levels_invert.sel(lat=lat, lon=lon, 
                             method='nearest').plot(label='InVERT', color=color2)
    plt.legend(); plt.suptitle(location)
    plt.ylabel('return level T anonmaly [K]'); plt.xlabel('return period [years]')
    plt.xlim(-1,1000)
    plt.savefig(savepath + 'Return_level_curve_'+location+'.pdf')
    plt.savefig(savepath + 'Return_level_curve_'+location+'.png')


# #### Plot curves of return level T anomaly vs return period for locations:


plot_return_levels(return_levels_lens, return_levels_invert, 
                   40.7, -74, 'New York City')

plot_return_levels(return_levels_lens, return_levels_invert, 
                   39.9, 166.4, 'Beijing')

plot_return_levels(return_levels_lens, return_levels_invert, 
                   28.6, 77.2, 'New Delhi')

plot_return_levels(return_levels_lens, return_levels_invert, 
                   18.97, 72.84, 'Mumbai')

plot_return_levels(return_levels_lens, return_levels_invert, 
                   -23.56, -46.66, 'Sao Paolo')


# Select the 100-year return level data
return_level_lens_100yr = return_levels_lens.sel(return_period=100)
return_level_invert_100yr = return_levels_invert.sel(return_period=100)

# Calculate the absolute difference (InVERT - LENS)
return_level_diff_100yr = return_level_invert_100yr - return_level_lens_100yr

# Calculate the ratio (InVERT / LENS)
# Handle potential division by zero if LENS return level is 0
return_level_ratio_100yr = xr.where(return_level_lens_100yr != 0,
                                   return_level_invert_100yr / return_level_lens_100yr,
                                   np.nan) # Use NaN where LENS return level is 0

# Choose a colormap and value range for the difference map
diff_cmap = 'RdBu_r' # Red-Blue reversed colormap
diff_vmin = -2; diff_vmax = 2 

# Choose a colormap and value range for the ratio map
ratio_cmap = 'PRGn' # Example: adjust based on what you want to highlight
ratio_vmin = 0.5; ratio_vmax = 1.5 # Example: adjust based on what you want to highlight

# --- Plotting the Absolute Difference ---
fig_diff = plt.figure(figsize=(12, 6))
ax_diff = fig_diff.add_subplot(1, 1, 1, projection=ccrs.Robinson())

im_diff = ax_diff.pcolormesh(return_level_diff_100yr['lon'], return_level_diff_100yr['lat'],
                           return_level_diff_100yr.values,
                           transform=ccrs.PlateCarree(),
                           cmap=diff_cmap, vmin=diff_vmin, vmax=diff_vmax)

ax_diff.coastlines()
ax_diff.add_feature(cfeature.BORDERS, linestyle=':')
ax_diff.set_title('Difference in 100-year Return Level (InVERT - LENS)')

cbar_diff = fig_diff.colorbar(im_diff, orientation='horizontal', 
                              label='Temperature Anomaly Difference [K]',
                              shrink = 0.6)
plt.show()
# --- Plotting the Ratio ---
fig_ratio = plt.figure(figsize=(12, 6))
ax_ratio = fig_ratio.add_subplot(1, 1, 1, projection=ccrs.Robinson())

im_ratio = ax_ratio.pcolormesh(return_level_ratio_100yr['lon'], return_level_ratio_100yr['lat'],
                            return_level_ratio_100yr.values,
                            transform=ccrs.PlateCarree(),
                            cmap=ratio_cmap, vmin=ratio_vmin, vmax=ratio_vmax) # Use log scale if ranges are large: 

ax_ratio.coastlines()
ax_ratio.add_feature(cfeature.BORDERS, linestyle=':')
ax_ratio.set_title('Ratio of 100-year Return Level (InVERT / LENS)')

cbar_ratio = fig_ratio.colorbar(im_ratio, orientation='horizontal', 
                                label='Return Level Ratio', shrink = 0.6)
plt.show()

fig_diff.savefig(savepath + '100yr_return_level_difference_map.png')
fig_ratio.savefig(savepath + '100yr_return_level_ratio_map.png')


# Select the 1000-year return level data
return_level_lens_1000yr = return_levels_lens.sel(return_period=1000)
return_level_invert_1000yr = return_levels_invert.sel(return_period=1000)

# Calculate the absolute difference (InVERT - LENS)
return_level_diff_1000yr = return_level_invert_1000yr - return_level_lens_1000yr

# Calculate the ratio (InVERT / LENS)
# Handle potential division by zero if LENS return level is 0
return_level_ratio_1000yr = xr.where(return_level_lens_1000yr != 0,
                                   return_level_invert_1000yr / return_level_lens_1000yr,
                                   np.nan) # Use NaN where LENS return level is 0


# --- Plotting the Absolute Difference ---
fig_diff = plt.figure(figsize=(12, 6))
ax_diff = fig_diff.add_subplot(1, 1, 1, projection=ccrs.Robinson())

im_diff = ax_diff.pcolormesh(return_level_diff_1000yr['lon'], return_level_diff_1000yr['lat'],
                           return_level_diff_1000yr.values,
                           transform=ccrs.PlateCarree(),
                           cmap=diff_cmap, vmin=diff_vmin, vmax=diff_vmax)

ax_diff.coastlines()
ax_diff.add_feature(cfeature.BORDERS, linestyle=':')
ax_diff.set_title('Difference in 1000-year Return Level (InVERT - LENS)')

cbar_diff = fig_diff.colorbar(im_diff, orientation='horizontal', 
                              label='Temperature Anomaly Difference [K]',
                              shrink = 0.6)
plt.show()
# --- Plotting the Ratio ---
fig_ratio = plt.figure(figsize=(12, 6))
ax_ratio = fig_ratio.add_subplot(1, 1, 1, projection=ccrs.Robinson())

im_ratio = ax_ratio.pcolormesh(return_level_ratio_1000yr['lon'], return_level_ratio_1000yr['lat'],
                            return_level_ratio_1000yr.values,
                            transform=ccrs.PlateCarree(),
                            cmap=ratio_cmap, vmin=ratio_vmin, vmax=ratio_vmax) # Use log scale if ranges are large: 
ax_ratio.coastlines()
ax_ratio.add_feature(cfeature.BORDERS, linestyle=':')
ax_ratio.set_title('Ratio of 1000-year Return Level (InVERT / LENS)')

cbar_ratio = fig_ratio.colorbar(im_ratio, orientation='horizontal', 
                                label='Return Level Ratio', shrink = 0.6)

# Optional: Save figures
fig_diff.savefig(savepath + '1000yr_return_level_difference_map.png')
fig_ratio.savefig(savepath + '1000yr_return_level_ratio_map.png')


# ## Figure 2 (GMST pdf, psd, lag and autocorrelations)

print('Plotting figure 2')


titlefontsize=20
plot_GMST_comparisons(T_unstacked, InVERT_T,
                      Tanoms_lens, InVERT_stacked,
                      color1, color2, savepath)

## Comment when done -- save regional mean data

# # Original data (stacked in time) -- saved in 'LENS2_historical_info' folder
# save_region_means(Tanoms_lens, name='LENS2_'+scenario+'_stacked', path=lpath)

# # Emulated data (stacked in time)
save_region_means(InVERT_stacked, name='InVERT_'+scenario+'_stacked', path=savepath)

# # Original data unstacked -- saved in 'LENS2_historical_info' folder
# save_region_means(T_unstacked, name='LENS2_'+scenario, path=lpath)

# # Emulated data (not stacked in time)
save_region_means(InVERT_T, name='InVERT_'+scenario, path=savepath)



# # Load regional mean data 

### Single time series 
# LENS
T_regional_stacked = xr.open_dataset(lpath + \
            'LENS2_'+scenario+'_stacked' + '_AR6_region_mean_Tanoms.nc')
# INVERT
InVERT_regional_stacked = xr.open_dataset(savepath + \
                               'InVERT_'+scenario+'_stacked' + '_AR6_region_mean_Tanoms.nc')

### By ensemble member
# LENS
T_regional = xr.open_dataset(lpath + \
        'LENS2_'+scenario + '_AR6_region_mean_Tanoms.nc')
# InVERT
InVERT_regional = xr.open_dataset(savepath + \
                        'InVERT_'+scenario+'_AR6_region_mean_Tanoms.nc')

for region in range(45):
    plot_regional_comparison(T_regional, InVERT_regional, 
                             T_regional_stacked, InVERT_regional_stacked, 
                             region, name='LENS2', color1=color1, color2=color2)

    

f, mean_psd, std_psd = calc_psd_stats(T_unstacked.gmean)
invert_f, invert_mean_psd, invert_std_psd = calc_psd_stats(InVERT_T.gmean)

# t test on mean of spectral power at each frequency
print('Compare ensemble means of power at each f')
print(stats.ttest_ind(mean_psd, invert_mean_psd))

# t test on standard deviation of spectral power at each frequency
print('\nCompare ensemble stds of power at each f')
print(stats.ttest_ind(std_psd, invert_std_psd))

# t test on variance of spectral power at each frequency
print('\nCompare ensemble variances of power at each f')
print(stats.ttest_ind(std_psd**2, invert_std_psd**2))



## Compare global mean and stds of variances of global mean T anomalies

stds = T_unstacked.gmean.std('time'); stds_invert = InVERT_T.gmean.std('time')

means = T_unstacked.gmean.mean('time'); means_invert = InVERT_T.gmean.mean('time')

variances = T_unstacked.gmean.var('time'); variances_invert = InVERT_T.gmean.var('time')

print('p value: variances of global mean T anomalies', stats.ttest_ind(variances, variances_invert).pvalue)
print('p value: means of GMST anoms', stats.ttest_ind(means, means_invert).pvalue)
print('p value: standard deviations of GMST anoms', stats.ttest_ind(stds, stds_invert).pvalue)



plot_regional_variance_stats(T_regional, InVERT_regional, color1, color2, 'std', savepath);


plot_regional_variance_stats(T_regional, InVERT_regional, color1, color2, 'var', savepath);


region_mean_efts_lens, region_mean_efts_invert = plot_regional_eft_stats(T_regional, InVERT_regional, 
                                                                         savepath, color1, color2);


# ## Map plots

# ## Calculate gridcell standard deviations of T anomaly data separated into ensemble members

# Calculate std of T anomalies for InVERT and LENS
std_invert = InVERT_T.T.std('time') 

# Save gridcell T anomaly standard deviations
std_invert.to_netcdf(savepath + 'std_invert.nc'); 

# std_lens = T_unstacked.anoms.std('time') # already calculated and saved in lpath
# std_lens.to_netcdf(path + 'std_lens.nc')


# Load gridcell T anomaly standard deviations 
std_invert = xr.open_dataarray(savepath + 'std_invert.nc')
std_lens = xr.open_dataarray(lpath + 'std_lens.nc')


# Compute ensemble mean standard deviations at each gridcell
emean_std_invert = std_invert.mean('ensemble')
emean_std_lens = std_lens.mean('ensemble')


# Save LENS gridcell PSD to common directory
# LENS_grid_psd = calc_gridcell_psd(T_unstacked.anoms)
# LENS_grid_psd.to_netcdf(lpath + 'LENS_gridcell_psd.nc')

InVERT_grid_psd = calc_gridcell_psd(InVERT_T.T)

# Save InVERT gridcell PSD 
InVERT_grid_psd.to_netcdf(savepath + 'InVERT_gridcell_psd.nc')


LENS_grid_psd = xr.open_dataset(lpath + 'LENS_gridcell_psd.nc')
InVERT_grid_psd = xr.open_dataset(savepath + 'InVERT_gridcell_psd.nc')

# Compute ensemble mean MSE of PSDs at each gridcell
invert_grid_emean_mse = calc_emean_gridcell_MSE(LENS_grid_psd, InVERT_grid_psd)


# # # ### Calculate e-folding times of T anomalies at every gridcell and save
# LENS_gridcell_tau = calc_efold_time_dataset(find_var_name(T_unstacked))
# # # Save LENS gridcell e-folding times in a common folder 
# LENS_gridcell_tau.to_netcdf(lpath + 'LENS_gridcell_efoldingtimes.nc')


print('calculating gridcell e-folding times')
InVERT_gridcell_tau = calc_efold_time_dataset(find_var_name(InVERT_T))
InVERT_gridcell_tau.to_netcdf(savepath + 'Invert_gridcell_efoldingtimes.nc')


# Load saved gridcell autocorrelation e-folding times
efts_invert = xr.open_dataarray(savepath + 'Invert_gridcell_efoldingtimes.nc')
efts_lens = xr.open_dataarray(lpath + 'LENS_gridcell_efoldingtimes.nc')


# Data for the maps

shrink = 0.78
# data1: gridcell % difference STD (top left)
vmin_pct_std, vmax_pct_std, img, ds1 = plot_gridcell_diff(D1=emean_std_lens, D2=emean_std_invert,
                                            stat='standard deviation', difference='percent',
                                            ax=None, fig=None, shrink = shrink, colorbar=False);
plt.clf();

# data2: regional percent difference STD (top right)
ds2 = plot_regional_diff_map(T_regional, InVERT_regional, ds_gridded=InVERT_T,
                               difference='percent', stat='std', vmin=vmin_pct_std, 
                               vmax=vmax_pct_std, ax=None, fig=None, shrink=shrink,
                               colorbar=False); plt.clf();

# data 3: gridcell absolute difference STD (bottom left)
vmin_pct_std, vmax_pct_std, img, ds3 = plot_gridcell_diff(D1=emean_std_lens, D2=emean_std_invert,
                                                stat='standard deviation', difference='absolute',
                                                ax=None, fig=None, shrink = shrink, colorbar=True,
                                                find_vlims=False, vmin=-0.6, vmax=0.6);  plt.clf();
# data4: gridcell PSD (bottom middle)
ds4 = invert_grid_emean_mse

# data5: gridcell absolute difference e-folding time
ds5 = efts_lens.mean('ensemble') - efts_invert.mean('ensemble')


print('Plotting figure 3')


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 8),
                         subplot_kw={'projection': ccrs.Robinson()})

# Remove the extra subplot in the first row (since you only need 2)
fig.delaxes(axes[0, 2])  

# First row (2 subplots)

ax = axes[0,0]
im1 = ax.pcolormesh(ds1['lon'], ds1['lat'], ds1.values, 
                            transform=ccrs.PlateCarree(), cmap='RdBu',
                    vmin=-50, vmax=50) 
# Add stock image to mask ocean
ax.stock_img()
ax.add_feature(cfeature.OCEAN, zorder=100, edgecolor='black', 
               facecolor='gainsboro')

ax = axes[0, 1]
im2 = axes[0, 1].pcolormesh(ds2['lon'], ds2['lat'], ds2.values, 
                            transform=ccrs.PlateCarree(), cmap='RdBu',
                            vmin=-50, vmax=50)

# Add stock image to mask ocean
ax.stock_img()
ax.add_feature(cfeature.OCEAN, zorder=100, edgecolor='black', 
               facecolor='gainsboro')

# Get AR6 land regions
regions = regionmask.defined_regions.ar6.land
# Plot AR6 region boundaries with custom edgecolor and linewidth
regions.plot(ax=ax,add_ocean=True,
             ocean_kws={'facecolor': 'lightgray'},
             label='number', text_kws={'visible':False},
             line_kws={'lw':1}) # linewidth of region boundaries

ax = axes[1,0]
im3 = ax.pcolormesh(ds3['lon'], ds3['lat'], ds3.values, 
                    transform=ccrs.PlateCarree(), cmap='RdBu',
                    vmin=-0.6, vmax=0.6)
# Add stock image to mask ocean
ax.stock_img()
ax.add_feature(cfeature.OCEAN, zorder=100, edgecolor='black', 
               facecolor='gainsboro')

ax = axes[1,1]
im4 = ax.pcolormesh(ds4['lon'], ds4['lat'], ds4.values, 
                    transform=ccrs.PlateCarree(), cmap='Reds',
                    vmin=0, vmax=1)
# Add stock image to mask ocean
ax.stock_img()
ax.add_feature(cfeature.OCEAN, zorder=100, edgecolor='black', 
               facecolor='gainsboro')
ax = axes[1,2]
im5 = ax.pcolormesh(ds5['lon'], ds5['lat'], ds5.values, 
                    transform=ccrs.PlateCarree(), cmap='RdBu',
                    vmin=-4, vmax=4)
# Add stock image to mask ocean
ax.stock_img()
ax.add_feature(cfeature.OCEAN, zorder=100, edgecolor='black', 
               facecolor='gainsboro')

fig.subplots_adjust(left=0.05,  # Adjust left margin
                    bottom=0.05,  # Adjust bottom margin
                    right=0.95,  # Adjust right margin
                    top=0.95,   # Adjust top margin
                    wspace=0.05,  # Reduce horizontal spacing between subplots
                    hspace=0.1)  # Reduce vertical spacing between subplots

# Increase horizontal space between top two subplots AND center them in the top row
extra_space = 0.1  # Adjust this value to control the extra space

# Get the original positions of both top subplots
pos1 = axes[0, 0].get_position(); pos2 = axes[0, 1].get_position()

# Calculate the total width of the two subplots plus the extra space
total_width = pos1.width + pos2.width + extra_space

# Calculate the starting position to center the subplots
start_position = (1 - total_width) / 2  # 1 represents the total figure width

# Shift the left subplot to the calculated starting position
axes[0, 0].set_position([start_position, pos1.y0, pos1.width, pos1.height])

# Shift the right subplot to the right of the left subplot with extra space
axes[0, 1].set_position([start_position + pos1.width + extra_space, pos2.y0, pos2.width, pos2.height])

# Update Colorbar Calculation:
# The center position must be recalculated because the subplots positions have changed.
left_edge = axes[0, 0].get_position().x1
right_edge = axes[0, 1].get_position().x0
center_position = (left_edge + right_edge) / 2

cbar_ax = fig.add_axes([center_position - 0.018,  # Adjust position with padding as needed
                        axes[0, 1].get_position().y0, 
                        0.01,
                        axes[0, 1].get_position().height])
fig.colorbar(im1, cax=cbar_ax, orientation='vertical', label='Shared Colorbar Label')

# Add subplot labels
axes[0, 0].text(0.00, 1.00, '(a)', transform=axes[0, 0].transAxes, fontsize=14, fontweight='bold', va='top')
axes[0, 1].text(0.00, 1.00, '(b)', transform=axes[0, 1].transAxes, fontsize=14, fontweight='bold', va='top')
axes[1, 0].text(0.00, 1.00, '(c)', transform=axes[1, 0].transAxes, fontsize=14, fontweight='bold', va='top')
axes[1, 1].text(0.00, 1.00, '(d)', transform=axes[1, 1].transAxes, fontsize=14, fontweight='bold', va='top')
axes[1, 2].text(0.00, 1.00, '(e)', transform=axes[1, 2].transAxes, fontsize=14, fontweight='bold', va='top')

#Adjust tick parameters for vertical color bar:
cb_vert = fig.colorbar(im1, cax=cbar_ax, orientation='vertical', label='Shared Colorbar Label')
cb_vert.ax.tick_params(labelsize=14)

# Adjust font size of labels for all colorbars (only for colorbars in bottom row)
# Note: cax is required for setting the label on the vertical colorbar

for ax in axes[1,:]: # get axes for colorbars in the bottom row
    
    if ax == axes[1,0]:
        cb = plt.colorbar(im3, ax=ax, orientation='horizontal', shrink=0.6, pad=0.04)
        cb.set_label(label=r'($\sigma_{InVERT}-\sigma_{LENS2}$) [K]', size=14)  # Set colorbar label font size
        
    if ax == axes[1,1]:
        cb = plt.colorbar(im4, ax=ax, orientation='horizontal', shrink=0.6, pad=0.04)
        cb.set_label(label='MSE$_{InVERT}$ [(log($^\circ$C$^{2}$/year))$^{2}$]', size=14)  # Set colorbar label font size
        
    if ax == axes[1,2]:
        cb = plt.colorbar(im5, ax=ax, orientation='horizontal', shrink=0.6, pad=0.04,)
        cb.set_label(label=r'$(\tau_{InVERT} - \tau_{LENS2})$ [months]', size=14)  # Set colorbar label font size

    # Add stock image to mask ocean
    ax.stock_img()
    ax.add_feature(cfeature.OCEAN, zorder=100, edgecolor='black', 
                   facecolor='gainsboro')
        
#Adjust the colorbar label for the vertical colorbar:
cb_vert.set_label(label=r'$\frac{(\sigma_{InVERT}-\sigma_{LENS2})}{\sigma_{LENS2}}*100$ [%]', 
                  size=16)

plt.show()

fig.savefig(savepath + 'fig3_regional_subplots.pdf'); 
fig.savefig(savepath + 'fig3_regional_subplots.png'); 


# ## Supplemental Figures
# (a) regional STD absolute difference; (b) regional MSE; (c) regional EFT absolute difference


# Regional MSE: LENS  vs InVERT ensemble mean PSD
regional_MSEs_dict_InVERT_vs_LENS = {}

for region in range(46):    
   mses = compare_MSE_to_emean_PSD(tseries_1 = find_var_name(T_regional).sel(mask=region), 
                                   tseries_2 = find_var_name(InVERT_regional).sel(mask=region))
   regional_MSEs_dict_InVERT_vs_LENS[region] = mses 



import matplotlib.gridspec as gridspec

# Desired subplot sizes (in inches)
subplot_widths = [10, 10, 10]  # Widths for the 3 columns
subplot_heights = [5.5]  # Heights for the one row

# Calculate figure size
fig_width = sum(subplot_widths); fig_height = sum(subplot_heights) 

# Create figure and GridSpec
fig = plt.figure(figsize=(fig_width, fig_height))
gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=subplot_widths, height_ratios=subplot_heights)

shrink = 0.78 # input to change size of colorbar

# Create axes objects
axes = [fig.add_subplot(gs[i, j], projection=ccrs.Robinson()) for i in range(1) for j in range(3)]

# Add subplot labels
subplot_labels = ['(a)', '(b)', '(c)']
for i, ax in enumerate(axes):
    ax.text(-0.05, 1.15, subplot_labels[i], transform=ax.transAxes,  
            fontsize=16, fontweight='bold', va='top', ha='left', zorder=10)
    
# Top Left Subplot
plot_regional_diff_map(T_regional, InVERT_regional, ds_gridded=InVERT_T,
                        difference='absolute', stat='std', vmin=vmin_pct_std, 
                       vmax=vmax_pct_std, ax=axes[0], fig=fig, shrink=shrink)   
    
# Middle Subplot
plot_MSE_by_region(regional_MSEs_dict_InVERT_vs_LENS,
                   cbar_label='MSE [(log($^\circ$C$^{2}$/year))$^{2}$]',
                   title="AR6 region mean-squared error: \n ensemble mean PSD",
                   ds_gridded=InVERT_T, vmin=0, vmax=1, ax=axes[1], fig=fig, shrink=shrink)

# Right Subplot
plot_regional_diff_map(region_mean_efts_lens, region_mean_efts_invert, InVERT_T,
                       'absolute', 'eft', vmin=-4, vmax=4, ax=axes[2], 
                       fig=fig, shrink=shrink, cmap='RdBu_r')

plt.tight_layout(); plt.show()

fig.savefig(savepath + 'SI_figure2.pdf'); fig.savefig(savepath + 'SI_figure2.png')



# (a) gridcell EFT % difference; (b) regional EFT % difference


# Desired subplot sizes (in inches)
subplot_widths = [10, 10]  # Widths for the two columns
subplot_heights = [5.5]  # Heights for the one 

# Calculate figure size
fig_width = sum(subplot_widths); fig_height = sum(subplot_heights) 

# Create figure and GridSpec, adding a column for the colorbar
fig = plt.figure(figsize=(fig_width + 1, fig_height))  # +2 for colorbar space
gs = gridspec.GridSpec(1, 3, figure=fig, 
                      width_ratios=[subplot_widths[0], 0.5, subplot_widths[1]],  # Width for colorbar
                      height_ratios=subplot_heights)

shrink = 0.4 # input to change size of colorbar

# Create axes objects
ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.Robinson())
ax2 = fig.add_subplot(gs[0, 2], projection=ccrs.Robinson())

cbar_ax = fig.add_axes([0.495, 0.17, 0.012, 0.64])  # [left, bottom, width, height]


axes = [ax1, ax2]

# Add subplot labels
subplot_labels = ['(a)', '(b)']
for i, ax in enumerate(axes):
    ax.text(-0.05, 1.15, subplot_labels[i], transform=ax.transAxes,  
            fontsize=16, fontweight='bold', va='top', ha='left', zorder=10)

# Bottom Left Subplot
vmin_abs_eft, vmax_abs_eft, img, _ = plot_gridcell_diff(efts_lens.mean('ensemble'),
                                                efts_invert.mean('ensemble'),
                                                stat='autocorrelation e-folding time',
                                                difference='percent', cmap='RdBu_r',
                                                find_vlims=False, vmin=-50, vmax=50, ax=axes[0], 
                                                fig=fig, shrink=shrink, colorbar=False)

# Add the shared colorbar
cbar = fig.colorbar(img, cax=cbar_ax, shrink=shrink) 
cbar.set_label(r'($\tau_{InVERT}-\tau_{LENS2}$) / $\tau_{LENS} * 100$ [%]', 
               fontsize=axislabelfontsize)

cbarticks = [-40, -20, 0, 20, 40]; cbar.set_ticks(cbarticks)

# Create a list of tick labels with empty strings for unwanted ticks
labels = [str(val) if val in cbarticks else '' for val in cbarticks]
cbar.set_ticklabels(labels, fontsize=tickfontsize)


# Right Subplot
plot_regional_diff_map(region_mean_efts_lens, region_mean_efts_invert, InVERT_T,
                       'percent', 'eft', vmin_abs_eft, vmax_abs_eft, ax=axes[1], 
                       fig=fig, shrink=shrink, colorbar=False, cmap='RdBu_r')

fig.savefig(savepath + 'SI_figure_3_efts_percent.pdf'); 
fig.savefig(savepath + 'SI_figure_3_efts_percent.png')


print('Plotting seasonal STDs')


months = np.tile(np.arange(1, 13), len(Tanoms_lens.time) // 12 + 1)[:len(Tanoms_lens.time)] 

Tanoms_lens = Tanoms_lens.assign_coords({'month': ('time', months)})

InVERT_T_converted_longitude = convert_lon(InVERT_stacked, 'lon')
Tanoms_lens_converted_longitude = convert_lon(Tanoms_lens, 'lon')


# ### Plot seasonal STD differences btwn InVERT and LENS


Tanoms_lens_converted_longitude = Tanoms_lens_converted_longitude.assign_coords(
                                        {'month': ('time', months)})

InVERT_T_converted_longitude = InVERT_T_converted_longitude.assign_coords(
                    {'month': ('time', months[:len(InVERT_T_converted_longitude.time)])})



# Define seasons
seasonal_months = {
    'DJF': [12, 1, 2],
    'MAM': [3, 4, 5],
    'JJA': [6, 7, 8],
    'SON': [9, 10, 11]}

# Calculate seasonal standard deviations for LENS
seasonal_std_lens = {}
for season, months in seasonal_months.items():
    # Select data for the current season based on the 'month' coordinate
    seasonal_data = Tanoms_lens_converted_longitude.anoms.sel(
        time=Tanoms_lens_converted_longitude['month'].isin(months)
    )
    # Calculate standard deviation over the time dimension for the selected season
    seasonal_std_lens[season] = seasonal_data.std(dim='time')

# Calculate seasonal standard deviations for InVERT
seasonal_std_invert = {}
for season, months in seasonal_months.items():
    # Select data for the current season based on the 'month' coordinate
    seasonal_data = InVERT_T_converted_longitude.T.sel(
        time=InVERT_T_converted_longitude['month'].isin(months)
    )
    # Calculate standard deviation over the time dimension for the selected season
    seasonal_std_invert[season] = seasonal_data.std(dim='time')

print("Seasonal standard deviations calculated.")



# Calculate the seasonal differences
seasonal_diff = {}
for season in seasonal_months.keys():
    seasonal_diff[season] = seasonal_std_invert[season] - seasonal_std_lens[season]
    
seasonal_pct_diff = {}
for season in seasonal_months.keys():
    seasonal_pct_diff[season] = (seasonal_std_invert[season] - seasonal_std_lens[season]) / seasonal_std_lens[season] * 100 


import cartopy.feature as cfeature
from matplotlib.colors import Normalize
import numpy as np

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(15, 8)) # Adjust figure size as needed
gs = gridspec.GridSpec(2, 2, figure=fig)

# Create axes for the 2x2 grid, passing subplot_kw here
ax_djf = fig.add_subplot(gs[0, 0], projection=ccrs.Robinson())
ax_mam = fig.add_subplot(gs[0, 1], projection=ccrs.Robinson())
ax_jja = fig.add_subplot(gs[1, 0], projection=ccrs.Robinson())
ax_son = fig.add_subplot(gs[1, 1], projection=ccrs.Robinson())

axes = {'DJF': ax_djf,
        'MAM': ax_mam,
        'JJA': ax_jja,
        'SON': ax_son}

# Determine a common colorbar range for better comparison
# vmax = max([np.abs(diff).max() for diff in seasonal_diff.values()])
vmax = 0.5
vmin = -vmax

# Define a common colormap
cmap = 'RdBu_r' # Red-Blue reversed colormap

# Add subplot labels
subplot_labels = ['(a)', '(b)', '(c)', '(d)']
label_idx = 0

# Plot each seasonal difference map
for season, ax in axes.items():
    
    data = seasonal_diff[season]

    im = ax.pcolormesh(data['lon'], data['lat'], data.values,
                       transform=ccrs.PlateCarree(),
                       cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_title(season, fontsize=titlefontsize)
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.stock_img()
    ax.add_feature(cfeature.OCEAN, zorder=100, edgecolor='black', 
               facecolor='gainsboro')

    # Add subplot label
    ax.text(0.02, 0.98, subplot_labels[label_idx], transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', zorder=10)
    label_idx += 1

# Add a single colorbar at the bottom
cbar = fig.colorbar(im, ax=axes.values(), orientation='horizontal', 
             fraction=0.046, pad=0.04, 
             label='Difference in Standard Deviation [K]')
cbar.set_label('Difference in Standard Deviation [K]', 
               fontsize=axislabelfontsize)
cbar.ax.tick_params(labelsize=tickfontsize)

# Set custom tick labels for the colorbar
cbar_ticks = [ -0.5, 0, 0.5, ]
cbar.set_ticks(cbar_ticks)
cbar.set_ticklabels([str(tick) for tick in cbar_ticks])

plt.suptitle('Seasonal differences in standard deviations (InVERT - LENS2)',
             fontsize=22);

plt.savefig(savepath + 'Seasonal_std_differences.pdf')
plt.savefig(savepath + 'Seasonal_std_differences.png')


### Plot seasonal STDs for LENS
fig = plt.figure(figsize=(15, 8)) # Adjust figure size as needed
gs = gridspec.GridSpec(2, 2, figure=fig)

# Create axes for the 2x2 grid, passing subplot_kw here
ax_djf = fig.add_subplot(gs[0, 0], projection=ccrs.Robinson())
ax_mam = fig.add_subplot(gs[0, 1], projection=ccrs.Robinson())
ax_jja = fig.add_subplot(gs[1, 0], projection=ccrs.Robinson())
ax_son = fig.add_subplot(gs[1, 1], projection=ccrs.Robinson())

axes = {'DJF': ax_djf,
        'MAM': ax_mam,
        'JJA': ax_jja,
        'SON': ax_son}

# Determine a common colorbar range for better comparison
# vmax = max([np.abs(diff).max() for diff in seasonal_diff.values()])
vmax = 4
# vmin = -vmax
vmin = 0

# Define a common colormap
cmap = 'Oranges' # Red-Blue reversed colormap

# Add subplot labels
subplot_labels = ['(a)', '(b)', '(c)', '(d)']
label_idx = 0

# Plot each seasonal difference map
for season, ax in axes.items():
    
    data = seasonal_std_lens[season]

    im = ax.pcolormesh(data['lon'], data['lat'], data.values,
                       transform=ccrs.PlateCarree(),
                       cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_title(season, fontsize=titlefontsize)
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    
    ax.stock_img()
    ax.add_feature(cfeature.OCEAN, zorder=100, edgecolor='black', 
               facecolor='gainsboro')

    # Add subplot label
    ax.text(0.02, 0.98, subplot_labels[label_idx], transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', zorder=10)
    label_idx += 1

# Add a single colorbar at the bottom
cbar = fig.colorbar(im, ax=axes.values(), orientation='horizontal', 
             fraction=0.046, pad=0.04, 
             label='Standard Deviation [K]')
cbar.set_label('Standard Deviation [K]', 
               fontsize=axislabelfontsize)
cbar.ax.tick_params(labelsize=tickfontsize)

# Set custom tick labels for the colorbar
cbar_ticks = [0, 1, 2, 3, 4]
cbar.set_ticks(cbar_ticks)
cbar.set_ticklabels([str(tick) for tick in cbar_ticks])

plt.suptitle('Seasonally averaged temperature anomaly \n standard deviations (LENS2)',
             fontsize=22, y=0.999);

plt.savefig(savepath + 'Seasonal_stds_LENS.pdf')
plt.savefig(savepath + 'Seasonal_stds_LENS.png')


print('Plotting monthly EOF mode patterns')


# Determine the number of months (rows) and modes (columns) for the plot
num_months = 12
num_modes_to_plot = 4

fig, axes = plt.subplots(nrows=num_months, ncols=num_modes_to_plot,
                         figsize=(20, 32), # Adjusted figsize for swapped dimensions
                         subplot_kw={'projection': ccrs.Robinson()})

# Iterate through each month (rows)
for month in range(1, num_months + 1):
    # Iterate through each mode (columns)
    for mode_idx in range(num_modes_to_plot):
        ax = axes[month - 1, mode_idx] # Swapped indexing

        # Select the EOF data for the current month and mode
        eof_data = eofs_dict[month]['eofs'].sel(mode=mode_idx)

        # Plot the EOF pattern
        im = ax.pcolormesh(eof_data['lon'], eof_data['lat'], eof_data.values,
                           transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-0.04,
                           vmax=0.04)

        ax.coastlines()

        # Set titles and labels
        if month == 1: # Title for the first row (months)
            ax.set_title(f'Mode {mode_idx + 1}', fontsize=titlefontsize)
        if mode_idx == 0: # Label for the first column (modes)
            ax.text(-0.1, 0.5, f'Month {month}', va='center', ha='right',
                    rotation='vertical',
                    transform=ax.transAxes, fontsize=axislabelfontsize)
            
# Add a colorbar using a new axis
# [left,  bottom, width, height] in figure coordinates
            
cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.01])  
# Add a single colorbar to the new axis
cbar_ax = fig.colorbar(im, cax=cbar_ax, orientation='horizontal',
                       shrink=0.2, label='T anomaly [K]')
cbar.ax.tick_params(labelsize=tickfontsize+4)
cbar.set_label('T anomaly [K]', fontsize=axislabelfontsize+4)

plt.tight_layout(rect=[0, 0.07, 1, 0.98])
plt.suptitle('Monthly EOF Mode Patterns', y=1.0, fontsize=titlefontsize+4)
plt.show()

# # Optional: Save the figure
fig.savefig(savepath + 'monthly_eof_patterns.png')
fig.savefig(savepath + 'monthly_eof_patterns.pdf')



print('Local monthly standard deviations')

# Convert longitude for ensemble-separated datasets
months = np.tile(np.arange(1, 13), len(T_unstacked.time) // 12 + 1)[:len(T_unstacked.time)] 

T_unstacked = T_unstacked.assign_coords({'month': ('time', months)})

InVERT_T_converted_longitude = convert_lon(InVERT_T, 'lon')
LENS_converted_longitude = convert_lon(T_unstacked, 'lon')

# Calculate ensemble mean standard deviation of temperature anomaly at each grid cell by month
LENS_emean_monthly_stds = LENS_converted_longitude.anoms.groupby('month').std('time').mean('ensemble')
InVERT_emean_monthly_stds = InVERT_T_converted_longitude.T.groupby('month').std('time').mean('ensemble')

# Ensemble standard deviation of the above
LENS_estd_monthly_stds = LENS_converted_longitude.anoms.groupby('month').std('time').std('ensemble')
InVERT_estd_monthly_stds = InVERT_T_converted_longitude.T.groupby('month').std('time').std('ensemble')


def plot_local_monthly_T_stds(LENS_emean_monthly_stds, InVERT_emean_monthly_stds,
                              LENS_estd_monthly_stds, InVERT_estd_monthly_stds,
                              locations_lat_lon, ylim=3, markersize=5, capsize=4, elinewidth=1.5, 
                              markeredgewidth=1.5,):
    """
    Plots monthly temperature anomaly standard deviations for multiple locations
    in a 2x2 grid of subplots.

    Parameters:
        locations_lat_lon (list of tuples): A list of exactly four tuples,
                                            each containing (latitude, longitude, location_name).
        ylim (float): The upper limit for the y-axis of the plots.
    """
    n_locations = len(locations_lat_lon)
    if n_locations != 4:
        print("Please provide exactly four locations.")
        return

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10), squeeze=False) # 2x2 grid

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Add subplot labels
    subplot_labels = ['(a)', '(b)', '(c)', '(d)']

    for i, (lat, lon, location) in enumerate(locations_lat_lon):
        ax = axes[i] # Get the current subplot axis

        invert_emean_local = InVERT_emean_monthly_stds.sel(lat=lat, lon=lon, method='nearest')
        lens_emean_local = LENS_emean_monthly_stds.sel(lat=lat, lon=lon, method='nearest')
        
        invert_estd_local = InVERT_estd_monthly_stds.sel(lat=lat, lon=lon, method='nearest')
        lens_estd_local = LENS_estd_monthly_stds.sel(lat=lat, lon=lon, method='nearest')

        lat_selected = np.round(invert_emean_local.lat.values, 2)
        lon_selected = np.round(invert_emean_local.lon.values, 2)

        print(f"{location}: {invert_emean_local.lat.values}, {invert_emean_local.lon.values}")

        if lat_selected >= 0:
            NS_hemisphere = 'N'
        else: NS_hemisphere = 'S'

        if lon_selected >= 0:
            EW_hemisphere = 'E'
        else: EW_hemisphere = 'W'

        # Plot error bars with desired color
        ax.errorbar(np.arange(1, 13), lens_emean_local, yerr=lens_estd_local,
                    fmt='none', markersize=markersize, capsize=capsize, color=color1, 
                    label= 'LENS2 ',# name1+' $\mu$ +/- 1$\sigma$',  # Use fmt='none' to hide default marker
                    ecolor=color1, elinewidth=elinewidth,
                    markeredgewidth=markeredgewidth)  
        # Plot markers separately with unfilled style
        ax.plot(np.arange(1, 13), lens_emean_local, 'o', markersize=markersize, color=color1,  
                 markerfacecolor=color1, markeredgecolor=color1, label='LENS2')  

        # Plot error bars with desired color
        ax.errorbar(np.arange(1, 13), invert_emean_local, yerr=invert_estd_local,
                    fmt='none', markersize=markersize, capsize=capsize, color=color2, 
                    label= 'InVERT',# name1+' $\mu$ +/- 1$\sigma$',  # Use fmt='none' to hide default marker
                    ecolor=color2, elinewidth=elinewidth,
                    markeredgewidth=markeredgewidth)  
        # Plot markers separately with unfilled style
        ax.plot(np.arange(1, 13), invert_emean_local, 'o', markersize=markersize, color=color2, 
                 markerfacecolor=color2, markeredgecolor=color2, label='InVERT') 
        ax.set_ylim(0,ylim[i])
        ax.set_xlabel('month', fontsize=axislabelfontsize)
        ax.set_xticks([2,4,6,8,10,12])#, size=tickfontsize)
        ax.set_xticklabels(['2', '4', '6', '8', '10', '12'], fontsize=tickfontsize)
        ax.set_xlim(0.5,12.5)
        
        if ylim[i] == 2:
            ax.set_yticks([0,1,2])#,3])#, fontsize=tickfontsize)
            ax.set_yticklabels(['0', '1', '2'])#, '3'])
        if ylim[i] == 3:
            ax.set_yticks([0,1,2,3])#, fontsize=tickfontsize)
            ax.set_yticklabels(['0', '1', '2', '3'])
            

        ax.set_title(location, fontsize=titlefontsize)# + '\n ('+ str(lat_selected) + 'º'+NS_hemisphere + ', ' + \
#                      str(lon_selected) + 'º' + EW_hemisphere+')', fontsize=titlefontsize)

        if i == 0: # Add legend only to the top-left subplot
            ax.legend(fontsize=legendfontsize)

        ax.set_ylabel('standard deviation [K]', fontsize=axislabelfontsize)
        ax.text(-0.05, 1.1, subplot_labels[i], transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top', ha='left', zorder=10)

    plt.tight_layout()
    
    fig.savefig(savepath + 'emean_monthly_stds_multiple_locations.png')
    fig.savefig(savepath + 'emean_monthly_stds_multiple_locations.pdf')



locations_lat_lon = [(40.7, -74, 'New York City, United States'),
                           (39.9, 116.4, 'Beijing, China'),
                           (28.7, 77.2, 'New Delhi, India'),
                     (30.05, 31.23, 'Cairo, Egypt' )]

plot_local_monthly_T_stds(LENS_emean_monthly_stds, InVERT_emean_monthly_stds,
                          LENS_estd_monthly_stds, InVERT_estd_monthly_stds,
                          locations_lat_lon, ylim=[3,3,2,2])


# In[ ]:


# locations_lat_lon = [(28.7, 77.2, 'New Delhi, India'),
#                      (31.25, 121.47, 'Shanghai, China'),
#                      (30.05, 31.23, 'Cairo, Egypt' ),
#                      (40.7, -74, 'New York City, United States')]
                     
# plot_local_monthly_T_stds(LENS_emean_monthly_stds, InVERT_emean_monthly_stds,
#                           LENS_estd_monthly_stds, InVERT_estd_monthly_stds,
#                           locations_lat_lon, ylim=[2,2,2,3])

print('Script complete')

