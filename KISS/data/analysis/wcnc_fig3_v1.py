# TODO - fix the x axis -5 dBm to -inf
# Quick and dirty SE and EE plots for GLOBECOM paper

import datetime
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import pandas as pd
from pathlib import Path
import sys
sys.path.append('/Users/apw804/dev_02/EnergyModels/KISS')

# Set the plot style
plt.style.use(['science', 'ieee'])
plt.rcParams.update({'figure.dpi': '200'}) # uncomment this to make the plot smaller while testing


# Set the project path
project_path = Path("~/dev_02/EnergyModels/KISS").expanduser().resolve()
project_path_str = str(project_path)
print(f'Project path: {project_path}')
cells_turned_down = []

######################
### Scenario Setup ###
######################

centre_all = pd.read_feather(project_path / 'data/analysis/GLOBECOM/throughput_and_EC_plots/centre_v2_scenario_all_cells.feather')
centre_static = pd.read_feather(project_path / 'data/analysis/GLOBECOM/throughput_and_EC_plots/centre_v2_scenario_static_cells.feather')
centre_variable = pd.read_feather(project_path / 'data/analysis/GLOBECOM/throughput_and_EC_plots/centre_v2_scenario_variable_cells.feather')


inner_antipodal_all = pd.read_feather(project_path / 'data/analysis/GLOBECOM/throughput_and_EC_plots/opposite_inner_scenario_all_cells.feather')
inner_antipodal_static = pd.read_feather(project_path / 'data/analysis/GLOBECOM/throughput_and_EC_plots/opposite_inner_scenario_static_cells.feather')
inner_antipodal_variable = pd.read_feather(project_path / 'data/analysis/GLOBECOM/throughput_and_EC_plots/opposite_inner_scenario_variable_cells.feather')


inner_alternate_all = pd.read_feather(project_path / 'data/analysis/GLOBECOM/throughput_and_EC_plots/inner_non_adjacent_scenario_all_cells.feather')
inner_alternate_static = pd.read_feather(project_path / 'data/analysis/GLOBECOM/throughput_and_EC_plots/inner_non_adjacent_scenario_static_cells.feather')
inner_alternate_variable = pd.read_feather(project_path / 'data/analysis/GLOBECOM/throughput_and_EC_plots/inner_non_adjacent_scenario_variable_cells.feather')


central_triad_all = pd.read_feather(project_path / 'data/analysis/GLOBECOM/throughput_and_EC_plots/centre_plus_2_inner_scenario_all_cells.feather')
central_triad_static = pd.read_feather(project_path / 'data/analysis/GLOBECOM/throughput_and_EC_plots/centre_plus_2_inner_scenario_static_cells.feather')
central_triad_variable = pd.read_feather(project_path / 'data/analysis/GLOBECOM/throughput_and_EC_plots/centre_plus_2_inner_scenario_variable_cells.feather')


######################
### PLOTTING SETUP ###
######################

# Create figure
fig = plt.figure(figsize=(9, 5))

# Define the dataframes
dataframes = [
    centre_all,
    centre_static,
    centre_variable,
    inner_antipodal_all,
    inner_antipodal_static,
    inner_antipodal_variable,
    inner_alternate_all,
    inner_alternate_static,
    inner_alternate_variable,
    central_triad_all,
    central_triad_static,
    central_triad_variable
]

# Set dodge 
dodge = 0.0

# Set the `cell_set_power_level` column where the value is -inf to -5
for df in dataframes:
    df.loc[df['experiment_reduced_power_dBm'] == -np.inf, 'experiment_reduced_power_dBm'] = -5
    df['experiment_reduced_power_dBm'] = df['experiment_reduced_power_dBm'] #+ dodge
    dodge = dodge # + 0.77



#######################
### CUSTOM PLOTTING ###
#######################

def plot_errorbar(ax=None, x=None, y=None, yerr=None,
                  fmt='.', label=None, col=None, legend=False):
    '''Plot the errorbar for the data.'''
    if ax is None:
        ax = plt.gca()
    # Plot the errorbar
    ax.errorbar(x=x,
                y=y,
                yerr=yerr,
                fmt=fmt,
                color=col,
                capsize=1.5,
                capthick=0.5,
                elinewidth=0.5,
                linewidth=0.5,
                clip_on=False,
                markersize=3,
                markeredgewidth=0.5,
                markerfacecolor='white',
                label=label)
    
    if legend:
        # Create the legend and auto scale the legend to keep within the axes spines 
        legend = ax.legend(loc='lower right', 
                           bbox_to_anchor=(1.0, 0.0), 
                           frameon=False, 
                           ncols=2, 
                           prop={'size': 7},
                           borderpad=0.5,
                           handletextpad=0.3,
                           handlelength=1.0,
                           columnspacing=0.3,
                           markerscale=1.0,
                           )


        # Get the legend handles and texts
        handles, labels = ax.get_legend_handles_labels()

        # Customize the legend handles and texts
        # for handle, label in zip(handles, labels):
        #    handle.set_linewidth(1.5)  # Set the line width of the handle
    
    if legend:
        return ax, legend
    else:
        return ax


######################
### DATASET PARAMS ###
######################

# Define the dataset parameters
dataset_all = [
    (centre_all, 'Centre', '-s', 'black'),
    (inner_antipodal_all, 'Inner Ring, antipodal', '--^', 'red'),
    (inner_alternate_all, 'Inner Ring, alternate', ':d', 'green'),
    (central_triad_all, 'Central triad', '-.o', 'blue')
]

dataset_static = [
    (centre_static, 'Centre', '-s', 'black'),
    (inner_antipodal_static, 'Inner Ring, antipodal', '--^', 'red'),
    (inner_alternate_static, 'Inner Ring, alternate', ':d', 'green'),
    (central_triad_static, 'Central triad', '-.o', 'blue')
]

dataset_variable = [
    (centre_variable, 'Centre', '-s', 'black'),
    (inner_antipodal_variable, 'Inner Ring, antipodal', '--^', 'red'),
    (inner_alternate_variable, 'Inner Ring, alternate', ':d', 'green'),
    (central_triad_variable, 'Central triad', '-.o', 'blue')
]

datasets = [
    dataset_all,
    dataset_static,
    dataset_variable
]

######################
### Correct Values ###
######################

# Correct the values for the throughput
for set in datasets:
    for data, label, fmt, col in set:
        data['mean_cell_throughput_mean'] = data['mean_cell_throughput_mean']
        data['mean_cell_throughput_std'] = data['mean_cell_throughput_std']


#########################
### CELL Throughput   ###
#########################

x_data_series = 'experiment_reduced_power_dBm'
y_data_series = 'mean_cell_throughput_mean'
yerr_series = 'mean_cell_throughput_std'

temp_store = {}

tp_subplot_axes = plt.subplot(111)

def plot_throughput(ax, dataset, ax_index, letter_label, dodge=0.0):

    for data, label, fmt, col in dataset:
        # Plot on ax1 for the -5 dBm (-inf) value
        marker_only = fmt[-1]  # Get the marker only (remove the line)
        
        plot_errorbar(ax=ax,
                      x=data[x_data_series].iloc[0],
                      y=data[y_data_series].iloc[0],
                      fmt=fmt,
                      label=label,
                      col=col)

        # Plot once on ax for the values over 0 dBm
        plot_errorbar(ax=ax,
                      x=data[x_data_series].iloc[1:] + dodge,
                      y=data[y_data_series].iloc[1:],
                      fmt=fmt,
                      label=label,
                      col=col)
        dodge += 0.33
        
    ax.set_xlabel(letter_label)
    ax.set_xlim(left=-6, right=49)
    ax.set_xticks(np.arange(-5, 49, 3))
    ax.set_xticklabels(['-inf', '', '1', '4', '7', '10', '13', '16', '19', '22', '25', '28', '31', '34', '37', '40', '43', '46'], fontsize=6)

# Create dataframe from temp_store
temp_store = pd.DataFrame(temp_store)

tp_subplot_axes.set_ylabel('Throughput (Mbps)')


# Define the dataset list with respective axes indices
datasets = [(dataset_all, 0, '(a)')]

# Iterate over the datasets and call the plot_dataset function with dodge enabled
for dataset, ax_index, letter_label in datasets:
    plot_throughput(tp_subplot_axes, dataset, ax_index, letter_label, dodge=True)



#########################
### Cell Power Cons  ###
#########################

x_data_series = 'experiment_reduced_power_dBm'
y_data_series = 'mean_cell_power_mean'
yerr_series = 'mean_cell_power_std'

power_subplot_axes = plt.subplot(223)

# Label the y axis
power_subplot_axes.set_ylabel('Power Consumption (kW)')

def plot_dataset(ax, dataset, ax_index, letter_label, normalize=False, include_legend=False, dodge=0.0):
    for data, label, fmt, col in dataset:
        marker_only = fmt[-1]  # Get the marker only (remove the line)

        # Normalize the data if 'normalize' is True
        if normalize:
            norm_x = data[x_data_series]  # Only normalising the y-axis
            norm_y = (data[y_data_series] - data[y_data_series].min()) / (data[y_data_series].max() - data[y_data_series].min())
        else:
            norm_x = data[x_data_series] + dodge
            norm_y = data[y_data_series]

        # Plot on ax1 for the -15 dBm (-inf) value
        plot_errorbar(ax=ax,
                      x=norm_x.iloc[0],
                      y=norm_y.iloc[0],
                      fmt=fmt,
                      label=label,
                      col=col)

        # Plot once on ax for the values over 0 dBm
        plot_errorbar(ax=ax,
                      x=norm_x.iloc[1:] + dodge,
                      y=norm_y.iloc[1:],
                      fmt=fmt,
                      label=label,
                      col=col,
                      legend=True)

        ax.set_xlabel(letter_label)
        ax.set_xlim(left=-6, right=49)
        ax.set_xticks(np.arange(-5, 49, 3))
        ax.set_xticklabels(['-inf', '', '1', '4', '7', '10', '13', '16', '19', '22', '25', '28', '31', '34', '37', '40', '43', '46'], fontsize=6)

        dodge += 0.33

# Set axis label font size to 8
power_subplot_axes.tick_params(axis='both', which='major', labelsize=8)

# Call the function for the dataset
plot_dataset(power_subplot_axes, dataset_all, 0, '(b)', normalize=False)



# Get the font size of the legend labels
legend_label_size = power_subplot_axes.legend_.get_texts()[0].get_size()
print(f'Legend label size: {legend_label_size} points')


# Legend formatting
power_subplot_axes.legend(ncol=2)   # Set the number of columns in the legend
power_subplot_axes.legend().set_visible(True)   # Make the legend visible
power_subplot_axes.legend().set_draggable(True)   # Make the legend draggable

# Put a border around the legend
power_subplot_axes.legend(frameon=True, edgecolor='black', borderpad=0.5, handletextpad=0.5, handlelength=1.0, columnspacing=0.5)


# Set legend label order
handles, labels = power_subplot_axes.get_legend_handles_labels()
order = ['Centre', 'Central triad', 'Inner Ring, antipodal', 'Inner Ring, alternate']
# Set the frame on the legend
power_subplot_axes.legend([handles[labels.index(key)] for key in order], order, ncol=2, frameon=True, edgecolor='black', borderpad=0.5, handletextpad=0.5, handlelength=1.0, columnspacing=0.5)


# power_subplot_axes.legend([handles[labels.index(key)] for key in order], order, ncol=2)




#########################
###  EE STUFF BELOW   ###
#########################

######################
### Scenario Setup ###
######################

center_power_reduced = pd.read_feather(project_path / 'data/analysis/GLOBECOM/center_power_reduced.feather')

opposite_inner_power_reduced = pd.read_feather(project_path / 'data/analysis/GLOBECOM/opposite_inner_power_reduced.feather')

non_adjacent_inner_power_reduced = pd.read_feather(project_path / 'data/analysis/GLOBECOM/non_adjacent_inner_power_reduced.feather')

center_and_adjacent_power_reduced = pd.read_feather(project_path / 'data/analysis/GLOBECOM/center_and_adjacent_power_reduced.feather')


######################
### PLOTTING SETUP ###
######################

# Create figure and axes
ee_subplot_axes = plt.subplot(222)

# Create a twin axes that shares the x and y axes with the ax axes
ax1 = ee_subplot_axes.twinx()

# Define the dataframes
dataframes = [
    center_power_reduced,
    opposite_inner_power_reduced,
    non_adjacent_inner_power_reduced,
    center_and_adjacent_power_reduced
]

# Set dodge (for error bars)
# dodge = 0.0

# Set the `cell_set_power_level` column where the value is -inf to -5
for df in dataframes:
    df.loc[df['experiment_reduced_power_dBm'] == -np.inf, 'experiment_reduced_power_dBm'] = -5
    df['experiment_reduced_power_dBm'] = df['experiment_reduced_power_dBm']
    #dodge = dodge + 0.77


def plot_errorbar(ax=None, x=None, y=None, yerr=None, dodge=0.0,
                  fmt='.', label=None, col=None, legend=False):
    '''Plot the errorbar for the data.'''
    if ax is None:
        ax = plt.gca()
    # Plot the errorbar
    ax.errorbar(x=x +dodge,
                y=y,
                yerr=yerr,
                fmt=fmt,
                color=col,
                capsize=1.5,
                capthick=0.5,
                elinewidth=0.5,
                linewidth=0.5,
                clip_on=False,
                markersize=3,
                markeredgewidth=0.5,
                markerfacecolor='white',
                label=label)
    
    if legend:
        # Create the legend and auto scale the legend to keep within the axes spines 
        legend = ax.legend(loc='upper right', 
                           bbox_to_anchor=(1.0, 1.0), 
                           frameon=False, 
                           ncols=2, 
                           prop={'size': 7},
                           borderpad=0.5,
                           handletextpad=0.5,
                           handlelength=1.0,
                           columnspacing=0.5,
                           markerscale=1.0,
                           )


        # Get the legend handles and texts
        handles, labels = ax.get_legend_handles_labels()

        # Customize the legend handles and texts
        # for handle, label in zip(handles, labels):
        #    handle.set_linewidth(1.5)  # Set the line width of the handle

    if legend:
        return ax, legend
    else:
        return ax
    

# Define the dataset parameters
datasets = [
    (center_power_reduced, 'Centre', '-s', 'black', 0.0),
    (opposite_inner_power_reduced, 'Inner Ring, antipodal', '--^', 'red',0.3),
    (non_adjacent_inner_power_reduced, 'Inner Ring, alternate', ':d', 'green',0.6),
    (center_and_adjacent_power_reduced, 'Central triad', '-.o', 'blue',0.9)
]


#########################
### CELL EE (bps/J) ###
#########################

x_data_series = 'experiment_reduced_power_dBm'
y_data_series = 'energy_efficiency_mean'
yerr_series = 'energy_efficiency_std'

ee_subplot_axes.set_ylabel('Mean Network Energy Efficiency (Mbps/J)')

# Set the x axis label
ee_subplot_axes.set_xlabel('(c)')

# Set x and y axis lower limits
ee_subplot_axes.set_xlim(left=-6, right=44.5)
ax1.set_xlim(left=-6, right=44.5)

# # Set the y limits
ee_subplot_axes.set_ylim(bottom=8.7, top=11.3)
ax1.set_ylim(bottom=8.7, top=11.3)


# Change the first x axis tick label to -inf
ee_subplot_axes.set_xticklabels(['-inf', '', '1', '4', '7', '10', '13', '16', '19', '22', '25', '28', '31', '34', '37', '40', '43', '46'], fontsize=6)
ax1.set_xticklabels(['-inf', '', '1', '4', '7', '10', '13', '16', '19', '22', '25', '28', '31', '34', '37', '40', '43', '46'], fontsize=6)

# Set the x axis tick labels
ee_subplot_axes.set_xticks(np.arange(-5, 46, 3))

# Turn off ax1 y axis labels
ax1.set_yticklabels([])
ax1.set_ylabel('')


for data, label, fmt, col, dodge in datasets:
    
    # Plot on ax1 for the -5 dBm (-inf) value
    marker_only = fmt[-1]                     # Get the marker only (remove the line)
    plot_errorbar(ax = ax1,
                  x = data[x_data_series].iloc[0],
                  y = data[y_data_series].iloc[0],
                  yerr = data[yerr_series].iloc[0],
                  fmt=fmt,
                  label=label,
                  col=col,
                  legend=False,
                  dodge=dodge)
    
    # Plot once on ax for the values over 0 dBm
    plot_errorbar(ax = ee_subplot_axes,
                  x = data[x_data_series].iloc[1:],
                  y = data[y_data_series].iloc[1:],
                  yerr = data[yerr_series].iloc[1:],
                  fmt=fmt,
                  label=label,
                  col=col,
                  legend=False,
                  dodge=dodge)
    
ax1.spines['top'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['right'].set_visible(False)


#########################
# SE STUFF BELOW        #
#########################

######################
### PLOTTING SETUP ###
######################

se_subplot_axes = plt.subplot(224)

# Create a twin axes that shares the x and y axes with the ax axes
ax1 = se_subplot_axes.twinx()

# Define the dataframes
dataframes = [
    center_power_reduced,
    opposite_inner_power_reduced,
    non_adjacent_inner_power_reduced,
    center_and_adjacent_power_reduced
]

# Set dodge (for error bars)
#dodge = 0.0

# Set the `cell_set_power_level` column where the value is -inf to -5
for df in dataframes:
    df.loc[df['experiment_reduced_power_dBm'] == -np.inf, 'experiment_reduced_power_dBm'] = -5
    df['experiment_reduced_power_dBm'] = df['experiment_reduced_power_dBm'] #+ dodge
    # dodge = dodge + 0.77


def plot_errorbar(ax=None, x=None, y=None, yerr=None, dodge=0.0,
                  fmt='.', label=None, col=None, legend=False):
    '''Plot the errorbar for the data.'''
    if ax is None:
        ax = plt.gca()
    # Plot the errorbar
    ax.errorbar(x=x +dodge,
                y=y,
                yerr=yerr,
                fmt=fmt,
                color=col,
                capsize=1.5,
                capthick=0.5,
                elinewidth=0.5,
                linewidth=0.5,
                clip_on=False,
                markersize=3,
                markeredgewidth=0.5,
                markerfacecolor='white',
                label=label)
    
    if legend:
        # Create the legend and auto scale the legend to keep within the axes spines 
        legend = ax.legend(loc='lower right', 
                           bbox_to_anchor=(1.0, 0.0), 
                           frameon=False, 
                           ncols=2, 
                           prop={'size': 7},
                           borderpad=0.5,
                           handletextpad=0.5,
                           handlelength=1.0,
                           columnspacing=0.5,
                           markerscale=1.0,
                           )
        # Get the legend handles and texts
        handles, labels = ax.get_legend_handles_labels()
    
    if legend:
        return ax, legend
    else:
        return ax
    

# Define the dataset parameters
datasets = [
    (center_power_reduced, 'Centre', '-s', 'black',0.0),
    (opposite_inner_power_reduced, 'Inner Ring, antipodal', '--^', 'red',0.3),
    (non_adjacent_inner_power_reduced, 'Inner Ring, alternate', ':d', 'green',0.6),
    (center_and_adjacent_power_reduced, 'Central triad', '-.o', 'blue',0.9)
]

#########################
### CELL SE (bits/Hz) ###
#########################

x_data_series = 'experiment_reduced_power_dBm'
y_data_series = 'spectral_efficiency_mean'
yerr_series = 'spectral_efficiency_std'

for data, label, fmt, col, dodge in datasets:
    
    # Plot on ax1 for the -5 dBm (-inf) value
    marker_only = fmt[-1]                     # Get the marker only (remove the line)

    y_data_series_bps_hz = data[y_data_series] * 1e6
    yerr_series_bps_hz = data[yerr_series] * 1e6

    plot_errorbar(ax = ax1,
                  x = data[x_data_series].iloc[0],
                  y = y_data_series_bps_hz.iloc[0],
                  yerr = yerr_series_bps_hz.iloc[0],
                  fmt=fmt,
                  label=label,
                  col=col,
                  legend=False,
                  dodge=dodge)
    
    # Plot once on ax for the values over 0 dBm
    plot_errorbar(ax = se_subplot_axes,
                  x = data[x_data_series].iloc[1:],
                  y = y_data_series_bps_hz.iloc[1:],
                  yerr = yerr_series_bps_hz.iloc[1:],
                  fmt=fmt,
                  label=label,
                  col=col,
                  legend=False,
                  dodge=dodge)

se_subplot_axes.set_ylabel('Mean Network Spectral Efficiency (bps/Hz)')


#######################
### Plot formatting ###
#######################

# Set the x axis label
se_subplot_axes.set_xlabel('(d)')


# Set x and y axis lower limits
se_subplot_axes.set_xlim(left=-6, right=44.5)
ax1.set_xlim(left=-6, right=44.5)

# # Set the y limits
se_subplot_axes.set_ylim(bottom=1.8, top=2.06)
ax1.set_ylim(bottom=1.8, top=2.06)


# Change the first x axis tick label to -inf
se_subplot_axes.set_xticklabels(['-inf', '', '1', '4', '7', '10', '13', '16', '19', '22', '25', '28', '31', '34', '37', '40', '43', '46'], fontsize=6)
ax1.set_xticklabels(['-inf', '', '1', '4', '7', '10', '13', '16', '19', '22', '25', '28', '31', '34', '37', '40', '43', '46'], fontsize=6)

# Set the x axis tick labels
se_subplot_axes.set_xticks(np.arange(-5, 46, 3))

# Turn off ax1 y axis labels
ax1.set_yticklabels([])
ax1.set_ylabel('')

ax1.spines['top'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Set the x axis label
fig.supxlabel(r'Transmit power of all cells in $K_v$ (dBm)')


####################
### FILE OUTPUT ###
####################

# Add a timestamp to the figure
# fig_timestamp(fig, author='Kishan Sthankiya')

# Save the figure starting with the date in ISO format (YYYY_MM_DD)
date = datetime.datetime.now().date().isoformat()
date_str = date.replace('-', '_')

# Path to save the figure /Users/apw804/dev_02/EnergyModels/KISS/data/figures/GLOBECOM
fig_dir = project_path / 'data' / 'figures' / 'WCNC'
# If the path does not exist, create it
if not fig_dir.exists():
    fig_dir.mkdir(parents=True, exist_ok=True)
fig_filename = f'{date_str}_WCNC_tp_ec_ee_se_v1.pdf'
fig_path = fig_dir / fig_filename

fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
fig.tight_layout()

# Set figure output size
fig.savefig(fig_path, pad_inches=0.1, bbox_inches='tight', dpi=300, format='pdf')