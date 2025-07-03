# TODO - fix the x axis -5 dBm to -inf
# Quick and dirty SE and EE plots for GLOBECOM paper

import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys
sys.path.append('/Users/apw804/dev_02/EnergyModels/KISS')


# Set the project path
project_path = Path("~/dev_02/EnergyModels/KISS").expanduser().resolve()
project_path_str = str(project_path)
print(f'Project path: {project_path}')
cells_turned_down = []

######################
### Scenario Setup ###
######################

center_power_reduced = pd.read_feather(project_path / 'data/analysis/GLOBECOM/center_power_reduced.feather')

opposite_inner_power_reduced = pd.read_feather(project_path / 'data/analysis/GLOBECOM/opposite_inner_power_reduced.feather')

non_adjacent_inner_power_reduced = pd.read_feather(project_path / 'data/analysis/GLOBECOM/non_adjacent_inner_power_reduced.feather')

center_and_adjacent_power_reduced = pd.read_feather(project_path / 'data/analysis/GLOBECOM/center_and_adjacent_power_reduced.feather')




################
### PLOTTING ###
################


# Create figure and axes
fig, ax = plt.subplots(figsize=(12, 10))


# Define the dataframes
dataframes = [
    center_power_reduced,
    opposite_inner_power_reduced,
    non_adjacent_inner_power_reduced,
    center_and_adjacent_power_reduced
]

# Get a dataframe with the 'experiment_reduced_power_dBm', 'energy_efficiency_mean', and 'energy_efficiency_std' columns for each dataframe joined together and suffixed with the dataframe position in the list
ee_results = pd.concat([df[['experiment_reduced_power_dBm', 'energy_efficiency_mean', 'energy_efficiency_std']].add_suffix(f'_{i}') for i, df in enumerate(dataframes)], axis=1)

# Get a dataframe with the 'experiment_reduced_power_dBm', 'energy_efficiency_mean', and 'energy_efficiency_std' columns for each dataframe joined together and suffixed with the dataframe position in the list
ee_results = pd.concat([df[['experiment_reduced_power_dBm', 'energy_efficiency_mean', 'energy_efficiency_std']].add_suffix(f'_{i}') for i, df in enumerate(dataframes)], axis=1)

# Get a list of column names for the ee_results dataframe
ee_results_columns = ee_results.columns.tolist()
# Drop columns that start with 'experiment_reduced_power_dBm' and do not end with '_0'
ee_results_columns = [col for col in ee_results_columns if not col.startswith('experiment_reduced_power_dBm') or col.endswith('_0')]

# Update the ee_results dataframe to only include the columns in the ee_results_columns list
ee_results = ee_results.loc[:, ee_results_columns]

# Multiply all columns that start with `energy_efficiency` by 1e6 to convert from bps/Hz to Mbps/Hz
ee_results.loc[:, ee_results.columns.str.startswith('energy_efficiency')] = ee_results.loc[:, ee_results.columns.str.startswith('energy_efficiency')] * 1e6

# Describe the ee_results dataframe
ee_results.describe()


# Set dodge (for error bars)
dodge = 0.0

# Set the `cell_set_power_level` column where the value is -inf to -2
for df in dataframes:
    df.loc[df['experiment_reduced_power_dBm'] == -np.inf, 'experiment_reduced_power_dBm'] = -2
    df['experiment_reduced_power_dBm'] = df['experiment_reduced_power_dBm'] + dodge
    dodge = dodge + 0.3




def plot_errorbar(ax=None, x=None, y=None, yerr=None,
                  fmt='.', capsize=2, linewidth=1, label=None):
    '''Plot the errorbar for the data.'''
    if ax is None:
        ax = plt.gca()
    ax.errorbar(x=x,
                y=y,
                yerr=yerr,
                fmt=fmt,
                capsize=capsize,
                linewidth=linewidth,
                label=label,
                clip_on=False)


# Define the dataset parameters
datasets = [
    (center_power_reduced, 'S1', 's-'),
    (opposite_inner_power_reduced, 'S2', '^-'),
    (non_adjacent_inner_power_reduced, 'S3', 'd-'),
    (center_and_adjacent_power_reduced, 'S4', 'o-')
]



#########################
### CELL EE (bits/J) ###
#########################

x_data_series = 'experiment_reduced_power_dBm'
y_data_series = 'energy_efficiency_mean'
yerr_series = 'energy_efficiency_std'

for data, label, fmt in datasets:
    plot_errorbar(ax = ax,
                  x = data[x_data_series],
                  y = data[y_data_series],
                  yerr = data[yerr_series],
                  fmt=fmt,
                  label=label)

ax.set_ylabel('Mean Network Energy Efficiency (bits/J)')


#######################
### Plot formatting ###
#######################

# Auto adjust the padding between subplot
fig.tight_layout(pad=3.0)

# Set the x axis label
ax.set_xlabel(r'Transmit power of all cells in $K_v$ (dBm)')
# Set x and y axis lower limits
ax.set_xlim(left=-3, right=45)
# Enable minor ticks
ax.minorticks_on()
# Set the grid on
ax.grid(linestyle='--', linewidth=0.5,)

# Format the legend with title bold and underlined
ax.legend(title='Scenario', title_fontsize=11.5, fancybox=True, framealpha=0.5)


# Set the figure size to 3.45 x 2.5 inches
fig.set_size_inches(3.45, 2.5)

# Scale the axes to fit the figure
fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)

# Change the first x axis tick label to -inf
ax.set_xticklabels(['-inf', '1', '4', '7', '10', '13', '16', '19', '22', '25', '28', '31', '34', '37', '40', '43', '46'])

# Set the x axis tick labels
ax.set_xticks(np.arange(-3, 46, 3))

# Set the y limits
ax.set_ylim(bottom=8.5, top=11.5)

# Increase the axis label font size
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(10)

# Put major and minor ticks inside the figure
ax.tick_params(which='both', direction='in')

####################
### FILE OUTPUT ###
####################

# Add a timestamp to the figure
# fig_timestamp(fig, author='Kishan Sthankiya')

# Save the figure starting with the date in ISO format (YYYY_MM_DD)
date = datetime.datetime.now().date().isoformat()
date_str = date.replace('-', '_')

# Path to save the figure /Users/apw804/dev_02/EnergyModels/KISS/data/figures/GLOBECOM
fig_dir = project_path / 'data' / 'figures' / 'GLOBECOM'


fig_filename = f'{date_str}_rcp_energy_efficiency_scenarios_v2_3.pdf'
fig_path = fig_dir / fig_filename
fig.savefig(
    fig_path,
    dpi=300,
    format='pdf',
    bbox_inches='tight',
)


