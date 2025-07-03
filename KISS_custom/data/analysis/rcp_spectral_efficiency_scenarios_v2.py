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
fig, ax = plt.subplots()


# Define the dataframes
dataframes = [
    center_power_reduced,
    opposite_inner_power_reduced,
    non_adjacent_inner_power_reduced,
    center_and_adjacent_power_reduced
]

# Set dodge (for error bars)
dodge = 0.0

# Set the `cell_set_power_level` column where the value is -inf to -2
for df in dataframes:
    df.loc[df['experiment_reduced_power_dBm'] == -np.inf, 'experiment_reduced_power_dBm'] = -2
    df['experiment_reduced_power_dBm'] = df['experiment_reduced_power_dBm'] + dodge
    dodge = dodge + 0.3




def plot_errorbar(ax=None, x=None, y=None, yerr=None,
                  fmt='.', label=None):
    '''Plot the errorbar for the data.'''
    if ax is None:
        ax = plt.gca()
    # Plot the errorbar
    ax.errorbar(x=x,
                y=y,
                yerr=yerr,
                fmt=fmt,
                capsize=1.5,
                capthick=0.5,
                elinewidth=0.5,
                linewidth=0.5,
                clip_on=False,
                markersize=3,
                markeredgewidth=0.5,
                markerfacecolor='white',)
    


# Define the dataset parameters
datasets = [
    (center_power_reduced, 'S1', '-s'),
    (opposite_inner_power_reduced, 'S2', '--^'),
    (non_adjacent_inner_power_reduced, 'S3', ':d'),
    (center_and_adjacent_power_reduced, 'S4', '-.o')
]



#########################
### CELL SE (bits/Hz) ###
#########################

x_data_series = 'experiment_reduced_power_dBm'
y_data_series = 'spectral_efficiency_mean'
yerr_series = 'spectral_efficiency_std'

for data, label, fmt in datasets:
    plot_errorbar(ax = ax,
                  x = data[x_data_series],
                  y = data[y_data_series] * 1e6,
                  yerr = data[yerr_series]* 1e6,
                  fmt=fmt,
                  label=label)

ax.set_ylabel('Mean Network Spectral Efficiency (bps/Hz)')


#######################
### Plot formatting ###
#######################

# plt.style.use(['science', 'ieee'])

# Set the x axis label
ax.set_xlabel(r'Transmit power of all cells in $K_v$ (dBm)')
# Set x and y axis lower limits
ax.set_xlim(left=-3, right=45)


# Change the first x axis tick label to -inf
ax.set_xticklabels(['-inf', '1', '4', '7', '10', '13', '16', '19', '22', '25', '28', '31', '34', '37', '40', '43', '46'])

# Set the x axis tick labels
ax.set_xticks(np.arange(-3, 46, 3))

# # Set the y limits
ax.set_ylim(bottom=1.80, top=2.07)

# # # Increase the axis label font size
# ax.xaxis.label.set_size(10)
# ax.yaxis.label.set_size(10)

# # # Increase the tick font size
# ax.tick_params(axis='both', which='major', labelsize=10)


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


fig_filename = f'{date_str}_rcp_spectral_efficiency_scenarios_v2_3.pdf'
fig_path = fig_dir / fig_filename
fig.savefig(fig_path)


##################
### IEEE style ###
##################

# # IEEE style for plots taken from SciencePlots

# # Matplotlib style for IEEE plots
# # This style should work for most two-column journals

# # Set color cycle
# # Set line style as well for black and white graphs
# axes.prop_cycle : (cycler('color', ['k', 'r', 'b', 'g']) + cycler('ls', ['-', '--', ':', '-.']))

# # Set default figure size
# figure.figsize : 3.3, 2.5
# figure.dpi : 600

# # Font sizes
# font.size : 8
# font.family : serif
# font.serif : Times


