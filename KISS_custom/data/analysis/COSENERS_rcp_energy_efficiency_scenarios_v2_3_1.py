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
plt.style.use(['science', 'ieee', 'no-latex'])
plt.rcParams.update({'figure.dpi': '200'})


# Set the global font family to sans-serif
plt.rcParams['font.family'] = 'sans-serif'
# Set the font to Helvetica Semibold
plt.rcParams['font.sans-serif'] = ['Helvetica', 'sans-serif']


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


######################
### PLOTTING SETUP ###
######################

# Create figure and axes
fig, ax = plt.subplots()

# Create a twin axes that shares the x and y axes with the ax axes
ax1 = ax.twinx()

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
                           ncols=1, 
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

######################
### Correct Values ###
######################

# Convert `energy_efficiency_mean` from Mbps/J to bps/J
for df in dataframes:
    df['energy_efficiency_mean'] = df['energy_efficiency_mean'] * 1000000
    df['energy_efficiency_std'] = df['energy_efficiency_std'] * 1000000

#########################
### CELL EE (bps/J) ###
#########################

x_data_series = 'experiment_reduced_power_dBm'
y_data_series = 'energy_efficiency_mean'
yerr_series = 'energy_efficiency_std'

for data, label, fmt, col, dodge in datasets:
    
    # Plot on ax1 for the -15 dBm (-inf) value
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
    plot_errorbar(ax = ax,
                  x = data[x_data_series].iloc[1:],
                  y = data[y_data_series].iloc[1:],
                  yerr = data[yerr_series].iloc[1:],
                  fmt=fmt,
                  label=label,
                  col=col,
                  legend=True,
                  dodge=dodge)

ax.set_ylabel('Mean Network Energy Efficiency (bps/J)')

# Set the x axis label
ax.set_xlabel(r'Transmit power of all cells (dBm)')
ax.set_xlabel(r'Transmit power of all cells (dBm)')

# Set x and y axis lower limits
ax.set_xlim(left=-6, right=44.5)
ax1.set_xlim(left=-6, right=44.5)

# # Set the y limits
ax.set_ylim(bottom=8.7 * 1e6, top=11.3 * 1e6)
ax1.set_ylim(bottom=8.7 * 1e6, top=11.3 * 1e6)


# Change the first x axis tick label to -inf
ax.set_xticklabels(['-inf', '', '1', '4', '7', '10', '13', '16', '19', '22', '25', '28', '31', '34', '37', '40', '43', '46'])
ax1.set_xticklabels(['-inf', '', '1', '4', '7', '10', '13', '16', '19', '22', '25', '28', '31', '34', '37', '40', '43', '46'])

# Set the x axis tick labels
ax.set_xticks(np.arange(-5, 46, 3))

# Turn off ax1 y axis labels
ax1.set_yticklabels([])
ax1.set_ylabel('')

#######################
### Plot formatting ###
#######################

# plt.style.use(['science', 'ieee'])




# Default figure sizes for IEEE journals are:
# Single column: 3.5 inches (width) x 2.163 inches (height)     # height given by golden ratio
# Double column: 7.16 inches (width) x 4.425 inches (height)    # height given by golden ratio

ieee_1_col = (3.5, 2.163)
ieee_2_col = (7.16, 4.425)

# Set the figure size to the default 1 column size
fig.set_size_inches(ieee_1_col)

# Print the figure size in inches
fig_size = fig.get_size_inches()
print(f'Figure size: {fig_size[0]} inches wide by {fig_size[1]} inches high')

# Get the size of the y axis label font size
y_label_size = ax.yaxis.label.get_size()
x_label_size = ax.xaxis.label.get_size()
print(f'Y axis label size: {y_label_size} points')
print(f'X axis label size: {x_label_size} points')

# Get the size of the tick labels
y_tick_label_size = ax.yaxis.get_ticklabels()[0].get_size()
print(f'Y axis tick label size: {y_tick_label_size} points')
x_tick_label_size = ax.xaxis.get_ticklabels()[0].get_size()
print(f'X axis tick label size: {x_tick_label_size} points')

# Get the font size of the legend labels
legend_label_size = ax.legend_.get_texts()[0].get_size()
print(f'Legend label size: {legend_label_size} points')

# Centre the axes in the figure
fig.tight_layout()

# Legend formatting
ax.legend(ncol=2)   # Set the number of columns in the legend
ax.legend().set_visible(True)   # Make the legend visible
ax.legend().set_draggable(True)   # Make the legend draggable

# Set legend label order
handles, labels = ax.get_legend_handles_labels()
order = ['Centre', 'Central triad', 'Inner Ring, antipodal', 'Inner Ring, alternate']
ax.legend([handles[labels.index(key)] for key in order], order, ncol=2)

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

fig_filename = f'{date_str}_rcp_energy_efficiency_scenarios_v2_3_2.pdf'
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



#####################
### COSENERS PLOT ###
#####################

# Set the global font family to sans-serif
plt.rcParams['font.family'] = 'sans-serif'
# Set the font to Helvetica Semibold
plt.rcParams['font.sans-serif'] = ['Helvetica', 'sans-serif']

cos_fig, cos_ax = plt.subplots()


data, label, fmt, col, dodge = datasets[3]
dodge=0.0
    
# Plot on cos_ax for the -15 dBm (-inf) value
marker_only = fmt[-1]                     # Get the marker only (remove the line)
plot_errorbar(ax = cos_ax,
                x = data[x_data_series].iloc[0],
                y = data[y_data_series].iloc[0],
                yerr = data[yerr_series].iloc[0],
                fmt=fmt,
                label=label,
                col=col,
                legend=False,
                dodge=dodge)

# Plot once on cos_ax for the values over 0 dBm
plot_errorbar(ax = cos_ax,
                x = data[x_data_series].iloc[1:],
                y = data[y_data_series].iloc[1:],
                yerr = data[yerr_series].iloc[1:],
                fmt=fmt,
                label=label,
                col=col,
                legend=True,
                dodge=dodge)



cos_ax.set_ylabel('Mean Network Energy Efficiency (bps/J)')
cos_ax.set_xlabel('Transmit power of all cells (dBm)')


# Set x and y axis lower limits
cos_ax.set_xlim(left=-6, right=46)

# Change the first x axis tick label to -inf
cos_ax.set_xticklabels(['-inf', '', '1', '4', '7', '10', '13', '16', '19', '22', '25', '28', '31', '34', '37', '40', '43', '46'])

# Set the x axis tick labels
cos_ax.set_xticks(np.arange(-5, 46, 3))

# Centre the axes in the figure
cos_fig.tight_layout()
# Reduce the white space around the figure
cos_fig.subplots_adjust(left=0.14, right=0.995, top=0.95, bottom=0.125)

# Turn off the legend
cos_ax.legend().set_visible(False)



print('Done!')