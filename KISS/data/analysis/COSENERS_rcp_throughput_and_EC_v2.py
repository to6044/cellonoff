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

centre_all = pd.read_feather(project_path / 'data/analysis/GLOBECOM/throughput_and_EC_plots/centre_v2_scenario_all_cells.feather')
centre_static = pd.read_feather(project_path / 'data/analysis/GLOBECOM/throughput_and_EC_plots/centre_v2_scenario_static_cells.feather')
centre_variable = pd.read_feather(project_path / 'data/analysis/GLOBECOM/throughput_and_EC_plots/centre_v2_scenario_variable_cells.feather')


opposite_inner_all = pd.read_feather(project_path / 'data/analysis/GLOBECOM/throughput_and_EC_plots/opposite_inner_scenario_all_cells.feather')
opposite_inner_static = pd.read_feather(project_path / 'data/analysis/GLOBECOM/throughput_and_EC_plots/opposite_inner_scenario_static_cells.feather')
opposite_inner_variable = pd.read_feather(project_path / 'data/analysis/GLOBECOM/throughput_and_EC_plots/opposite_inner_scenario_variable_cells.feather')


inner_non_adjacent_all = pd.read_feather(project_path / 'data/analysis/GLOBECOM/throughput_and_EC_plots/inner_non_adjacent_scenario_all_cells.feather')
inner_non_adjacent_static = pd.read_feather(project_path / 'data/analysis/GLOBECOM/throughput_and_EC_plots/inner_non_adjacent_scenario_static_cells.feather')
inner_non_adjacent_variable = pd.read_feather(project_path / 'data/analysis/GLOBECOM/throughput_and_EC_plots/inner_non_adjacent_scenario_variable_cells.feather')


centre_plus_2_inner_all = pd.read_feather(project_path / 'data/analysis/GLOBECOM/throughput_and_EC_plots/centre_plus_2_inner_scenario_all_cells.feather')
centre_plus_2_inner_static = pd.read_feather(project_path / 'data/analysis/GLOBECOM/throughput_and_EC_plots/centre_plus_2_inner_scenario_static_cells.feather')
centre_plus_2_inner_variable = pd.read_feather(project_path / 'data/analysis/GLOBECOM/throughput_and_EC_plots/centre_plus_2_inner_scenario_variable_cells.feather')


######################
### PLOTTING SETUP ###
######################

# Create figure and axes
fig, ax = plt.subplots(2,3, figsize=(7.16, 4.77)) # 7.16, 8.8 - IEEE maximum width and height



# Inrease the spacing between subplots
fig.subplots_adjust(
    left=0.066,
    bottom=0.111,
    right=0.977,
    top=0.952,
    hspace=0.188, 
    wspace=0.208)

# # Create a twin axes that shares the x and y axes with the ax axes
# ax1 = ax.twinx()

# Define the dataframes
dataframes = [
    centre_all,
    centre_static,
    centre_variable,
    opposite_inner_all,
    opposite_inner_static,
    opposite_inner_variable,
    inner_non_adjacent_all,
    inner_non_adjacent_static,
    inner_non_adjacent_variable,
    centre_plus_2_inner_all,
    centre_plus_2_inner_static,
    centre_plus_2_inner_variable
]

# Set dodge 
dodge = 0.0

# Set the `cell_set_power_level` column where the value is -inf to -5
for df in dataframes:
    df.loc[df['experiment_reduced_power_dBm'] == -np.inf, 'experiment_reduced_power_dBm'] = -5
    df['experiment_reduced_power_dBm'] = df['experiment_reduced_power_dBm'] #+ dodge
    dodge = dodge # + 0.77

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
        # Create the legend in the position of ax[1,1]
        legend = ax.legend(loc='upper right')

        # Set the legend title and labels
        legend.set_title('Legend')

        # Get the legend handles and texts
        handles, labels = ax.get_legend_handles_labels()

        # Customize the legend handles and texts
        for handle, label in zip(handles, labels):
            handle.set_linewidth(1.5)  # Set the line width of the handle

        # Set the legend in ax[1,1]
        ax.add_artist(legend)
    
    if legend:
        return ax, legend
    else:
        return ax

    

# Define the dataset parameters
dataset_all = [
    (centre_all, 'Centre', '-s', 'black'),
    (opposite_inner_all, 'Inner Ring, antipodal', '--^', 'red'),
    (inner_non_adjacent_all, 'Inner Ring, alternate', ':d', 'green'),
    (centre_plus_2_inner_all, 'Central triad', '-.o', 'blue')
]

dataset_static = [
    (centre_static, 'Centre', '-s', 'black'),
    (opposite_inner_static, 'Inner Ring, antipodal', '--^', 'red'),
    (inner_non_adjacent_static, 'Inner Ring, alternate', ':d', 'green'),
    (centre_plus_2_inner_static, 'Central triad', '-.o', 'blue')
]

dataset_variable = [
    (centre_variable, 'Centre', '-s', 'black'),
    (opposite_inner_variable, 'Inner Ring, antipodal', '--^', 'red'),
    (inner_non_adjacent_variable, 'Inner Ring, alternate', ':d', 'green'),
    (centre_plus_2_inner_variable, 'Central triad', '-.o', 'blue')
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
        data['mean_cell_throughput_mean'] = data['mean_cell_throughput_mean'] * 1e6
        data['mean_cell_throughput_std'] = data['mean_cell_throughput_std'] * 1e6


#########################
### CELL Throughput   ###
#########################

x_data_series = 'experiment_reduced_power_dBm'
y_data_series = 'mean_cell_throughput_mean'
yerr_series = 'mean_cell_throughput_std'

temp_store = {}

def plot_throughput(ax, dataset, ax_index, letter_label, dodge=0.0):

    for data, label, fmt, col in dataset:
        # Plot on ax1 for the -15 dBm (-inf) value
        marker_only = fmt[-1]  # Get the marker only (remove the line)
        
        plot_errorbar(ax=ax[0, ax_index],
                      x=data[x_data_series].iloc[0],
                      y=data[y_data_series].iloc[0],
                      fmt=fmt,
                      label=label,
                      col=col)

        # Plot once on ax for the values over 0 dBm
        plot_errorbar(ax=ax[0, ax_index],
                      x=data[x_data_series].iloc[1:] + dodge,
                      y=data[y_data_series].iloc[1:],
                      fmt=fmt,
                      label=label,
                      col=col)
        
        ax[0, ax_index].set_xlabel(letter_label)
        ax[0, ax_index].set_xlim(left=-6, right=49)
        ax[0, ax_index].set_xticks(np.arange(-5, 49, 3))
        ax[0, ax_index].set_xticklabels(['-inf', '', '1', '4', '7', '10', '13', '16', '19', '22', '25', '28', '31', '34', '37', '40', '43', '46'], fontsize=6)

        dodge += 0.33
        
# Create dataframe from temp_store
temp_store = pd.DataFrame(temp_store)

ax[0, 0].set_ylabel('Throughput (bits/s)')
# ax[0, 1].set_xlabel(r'Transmit power of all cells in $K_v$ (dBm)')
ax[0, 1].set_xlabel(r'Transmit power of all cells (dBm)')

# Define the dataset list with respective axes indices
datasets = [(dataset_all, 0, '(a)'), (dataset_static, 1, '(b)'), (dataset_variable, 2, '(c)')]

# Iterate over the datasets and call the plot_dataset function with dodge enabled
for dataset, ax_index, letter_label in datasets:
    plot_throughput(ax, dataset, ax_index, letter_label, dodge=True)


# cos_fig = plt.figure(figsize=(7.16, 4.77))
# cos_ax = cos_fig.add_subplot(111)
# plot_errorbar(ax=cos_ax,
#                 x=data[x_data_series].iloc[0],
#                 y=data[y_data_series].iloc[0],
#                 fmt=fmt,
#                 label=label,
#                 col=col)

# # Plot once on ax for the values over 0 dBm
# plot_errorbar(ax=cos_ax,
#                 x=data[x_data_series].iloc[1:] + dodge,
#                 y=data[y_data_series].iloc[1:],
#                 fmt=fmt,
#                 label=label,
#                 col=col)


#########################
### CELL Energy Cons  ###
#########################

x_data_series = 'experiment_reduced_power_dBm'
y_data_series = 'mean_cell_power_mean'
yerr_series = 'mean_cell_power_std'



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
        plot_errorbar(ax=ax[1, ax_index],
                      x=norm_x.iloc[0],
                      y=norm_y.iloc[0],
                      fmt=fmt,
                      label=label,
                      col=col,
                      legend=include_legend)

        # Plot once on ax for the values over 0 dBm
        plot_errorbar(ax=ax[1, ax_index],
                      x=norm_x.iloc[1:] + dodge,
                      y=norm_y.iloc[1:],
                      fmt=fmt,
                      label=label,
                      col=col,
                      legend=include_legend)

        ax[1, ax_index].set_xlabel(letter_label)
        ax[1, ax_index].set_xlim(left=-6, right=49)
        ax[1, ax_index].set_xticks(np.arange(-5, 49, 3))
        ax[1, ax_index].set_xticklabels(['-inf', '', '1', '4', '7', '10', '13', '16', '19', '22', '25', '28', '31', '34', '37', '40', '43', '46'], fontsize=6)
        


        dodge += 0.33

# Set axis label font size to 8
ax[1, 0].tick_params(axis='both', which='major', labelsize=8)

# ax[1, ax_index].set_xlabel(r'Transmit power of all cells in $K_v$ (dBm)')
        

# Call the function for each dataset
plot_dataset(ax, dataset_all, 0, '(d)', normalize=False)
plot_dataset(ax, dataset_static, 1, '(e)', normalize=False)
plot_dataset(ax, dataset_variable, 2, '(f)', normalize=False)

# Label the y axis
ax[1, 0].set_ylabel('Power Consumption (kW)')



#######################
### Plot formatting ###
#######################

# plt.style.use(['science', 'ieee'])

# Add a legend to ax[0,0]
# Move the legend up a bit
ax[1, 1].legend(loc='upper center', bbox_to_anchor=(0.5, 0.9), ncol=1, fontsize=6)
# Remove duplicate labels from the legend
handles, labels = ax[1, 1].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax[1, 1].legend(by_label.values(), 
                by_label.keys(), 
                bbox_to_anchor=(0.9, 0.8), 
                ncol=1, 
                fontsize=6, 
                borderpad=0.5,
                handletextpad=0.5,
                handlelength=1.0,
                columnspacing=0.5,
                markerscale=1.0)
order = ['Centre', 'Central triad', 'Inner Ring, antipodal', 'Inner Ring, alternate']
ax[1,1].legend([handles[labels.index(key)] for key in order], order, ncol=1)

# Adjust the y-limits for ax[1,1]
ax[1, 1].set_ylim(bottom=2.10, top=2.30)


# Reduce the space between the subplots
plt.subplots_adjust(wspace=0.1, hspace=0.1)

# Set the x axis label
fig.supxlabel(r'Transmit power of all cells in $K_v$ (dBm)')

# Set a title for axes in the first row
ax[0, 0].set_title('All cells', fontsize=8)
ax[0, 1].set_title('Static cells', fontsize=8)
ax[0, 2].set_title('Variable cells', fontsize=8)

# Default figure sizes for IEEE journals are:
# Single column: 3.5 inches (width) x 2.163 inches (height)     # height given by golden ratio
# Double column: 7.16 inches (width) x 4.425 inches (height)    # height given by golden ratio

ieee_1_col = (3.5, 2.163)
ieee_2_col = (7.16, 4.425)

# Set the figure size to the default 1 column size
fig.set_size_inches(ieee_2_col)

# Print the figure size in inches
fig_size = fig.get_size_inches()
print(f'Figure size: {fig_size[0]} inches wide by {fig_size[1]} inches high')

# Get the size of the y axis label font size
#y_label_size = ax.yaxis.label.get_size()
#x_label_size = ax.xaxis.label.get_size()
#print(f'Y axis label size: {y_label_size} points')
#print(f'X axis label size: {x_label_size} points')

# # Get the size of the tick labels
# y_tick_label_size = ax.yaxis.get_ticklabels()[0].get_size()
# print(f'Y axis tick label size: {y_tick_label_size} points')
# x_tick_label_size = ax.xaxis.get_ticklabels()[0].get_size()
# print(f'X axis tick label size: {x_tick_label_size} points')

# # Get the font size of the legend labels
# legend_label_size = ax.legend_.get_texts()[0].get_size()
# print(f'Legend label size: {legend_label_size} points')

# Centre the axes in the figure
fig.tight_layout()

# Legend formatting
# ax.legend(ncol=2)   # Set the number of columns in the legend
# ax.legend().set_visible(True)   # Make the legend visible
# ax.legend().set_draggable(True)   # Make the legend draggable

# # Set legend label order
# handles, labels = ax.get_legend_handles_labels()
# order = ['Centre', 'Central triad', 'Inner Ring, antipodal', 'Inner Ring, alternate']
# ax.legend([handles[labels.index(key)] for key in order], order, ncol=2)




#####################
### COSENERS PLOT ###
#####################

###################
# Throughput plot #
###################
x_data_series = 'experiment_reduced_power_dBm'
y_data_series = 'mean_cell_throughput_mean'
yerr_series = 'mean_cell_throughput_std'

cos_fig_tp = plt.figure()
cos_ax_tp = cos_fig_tp.add_subplot(111)

for data, label, fmt, col in dataset_all:

    # Plot on ax1 for the -15 dBm (-inf) value
    marker_only = fmt[-1]  # Get the marker only (remove the line)

    plot_errorbar(ax=cos_ax_tp,
                    x=data[x_data_series].iloc[0],
                    y=data[y_data_series].iloc[0],
                    fmt=fmt,
                    label=label,
                    col=col)

    # Plot once on ax for the values over 0 dBm
    plot_errorbar(ax=cos_ax_tp,
                    x=data[x_data_series].iloc[1:] + dodge,
                    y=data[y_data_series].iloc[1:],
                    fmt=fmt,
                    label=label,
                    col=col)

cos_ax_tp.set_xlabel('Transmit power of all cells (dBm)')
cos_ax_tp.set_ylabel('Throughput (bits/s)')
cos_ax_tp.set_xlim(left=-6, right=49)
cos_ax_tp.set_xticks(np.arange(-5, 49, 3))
cos_ax_tp.set_xticklabels(['-inf', '', '1', '4', '7', '10', '13', '16', '19', '22', '25', '28', '31', '34', '37', '40', '43', '46'], fontsize=6)

# Centre the axes in the figure
cos_fig_tp.tight_layout()
# Reduce the white space around the figure
cos_fig_tp.subplots_adjust(left=0.14, right=0.995, top=0.95, bottom=0.125)

###################
# Power Cons Plot #
###################
x_data_series = 'experiment_reduced_power_dBm'
y_data_series = 'mean_cell_power_mean'
yerr_series = 'mean_cell_power_std'

cos_fig_ec = plt.figure()
cos_ax_ec = cos_fig_ec.add_subplot(111)

data, label, fmt, col = dataset_variable[0]
dodge=0.0
marker_only = fmt[-1]  # Get the marker only (remove the line)

# Plot on ax1 for the -15 dBm (-inf) value
plot_errorbar(ax=cos_ax_ec,
                x=data[x_data_series].iloc[0],
                y=data[y_data_series].iloc[0],
                fmt=fmt,
                label=label,
                col=col)

# Plot once on ax for the values over 0 dBm
plot_errorbar(ax=cos_ax_ec,
                x=data[x_data_series].iloc[1:] + dodge,
                y=data[y_data_series].iloc[1:],
                fmt=fmt,
                label=label,
                col=col)

cos_ax_ec.set_xlabel('Transmit power of all cells (dBm)')
cos_ax_ec.set_xlim(left=-6, right=49)
cos_ax_ec.set_ylabel('Power Consumption (kW)')
cos_ax_ec.set_xticks(np.arange(-5, 49, 3))
cos_ax_ec.set_xticklabels(['-inf', '', '1', '4', '7', '10', '13', '16', '19', '22', '25', '28', '31', '34', '37', '40', '43', '46'], fontsize=6)

# Centre the axes in the figure
cos_fig_ec.tight_layout()
# Reduce the white space around the figure
cos_fig_ec.subplots_adjust(left=0.14, right=0.995, top=0.95, bottom=0.125)
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


fig_filename = f'{date_str}_rcp_throughput_and_EC_v2_2.pdf'
fig_path = fig_dir / fig_filename
fig.savefig(fig_path)