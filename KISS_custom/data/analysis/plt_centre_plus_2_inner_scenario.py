# TODO - fix the x axis -5 dBm to -inf

# If editing to a new scenario - update the following:
# - data_path
# - cells_power_reduced
# - cell sleep power tweak (e.g. 0.78 * n)
# - fig_filename


import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys
sys.path.append('/Users/apw804/dev_02/EnergyModels/KISS')
import seaborn as sns
from kiss import fig_timestamp


# Set the project path
project_path = Path("~/dev_02/EnergyModels/KISS").expanduser().resolve()
project_path_str = str(project_path)
print(f'Project path: {project_path}')
data_path = project_path / 'data' / 'output' / 'reduce_adjacent_cell_4_and_5_and_9_power' / '2023_05_03'
rcp_data = data_path
cells_turned_down = []



def tsv_to_df_generator(dir_path):
    for f in dir_path.glob('*.tsv'):
        df = pd.read_csv(f, sep='\t')
        # Add a column for the experiment id
        df['experiment_id'] = f.stem
        # Split the experiment_id column on the underscore and drop the last part
        df["experiment_id"] = df["experiment_id"].str.split("_").str[:-1].str.join("_")
        # Split the experiment id on the underscore and get the 2nd part
        df["reduced_cells_seed"] = df["experiment_id"].str.split("_").str[1].str.replace("s", "")

        # Detect the cells that are turned down from the dir_path
        rcp_cells = []
        # Get the sub-string between 'reduce_cell_' and '_power' in the dir_path
        string = str(f)
        sub_string = string[string.find("reduce_cell_") + len("reduce_cell_"):string.rfind("_power")]
        # Split the sub-string on the underscore
        sub_string = sub_string.split("_")
        # Get strings that are digits and convert to int
        for cell in sub_string:
            if cell.isdigit():
                rcp_cells.append(int(cell))

        # Add columns for each of the rcp_cells where the value is the sc_power(dBm) for that cell, 
        # or -np.inf if indicated by the experiment_id
        for cell in rcp_cells:
            if 'inf' in df['experiment_id'].values[0]:
                # Set the reduced_cells_{cell}_power_dBm to -inf
                df[f'reduced_cell_{cell}_power'] = -np.inf

            df[f'reduced_cell_{cell}_power'] = df.loc[df['serving_cell_id'] == cell, 'sc_power(dBm)'].values[0]

            # Convert all values in the reduced_cells_{cell}_power_dBm column to floats
            df[f'reduced_cell_{cell}_power'] = df[f'reduced_cell_{cell}_power'].astype(float)

        yield df, rcp_cells


# Create a generator object from the generator function
df_generator = tsv_to_df_generator(rcp_data)

# Create a list to store the dataframes
df_list = []

cells_power_reduced = [4,5,9] 
cells_all = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
cells_power_static = list(set(cells_all) - set(cells_power_reduced))

def get_seed_stats(df, cell_set: list, sleep_fix, rcp_cells: list):
    # Get the seed value from the dataframe
    seed = df['seed'].values[0]

    # Get experiment_reduced_power_dBm from the dataframe
    experiment_reduced_power_dBm = df[f'reduced_cell_{rcp_cells[0]}_power'].values[0]

    # Get the experiment_id from the dataframe
    experiment_id = df['experiment_id'].values[0]

    # Get the rows where serving_cell_id is in cell_set
    cell_set_rows = df.loc[df['serving_cell_id'].isin(cell_set)]
    # Get the unique serving_cell_id's in the cell_set_rows and return the sc_power(dBm) for each
    df_cell_set_power_level = cell_set_rows.groupby('serving_cell_id')[
        ['serving_cell_id', 'sc_power(dBm)']
    ].first().values
    # Update column labels
    df_cell_set_power_level = pd.DataFrame(
        df_cell_set_power_level, columns=['serving_cell_id', 'sc_power(dBm)']
    )

    # Get the scalar value for the cell power level
    cell_set_power_level = df_cell_set_power_level['sc_power(dBm)'].values[0]

    # Get the sum of throughput for all UE's attached to cells in the cell_set
    T = df.loc[df['serving_cell_id'].isin(cell_set), 'ue_throughput(Mb/s)'].sum()

    # Sum the bandwidth for all cells in the cell_set
    B = len(cell_set) * 10e6

    # Sum the power of all cells in cell_set
    # Get the rows where serving_cell_id is in cell_set
    P_temp = df.loc[df['serving_cell_id'].isin(cell_set)]
    # Get the first `cell_power(kW)` for each unique serving_cell_id
    P = P_temp.groupby('serving_cell_id')['cell_power(kW)'].first().sum()
    P_min = P_temp.groupby('serving_cell_id')['cell_power(kW)'].first().min()

    if P_min <= 0.01 or pd.isna(P_min):
        # print(f'P_min is {type(P_min)}')
        P += sleep_fix
    if P < 0.01:
        print("P is less than 0.01")
    mean_cell_throughput = T / len(cell_set)
    mean_cell_power = P / len(cell_set)
    energy_efficiency = T / P
    spectral_efficiency = T / B

    return (
        experiment_id,
        experiment_reduced_power_dBm,
        seed,
        cell_set_power_level,
        mean_cell_throughput,
        mean_cell_power,
        energy_efficiency,
        spectral_efficiency,
    )


def process_seed_stats(df, cell_type, power_factor, rcp_cells, stats_list):
    (
        experiment_id,
        experiment_reduced_power_dBm,
        seed,
        cell_set_power_level,
        mean_cell_throughput,
        mean_cell_power,
        energy_efficiency,
        spectral_efficiency,
    ) = get_seed_stats(df, cell_type, power_factor, rcp_cells)
    stats_list.append(
        [
            experiment_id,
            experiment_reduced_power_dBm,
            seed,
            cell_set_power_level,
            mean_cell_throughput,
            mean_cell_power,
            energy_efficiency,
            spectral_efficiency,
        ]
    )

# Initialize empty lists for stats
cells_power_reduced_stats_per_seed_power_dBm = []
cells_power_static_stats_per_seed_power_dBm = []
cells_all_stats_per_seed_power_dBm = []

for df, rcp_cells in df_generator:
    # Process seed stats for cells_power_reduced
    process_seed_stats(
        df,
        cells_power_reduced,
        0.78 * 3,
        rcp_cells,
        cells_power_reduced_stats_per_seed_power_dBm,
    )

    # Process seed stats for cells_power_static
    process_seed_stats(
        df,
        cells_power_static,
        0.78 * 3,
        rcp_cells,
        cells_power_static_stats_per_seed_power_dBm,
    )

    # Process seed stats for cells_all
    process_seed_stats(
        df,
        cells_all,
        0.78 * 3,
        rcp_cells,
        cells_all_stats_per_seed_power_dBm,
    )


def create_dataframe(data, columns):
    unique_seed_values = len(np.unique(np.array(data)[:, 2]))
    unique_power_values = len(np.unique(np.array(data)[:, 1]))
    assert len(data) / unique_power_values == unique_seed_values, (
        f'The number of seed values does not equal the number of elements in the list '
        f'passed to this function. There are {unique_seed_values} unique seed values '
        f'in the data, but the data contains {len(data) / unique_power_values}.'
    )
    return pd.DataFrame(data, columns=columns)


# Column names for the DataFrames
column_names = [
    'experiment_id',
    'experiment_reduced_power_dBm',
    'seed',
    'cell_set_power_level',
    'mean_cell_throughput',
    'mean_cell_power',
    'energy_efficiency',
    'spectral_efficiency',
]

# Convert the stats_per_seed list to a DataFrame
df_cells_power_reduced_stats_per_seed_power_dBm = create_dataframe(
    cells_power_reduced_stats_per_seed_power_dBm,
    column_names,
)

# Convert the cells_power_static_stats_per_seed_power_dBm list to a DataFrame
df_cells_power_static_stats_per_seed_power_dBm = create_dataframe(
    cells_power_static_stats_per_seed_power_dBm,
    column_names,
)

# Convert the cells_all_stats_per_seed_power_dBm list to a DataFrame
df_cells_all_stats_per_seed_power_dBm = create_dataframe(
    cells_all_stats_per_seed_power_dBm,
    column_names,
)


def process_stats_per_power_level(df):
    grouped_df = df.groupby('experiment_reduced_power_dBm').agg(['mean', 'std'])
    grouped_df.columns = ['_'.join(col).strip() for col in grouped_df.columns.values]
    return grouped_df

# Process stats_per_power_level for df_cells_power_reduced_stats_per_seed_power_dBm
df_cells_power_reduced_stats_per_power_level_dBm = process_stats_per_power_level(
    df_cells_power_reduced_stats_per_seed_power_dBm,
)

# Process stats_per_power_level for df_cells_power_static_stats_per_seed_power_dBm
df_cells_power_static_stats_per_power_level_dBm = process_stats_per_power_level(
    df_cells_power_static_stats_per_seed_power_dBm,
)

# Process stats_per_power_level for df_cells_all_stats_per_seed_power_dBm
df_cells_all_stats_per_power_level_dBm = process_stats_per_power_level(
    df_cells_all_stats_per_seed_power_dBm,
)

# # Write the DataFrames to feather
df_cells_all_stats_per_power_level_dBm.reset_index(inplace=True)
df_cells_all_stats_per_power_level_dBm.to_feather(
    f'{project_path_str}/data/analysis/GLOBECOM/centre_plus_2_inner_scenario_all_cells.feather'
)

df_cells_power_reduced_stats_per_power_level_dBm.reset_index(inplace=True)
df_cells_power_reduced_stats_per_power_level_dBm.to_feather(
    f'{project_path_str}/data/analysis/GLOBECOM/centre_plus_2_inner_scenario_variable_cells.feather'
)

df_cells_power_static_stats_per_power_level_dBm.reset_index(inplace=True)
df_cells_power_static_stats_per_power_level_dBm.to_feather(
    f'{project_path_str}/data/analysis/GLOBECOM/centre_plus_2_inner_scenario_static_cells.feather'
)

################
### PLOTTING ###
################


# Create figure and axes
fig, ax = plt.subplots(2, 2, figsize=(12, 10))

# Create a string of the reduced cells joined by ' and '
reduced_cells_title_str = ' and '.join([str(cell) for cell in rcp_cells])

# Set figure title
fig.suptitle(f'Impact of reducing cell {reduced_cells_title_str} output power (dBm) on the network', fontsize=16)

# Reset indexes
df_cells_power_reduced_stats_per_power_level_dBm.reset_index(inplace=True)
df_cells_power_static_stats_per_power_level_dBm.reset_index(inplace=True)
df_cells_all_stats_per_power_level_dBm.reset_index(inplace=True)

# Set the `cell_set_power_level` column where the value is -inf to -2
df_cells_power_static_stats_per_power_level_dBm.loc[
    df_cells_power_static_stats_per_power_level_dBm['experiment_reduced_power_dBm'] == -np.inf,
    'experiment_reduced_power_dBm'
] = -2

df_cells_power_reduced_stats_per_power_level_dBm.loc[
    df_cells_power_reduced_stats_per_power_level_dBm['experiment_reduced_power_dBm'] == -np.inf,
    'experiment_reduced_power_dBm'
] = -2

df_cells_all_stats_per_power_level_dBm.loc[
    df_cells_all_stats_per_power_level_dBm['experiment_reduced_power_dBm'] == -np.inf,
    'experiment_reduced_power_dBm'
] = -2


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
    
# Dodge the error bars
dodge = 0.3

df_cells_power_reduced_stats_per_power_level_dBm['experiment_reduced_power_dBm'] = df_cells_power_reduced_stats_per_power_level_dBm['experiment_reduced_power_dBm'] - dodge

df_cells_power_static_stats_per_power_level_dBm['experiment_reduced_power_dBm'] = df_cells_power_static_stats_per_power_level_dBm['experiment_reduced_power_dBm'] + dodge

# Define the dataset parameters
datasets = [
    (df_cells_all_stats_per_power_level_dBm, 'All Network Cells', 's'),
    (df_cells_power_reduced_stats_per_power_level_dBm, 'Reduced Power Cells', '^'),
    (df_cells_power_static_stats_per_power_level_dBm, 'Static Power Cells', '.')
]


# Define the dataset parameters
datasets = [
    (df_cells_all_stats_per_power_level_dBm, 'All Network Cells', 's'),
    (df_cells_power_reduced_stats_per_power_level_dBm, 'Reduced Power Cells', '^'),
    (df_cells_power_static_stats_per_power_level_dBm, 'Static Power Cells', '.')
]


#######################
### CELL THROUGHPUT ###
#######################

x_data_series = 'experiment_reduced_power_dBm'
y_data_series = 'mean_cell_throughput_mean'
yerr_series = 'mean_cell_throughput_std'

for data, label, fmt in datasets:
    plot_errorbar(ax = ax[0,0],
                              x = data[x_data_series],
                              y = data[y_data_series],
                              yerr = data[yerr_series],
                              fmt=fmt,
                              label=label)
    
ax[0,0].set_ylabel('Throughput Mean (Mb/s)')


######################
### CELL POWER(kW) ###
######################

x_data_series = 'experiment_reduced_power_dBm'
y_data_series = 'mean_cell_power_mean'
yerr_series = 'mean_cell_power_std'

for data, label, fmt in datasets:
    plot_errorbar(ax = ax[0,1],
                              x = data[x_data_series],
                              y = data[y_data_series],
                              yerr = data[yerr_series],
                              fmt=fmt,
                              label=label)

ax[0,1].set_ylabel('Power Consumption Mean (kW)')


########################
### CELL EE (bits/J) ###
########################

x_data_series = 'experiment_reduced_power_dBm'
y_data_series = 'energy_efficiency_mean'
yerr_series = 'energy_efficiency_std'

for data, label, fmt in datasets:
    plot_errorbar(ax = ax[1,0],
                              x = data[x_data_series],
                              y = data[y_data_series],
                              yerr = data[yerr_series],
                              fmt=fmt,
                              label=label)
    
ax[1,0].set_ylabel('Energy Efficiency Mean (bits/J)')


#########################
### CELL SE (bits/Hz) ###
#########################

x_data_series = 'experiment_reduced_power_dBm'
y_data_series = 'spectral_efficiency_mean'
yerr_series = 'spectral_efficiency_std'

for data, label, fmt in datasets:
    plot_errorbar(ax = ax[1,1],
                              x = data[x_data_series],
                              y = data[y_data_series],
                              yerr = data[yerr_series],
                              fmt=fmt,
                              label=label)

ax[1,1].set_ylabel('Spectral Efficiency Mean (bits/Hz)')


#######################
### Plot formatting ###
#######################

# Auto adjust the padding between subplot
fig.tight_layout(pad=3.0)

# For all sub-plots
for ax_row in ax:
    for ax in ax_row:
        # Set the x axis label
        ax.set_xlabel('Reduced Cell Tx Power (dBm)')
        # Set x and y axis lower limits
        ax.set_xlim(left=-3, right=45)
        ax.set_ylim(bottom=0)
        # Enable minor ticks
        ax.minorticks_on()
        # Set the grid on
        ax.grid(linestyle='--', linewidth=0.5,)
        # Enable the figure legend
        ax.legend(fontsize=6)


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

# Create a string of the reduced cells joined by '_and_'
reduced_cells_str = '_and_'.join([str(cell) for cell in rcp_cells])

fig_filename = f'{date_str}_r3acp_cell_{reduced_cells_str}_power_dBm_v3_0.pdf'
fig_path = fig_dir / fig_filename
fig.savefig(
    fig_path,
    dpi=300,
    format='pdf',
    bbox_inches='tight',
)


