# Script to transform the data to chunks that are easier to work with

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
data_path = project_path / 'data' / 'output' / 'reduce_centre_cell_power' / '2023_04_12' / 'sinr_cell_to_zero_watts'

######################
###### Load Data #####
######################

# Each .tsv file is an experiment run.
# For each experiment, we change the power of the centre cell.
# Every row in the .tsv file tells us the relationship between an individual UE and a cell.
# For each UE, measurements are listed in the remaining columns.
# We repeat the experiment at a given power level 100 times (i.e. 100 seed values).
# We repeat the experiment at 16 different power levels.

def load_data_to_dict(data_path):
    """Load the data from the data_path into a dictionary of dataframes."""
    dfs = {}
    for path in data_path.glob('*.feather'):
        # if the file is called 'data.feather', then skip it
        if path.name == 'data.feather':
            continue
        dfs[path.stem] = pd.read_feather(path)
    return dfs

dfs = load_data_to_dict(data_path) # dictionary of dataframes


#######################
###### Cell sets ######
#######################

all_cells = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
var_pwr_cells = [9] 
static_pwr_cells = list(set(all_cells) - set(var_pwr_cells))

sets = {
    'all_cells': all_cells,
    'var_pwr_cells': var_pwr_cells,
    'static_pwr_cells': static_pwr_cells
}


#######################
### Define Metrics ####
#######################

def validate_rows(df):
    """
    Validates that the rows in the dataframe are correct.
    Any rows where the ue_id or cell_throughput are -2 should be removed.
    """

    # Get the rows where the ue_id or cell_throughput are -2
    df = df.loc[df['ue_id'] != -2]
    df = df.loc[df['sc_tp_Mbps'] != -2]

    return df


def get_ue_count(df, seed, cell_id, exp_cell_pwr_dBm):
    """Get the number of UEs attached to a given cell at a given experiment power level, for a given seed."""

    df = df.loc[df['sc_id'] == cell_id]
    df = df.loc[df['seed'] == seed]
    df = df.loc[df['exp_cell_pwr_dBm'] == exp_cell_pwr_dBm]

    df = validate_rows(df)

    return df.shape[0]


def get_ue_count_per_cell(df, seed, cell_set, exp_cell_pwr_dBm):
    """
    Get the number of UEs attached to each cell in a given cell set at a 
    given experiment power level, for a given seed value.
    """
    ue_count_per_cell = {}
    for cell_id in cell_set:
        ue_count_per_cell[cell_id] = get_ue_count(df, seed, cell_id, exp_cell_pwr_dBm)
    return ue_count_per_cell


def get_cell_throughput(df, seed, cell_id, exp_cell_pwr_dBm):
    """
    Get the cell throughput for a given cell at a given experiment power level for a given seed value.
    Filters the dataframe. Then adds up the throughput for each UE attached to the cell.
    """
    df = df.loc[df['sc_id'] == cell_id]
    df = df.loc[df['seed'] == seed]
    df = df.loc[df['exp_cell_pwr_dBm'] == exp_cell_pwr_dBm]
    df = validate_rows(df)
    cell_throughput = df['ue_tp_Mbps'].sum()
    return cell_throughput


def get_cell_throughput_per_cell(df, seed, cell_set, exp_cell_pwr_dBm):
    """
    Get the cell throughput for each cell in a given cell set at a given experiment power level for a given seed value.
    """
    cell_throughput_per_cell = {}
    for cell_id in cell_set:
        cell_throughput_per_cell[cell_id] = get_cell_throughput(df, seed, cell_id, exp_cell_pwr_dBm)
    return cell_throughput_per_cell


def get_cell_ec(df, seed, cell_id, exp_cell_pwr_dBm):
    """
    Get the cell energy consumption for a given cell at a given experiment power level for a given seed value.
    Filters the dataframe. Then adds up the energy consumption for each UE attached to the cell.
    """
    df = df.loc[df['sc_id'] == cell_id]
    df = df.loc[df['seed'] == seed]
    df = df.loc[df['exp_cell_pwr_dBm'] == exp_cell_pwr_dBm]
    df = validate_rows(df)
    cell_ec = df['sc_ec_kW'].unique()[0]
    return cell_ec


def get_cell_ec_per_cell(df, seed, cell_set, exp_cell_pwr_dBm):
    """
    Get the cell energy consumption for each cell in a given cell set at a given experiment power level for a given seed value.
    """
    cell_ec_per_cell = {}
    for cell_id in cell_set:
        cell_ec_per_cell[cell_id] = get_cell_ec(df, seed, cell_id, exp_cell_pwr_dBm)
    return cell_ec_per_cell

def get_cell_throughputs(df, cell_id, exp_cell_pwr_dBm):
    """
    At every experiment power level, each cell throughput is measured 100 times (with different seed values).
    This function returns a list of the cell throughputs for a given cell at a given experiment power level.
    """
    df = df.loc[df['sc_id'] == cell_id]
    df = df.loc[df['exp_cell_pwr_dBm'] == exp_cell_pwr_dBm]
    df = validate_rows(df)
    df = df.groupby(['seed'])[['sc_tp_Mbps']].first()
    return df


def get_cell_throughputs_v2(df, cell_id, exp_cell_pwr_dBm):
    """
    At every experiment power level, each cell throughput is measured 100 times (with different seed values).
    This function returns a list of the cell throughputs for a given cell at a given experiment power level.
    """
    df = df.loc[df['sc_id'] == cell_id]
    df = df.loc[df['exp_cell_pwr_dBm'] == exp_cell_pwr_dBm]
    df = validate_rows(df)
    df = df.groupby(['seed'])[['ue_tp_Mbps']].sum()
    return df

def get_cell_ecs(df, cell_id, exp_cell_pwr_dBm):
    """
    At every experiment power level, each cell energy consumption is measured 100 times (with different seed values).
    This function returns a list of the cell energy consumptions for a given cell at a given experiment power level.
    """
    df = df.loc[df['sc_id'] == cell_id]
    df = df.loc[df['exp_cell_pwr_dBm'] == exp_cell_pwr_dBm]
    df = validate_rows(df)
    df = df.groupby(['seed'])[['sc_ec_kW']].first()
    return df


p40_c3_tp = get_cell_throughputs_v2(dfs['all_cells'], 3, 40)
p40_c3_ec = get_cell_ecs(dfs['all_cells'], 3, 40)
p40_c3 = pd.concat([p40_c3_tp, p40_c3_ec], axis=1)
p40_c3.columns = ['tp_Mbps', 'ec_kW']
p40_c3['ee_bps_J'] = p40_c3['tp_Mbps'] / p40_c3['ec_kW'] * 1e6
p40_c3['se_bps_Hz'] = p40_c3['tp_Mbps'] * 1e6 / 10e6



def add_cell_summary(dictionary, dfs, cell_id, exp_cell_pwr_dBm):
    """
    Add a summarised dataframe for a given cell at a given experiment power to a dictionary.
    """
    if exp_cell_pwr_dBm not in dictionary:
        dictionary[exp_cell_pwr_dBm] = {}
    
    if cell_id not in dictionary[exp_cell_pwr_dBm]:
        dictionary[exp_cell_pwr_dBm][cell_id] = pd.DataFrame()

    tp = get_cell_throughputs_v2(dfs, cell_id, exp_cell_pwr_dBm)
    ec = get_cell_ecs(dfs, cell_id, exp_cell_pwr_dBm)
    df = pd.concat([tp, ec], axis=1)
    df.columns = ['tp_Mbps', 'ec_kW']
    df['ee_bps_J'] = df['tp_Mbps'] / df['ec_kW'] * 1e6
    df['se_bps_Hz'] = df['tp_Mbps'] * 1e6 / 10e6
    
    dictionary[exp_cell_pwr_dBm][cell_id] = pd.concat([dictionary[exp_cell_pwr_dBm][cell_id], df])

    return dictionary



cell_summary = {}

# Iterate over all cells and experiment power levels
for exp_cell_pwr_dBm in range(1, 46, 3):
  for cell_id in range(19):  # Adjusted range from 0 to 18
        cell_summary = add_cell_summary(cell_summary, dfs['all_cells'], cell_id, exp_cell_pwr_dBm)


import matplotlib.pyplot as plt
import numpy as np

# Initialize lists to store mean and standard deviation values
stats = []

# Iterate over exp_cell_pwr_dBm values
for exp_cell_pwr_dBm, cell_data in cell_summary.items():
    
    # Iterate over cell_id values
    for cell_id, df in cell_data.items():
        stats.append([exp_cell_pwr_dBm, cell_id, df['tp_Mbps'].mean(), df['tp_Mbps'].std()])



# Create the plot
x_values = range(1, 46, 3)  # exp_cell_pwr_dBm values
plt.errorbar(stats[0], stats[2], yerr=stats[3], fmt='o', capsize=5)
plt.xlabel('Experiment Power (exp_cell_pwr_dBm)')
plt.ylabel('Throughput (tp_Mbps)')
plt.title('Mean and Standard Deviation of Throughput')
plt.show()





print(func_dict)
print('This is the way...')
