# Data pre-processing script

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

cells_turned_down = []

######################
###### Read data #####
######################

# Each .tsv file is an experiment run.
# For each experiment, we change the power of the centre cell.
# Every row in the .tsv file tells is the relationship between an individual UE and a cell.
# For each UE, measurements are listed in the remaining columns.
# We repeat the experiment at a given power level 100 times (i.e. 100 seed values).
# We repeat the experiment at 16 different power levels.

def read_data(dir_path):
    """
    Read in the data from the directory path.
    """
    # Create placeholder for a dataframe
    df_temp_list = None    

    # Iterate over the files in the directory
    for f in dir_path.glob('*.tsv'):
            df = pd.read_csv(f, sep='\t')
            # Add a column for the experiment id
            exp_id = f.stem
            df['experiment_id'] = exp_id

            # Add a column for the set of cells being experimented on
            exp_cell_set = [9]
            df['exp_cell_set'] = str(exp_cell_set)

            # Add a column for the power level of the experiment
            # If the experiment id contains 'p_'
            if '_p_inf' in exp_id:
                  # Then the power level is -inf dBm, but we want to plot it as -2 dBm
                  df['variable_power_dBm'] = -2
            else:
                  # Otherwise, find the substring that begins with 'p'
                  p_string = exp_id[exp_id.find('_p'):]
                  # Split on the underscore
                  p_string_split = p_string.split('_')
                  # The power level is the second element in the list
                  exp_power_dBm = int(p_string_split[1].replace('p', ''))
                  df['variable_power_dBm'] = exp_power_dBm

            for cell in exp_cell_set:
                  # Get the rows where serving_cell_id is equal to cell and fill the NaNs with -2
                  df.loc[df['serving_cell_id'] == cell] = df.loc[df['serving_cell_id'] == cell].fillna(-2)

            # Append the dataframe to the list
            if df_temp_list is None:
                  df_temp_list = [df]
            else:
                  df_temp_list.append(df)

    # Concatenate the dataframes
    df = pd.concat(df_temp_list)

    # Reorder the columns to the order in cols_order
    cols_order = [
          'time',
          'experiment_id',
          'exp_cell_set',
          'variable_power_dBm',
          'seed',
          'ue_id',
          'serving_cell_id',
          'sc_power(dBm)',
          'sc_power(watts)',
          'sc_rsrp(dBm)',
          'distance_to_cell(m)',
          'noise_power(dBm)',
          'sinr(dB)',
          'cqi',
          'mcs',
          'ue_throughput(Mb/s)',
          'cell_throughput(Mb/s)',
          'cell_power(kW)',
          'cell_ee(bits/J)',
          'cell_se(bits/Hz)',
          'neighbour1_rsrp(dBm)',
          'neighbour2_rsrp(dBm)',
          'serving_cell_sleep_mode'
    ]
    df = df.reindex(columns=cols_order)

    # Update the column names
    cols_names = [
          'time',
          'exp_id',
          'exp_cell_set',
          'exp_cell_pwr_dBm',
          'seed',
          'ue_id',
          'sc_id',
          'sc_pwr_dBm',
          'sc_pwr_W',
          'ue_rsrp_dBm',
          'ue_dist_m',
          'noise_pwr_dBm',
          'sinr_dB',
          'cqi',
          'mcs',
          'ue_tp_Mbps',
          'sc_tp_Mbps',
          'sc_ec_kW',
          'sc_ee_bps_J',
          'sc_se_bps_Hz',
          'nbr1_rsrp_dBm',
          'nbr2_rsrp_dBm',
          'sc_sleep_mode'
    ]
    df.columns = cols_names

    return df

data = read_data(data_path)

# Write to a feather file
data.reset_index(drop=True, inplace=True)
data.to_feather(data_path / 'data.feather')

print('finished reading data')