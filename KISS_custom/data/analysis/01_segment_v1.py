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
data_path = project_path / 'data' / 'output' / 'reduce_centre_cell_power' / '2023_04_12' / 'sinr_cell_to_zero_watts' / 'data.feather'


######################
###### Load Data #####
######################

# Each .tsv file is an experiment run.
# For each experiment, we change the power of the centre cell.
# Every row in the .tsv file tells is the relationship between an individual UE and a cell.
# For each UE, measurements are listed in the remaining columns.
# We repeat the experiment at a given power level 100 times (i.e. 100 seed values).
# We repeat the experiment at 16 different power levels.

data = pd.read_feather(data_path)

#######################
###### Cell sets ######
#######################
all_cells = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
var_pwr_cells = [9] 
static_pwr_cells = list(set(all_cells) - set(var_pwr_cells))

sets = [all_cells, var_pwr_cells, static_pwr_cells]

#######################
#### Segment Data #####
#######################

df_all = data[data['sc_id'].isin(all_cells)].copy()
df_var_pwr = data[data['sc_id'].isin(var_pwr_cells)].copy()
df_static_pwr = data[data['sc_id'].isin(static_pwr_cells)].copy()
dfs = [df_all, df_var_pwr, df_static_pwr]

#######################
#### Store Results ####
#######################

df_all.reset_index().to_feather(data_path.parent / 'all_cells.feather')
df_var_pwr.reset_index().to_feather(data_path.parent / 'var_pwr_cells.feather')
df_static_pwr.reset_index().to_feather(data_path.parent / 'static_pwr_cells.feather')





