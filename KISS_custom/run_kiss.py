import argparse
from datetime import datetime
import json
import os
import subprocess
import time
import numpy as np
import signal


def signal_handler(signum, frame):
    """
    Signal handler function that is called when a signal is received.

    Args:
        signum (int): The signal number.
        frame (frame): The current stack frame.

    Returns:
        None
    """

    print(f"Received signal {signum}, exiting...")
    remove_temp_json(json_file_list)
    exit(1)


def generate_config_dict_list(config_file):
    """
    Generate a list of dictionaries containing different configurations based on the input JSON file.

    Args:
        config_file (str): The path to the JSON file containing the configuration.

    Returns:
        list: A list of dictionaries, where each dictionary represents a different configuration.
    """

    # Load the contents of the JSON file into a dictionary
    with open(config_file) as f:
        config = json.load(f)

    seed_max = config['seed']
    power_dBm = config['power_dBm']
    power_dBm_end = config['power_dBm_end']
    power_dBm_step = config['power_dBm_step']

    # Define the range of power values to test
    if power_dBm == power_dBm_end:
        power_values = [power_dBm]
    # If the finish power value is `-inf` then the range of power values will be from the start power value to `0dBm` in steps of `power_dBm_step`, and then an additional value of `-np.inf` will be added to the end of the list
    elif power_dBm_end == "-np.inf":
        power_values = np.arange(power_dBm, 0, -power_dBm_step)
        power_values = np.append(power_values, -np.inf)
    else:
        power_values = np.arange(power_dBm, power_dBm_end-power_dBm_step, -power_dBm_step)

    # Seed values to test 
    seeds = list(range(seed_max))

    # Define the number range of variable power cells
    n_variable_power_cells = list(range(1, config['n_variable_power_cells']+1))

    # Create a list of dictionaries
    config_dict_list = []
    for seed in seeds:
        for power_dBm in power_values:
            if len(n_variable_power_cells) == 0:
                # Create a new dictionary object for each iteration
                config_copy = config.copy()
                
                # Update the seed and power_dBm value in the new dictionary object
                config_copy['seed'] = seed
                config_copy['variable_cell_power_dBm'] = power_dBm
                config_copy['n_variable_power_cells'] = 0
                
                # Append the new dictionary object to the list
                config_dict_list.append(config_copy)
            else:
                # Create a new dictionary object for each iteration
                for variable_power_cell in n_variable_power_cells:
                    # Create a new dictionary object for each iteration
                    config_copy = config.copy()
                    
                    # Update the seed and power_dBm value in the new dictionary object
                    config_copy['seed'] = seed
                    config_copy['variable_cell_power_dBm'] = power_dBm
                    config_copy['n_variable_power_cells'] = variable_power_cell
                    
                    # Append the new dictionary object to the list
                    config_dict_list.append(config_copy)

    # Return the list of dictionaries
    return config_dict_list



# Define a function that dumps all dictionaries in a list to an individual JSON files and return the absolute path of the JSON file as a string
def dump_json(config_dict_list):
    # Create a list to store the absolute paths of the JSON files
    json_file_list = []

    # Loop through each dictionary in the list
    for dict in config_dict_list:
        # Get the current timestamp
        timestamp = datetime.now()
        timestamp_int = int(timestamp.timestamp()*1e6)
        timestamp_string = str(timestamp_int)

        # Create a temporary JSON file
        temp_json = f'temp{timestamp_string}.json'
        with open(temp_json, 'w') as f:
            json.dump(dict, f, indent=4)

            # Get the absolute file path of the temporary file
            temp_json_abs_path = os.path.abspath(temp_json)

            # Append the absolute file path of the temporary file to the list
            json_file_list.append(temp_json_abs_path)

    # Return the list of absolute paths of the JSON files
    return json_file_list


def run_kiss(temp_json_file):
    # Build the command to run the kiss.py script with the temp_json as an argument
    command = ["python", "kiss.py", "-c", f"{temp_json_file}"]

    # Execute the command as a subprocess
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Print the output of the subprocess
    print(process.stdout.decode())
    if process.returncode != 0:
        print(process.stderr.decode())
    else:
        return process.returncode

    



# Define a function that takes a list of temp JSON file paths and removes them from the file system
def remove_temp_json(json_file_list):
    # Loop through each JSON file in the list
    for json_file in json_file_list:
        # Remove the JSON file from the file system
        try:
            os.remove(json_file)
            # test if file exists
            if os.path.exists(json_file):
                # tried removing and it still exists, so raise an error by moving to the exept block below
                raise OSError
            else:
                # Get the json file name
                json_file_name = os.path.basename(json_file)
                print(f"File {json_file_name} removed successfully")
        except OSError as e:
            print(f"Error deleting file: {e}")
 



import multiprocessing as mp

if __name__ == '__main__':
    # Register the signal handler function to handle the SIGINT and SIGTERM signals
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Parse the command line arguments
    parser = argparse.ArgumentParser(
        description='Run kiss.py against a specified config value.')

    parser.add_argument(
        '-c',
        '--config-file',
        type=str,
        required=True,
        default='_test/data/input/kiss/test_kiss.json'
    )
    
    args = parser.parse_args()

    # Start the timer
    start_time = time.time()

    config_dict_list = generate_config_dict_list(args.config_file)

    # Dump the list of dictionaries to individual JSON files
    json_file_list = dump_json(config_dict_list)

    # Initialize a multiprocessing pool with 6 processes and 1 task per child
    num_processes = 6
    num_tasks_per_child = 1
    progress_counter = mp.Value('i', 0)  # Shared variable to keep track of progress

    with mp.Pool(processes=num_processes, maxtasksperchild=num_tasks_per_child) as pool:
        # Run the run_kiss function on each JSON file in the list
        for json_file in json_file_list:
            pool.apply(run_kiss, args=(json_file,)) 

            # Increment the progress counter by 1
            with progress_counter.get_lock():
                progress_counter.value += 1
                print(f"Processed {progress_counter.value} of {len(json_file_list)} files")

        # Close the pool and wait for all processes to complete
        pool.close()
        pool.join()

    # Remove the temporary JSON files
    remove_temp_json(json_file_list)

    # Stop the timer
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the elapsed time
    print(f"All subprocesses completed in {elapsed_time:.2f} seconds")

