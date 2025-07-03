import json
import os
from datetime import datetime

def dump_json(config_dict_list):
    # Create a list to store the absolute paths of the JSON files
    json_file_list = []

    # Loop through each dictionary in the list
    for config_dict in config_dict_list:
        # Get the current timestamp
        timestamp = datetime.now()
        timestamp_int = int(timestamp.timestamp()*1e6)
        timestamp_string = str(timestamp_int)

        # Create a temporary JSON file
        temp_json = f'temp{timestamp_string}.json'
        with open(temp_json, 'w') as f:
            json.dump(config_dict, f, indent=4)

            # Get the absolute file path of the temporary file
            temp_json_abs_path = os.path.abspath(temp_json)

            # Append the absolute file path of the temporary file to the list
            json_file_list.append(temp_json_abs_path)

    # Return the list of absolute paths of the JSON files
    return json_file_list

def remove_temp_json(json_file_list):
    """
    Takes a list of temp JSON file paths and removes them from the file system
    """
    # Loop through each JSON file in the list
    for json_file in json_file_list:
        # Remove the JSON file from the file system
        try:
            os.remove(json_file)
            # test if file exists
            if os.path.exists(json_file):
                # tried removing and it still exists, so raise an error by moving to the except block below
                raise OSError
            else:
                # Get the json file name
                json_file_name = os.path.basename(json_file)
                print(f"File {json_file_name} removed successfully")
        except OSError as e:
            print(f"Error deleting file: {e}")