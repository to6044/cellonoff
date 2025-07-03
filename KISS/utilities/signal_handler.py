# signal_handler.py

from utilities.json_handler import remove_temp_json

def signal_handler(signum, json_file_list):
    # Print a message to the console
    print(f"Received signal {signum}, exiting...")

    # Remove the temporary JSON files
    remove_temp_json(json_file_list)

    # Exit the program
    exit(1)
