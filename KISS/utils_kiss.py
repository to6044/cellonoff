from pathlib import Path
from pandas import Timestamp
from typing import Optional

def get_timestamp(date_only: Optional[bool] = False, time_only: Optional[bool] = False, for_seed: Optional[bool] = False) -> str:
    """Return the current timestamp in ISO 8601 format.

    Args:
        date_only (bool): If True, return only the date in YYYY-MM-DD format.
        time_only (bool): If True, return only the time in HHMMSS format.
        for_seed (bool): If True, return the timestamp as an integer for use as a seed value in the format HHMMSS.

    Returns:
        str: The current timestamp.
    """
    if date_only:
        return Timestamp.now(tz=None).strftime('%Y_%m_%d')
    elif time_only:
        return Timestamp.now(tz=None).strftime('%H%M%S')
    elif for_seed:
        return int(Timestamp.now(tz=None).strftime('%H%M%S'))
    else:
        return Timestamp.now(tz=None).isoformat(timespec='seconds')

def validate_str_path(directory: str) -> str:
    # Accepts a path as a string or pathlib.Path object. Checks if it is a valid path and returns a pathlib.Path object.

    # If the directory is a string, convert it to a Path object
    if isinstance(directory, str):
        # If the directory is a string, convert it to a Path object
        directory = Path(directory)
        # check if the directory is a relative path
        if not directory.is_absolute():
            # If it is a relative path, make it an absolute path
            directory = directory.resolve()

    # Check that the directory exists
    if directory.resolve() is None:
        # If the directory does not exist, return False
        return False
    else:
        # If the directory does exist, return the directory as a Path object
        return directory