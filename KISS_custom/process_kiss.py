import argparse
import os
from pathlib import Path

import pandas as pd

from logging_kiss import get_logger
from utils_kiss import get_timestamp

def read_files(directory, file_starts_with):
    """
    Read all TSV files in a directory that begin with a specified string and 
    return a list of DataFrames.
    """
    # If the directory is a string, convert it to a Path object
    if isinstance(directory, str):
        # If the directory is a string, convert it to a Path object
        directory = Path(directory)
        # check if the directory is a relative path
        if not directory.is_absolute():
            # If it is a relative path, make it an absolute path
            directory = directory.resolve()

    # Check that the directory exists
    directory = directory.resolve()

    # Read in the files
    files = [pd.read_csv(os.path.join(directory, filename), delimiter='\t')
             for filename in os.listdir(directory)
             if filename.endswith('.tsv') and 
                filename.startswith(file_starts_with)]
    if callable(logger.info):
        logger.info(f'Read {len(files)} files.')
    return files


def concat_dataframes(dataframes):
    """
    Concatenate a list of DataFrames and return a single DataFrame.
    """
    logger.info('Concatenating dataframes.')
    master_df = pd.concat(dataframes, ignore_index=True)
    logger.info('Sorting data by seed, serving_cell_id, and ue_id.')
    master_df = master_df.sort_values(['seed', 'serving_cell_id', 'ue_id'])
    logger.info('Finished processing dataframes.')
    return master_df


def write_dataframe(df: pd.DataFrame, path: str, outfile: str = None,
                    file_type: str = 'fea', logger=None, **kwargs):
    """
    Write a Pandas DataFrame to a file.
    """
    if path is None:
        # If no path is specified, use the current working directory
        path = Path.cwd()
    if outfile is None:
        # If no output file name is specified, use the name of the current script
        outfile = "_".join(["test", str(Path(__file__).stem)])

    outpath = "/".join([path, f'{outfile}_{get_timestamp()}.{file_type}'])
    outpath = outpath.replace(':', '_').replace('-', '_')
    if file_type == 'fea':
        df = df.reset_index()
        df.to_feather(outpath, **kwargs)
    elif file_type == 'csv':
        df.to_csv(outpath, index=True)
    elif file_type == 'tsv':
        df.to_csv(outpath, index=True, sep='\t')
    else:
        raise ValueError(f'Invalid file type: {file_type}')
    logger.info(f'DataFrame written to {outpath}.')


def main(directory, file_starts_with, logging_enabled=True, 
         outfile=None, outfile_path=None, file_type='csv'):

    # Set up logging
    if logging_enabled:
        
        logger.info('Starting script.')
        logger.info(f'Processing files in directory: {directory}')
        logger.info(f'Output file type: {file_type}')
        logger.info(f'Output file name: {outfile}')

    # Read in the TSV files and process them
    files = read_files(directory, file_starts_with)
    master_df = concat_dataframes(files)

    # Write the output file
    write_dataframe(df=master_df, 
                    path=outfile_path, 
                    outfile=outfile, 
                    file_type=file_type, 
                    logger=logger)

    if logging_enabled:
        logger.info('Finished processing files.')

    return master_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--no-logging', 
        action='store_false', 
        help='Turn off logging'
    )
    parser.add_argument(
        '--dir', 
        type=str, 
        help='Directory where files are located', 
        default='KISS/_test/data/input/process_kiss'
    )
    parser.add_argument(
        '--file_starts_with', 
        type=str, 
        help='String that all files start with', 
        default='test_'
    )
    parser.add_argument(
        '--outfile', 
        type=str, 
        help='Name of output file'
    )
    parser.add_argument(
        '--outfile_path',
        type=str,
        help='Path to output file',
        default='KISS/_test/data/output/process_kiss'
    )
    parser.add_argument(
        '--file_type', 
        type=str, 
        help='File type of output file (csv, fea, tsv)', 
        default='fea'
    )
    args = parser.parse_args()

    script_name = Path(__file__).resolve().stem
    if args.file_starts_with == 'test_':
        script_name = f'test_{script_name}'
    logger = get_logger(logger_name=script_name, logfile_path=args.outfile_path)

    main(
        directory=args.dir, 
        file_starts_with=args.file_starts_with, 
        logging_enabled=args.no_logging, 
        outfile=args.outfile,
        outfile_path=args.outfile_path, 
        file_type=args.file_type
    )



