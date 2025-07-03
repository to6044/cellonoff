import argparse
from pathlib import Path
import pandas as pd

from logging_kiss import get_logger
from process_kiss import write_dataframe

# Get current script name and current working directory
script_name = Path(__file__).stem
cwd = Path.cwd()

# Create a logger object
logger = get_logger(logger_name=script_name, log_dir=cwd)

def analyze_data(data_file, show_results=True):
    # Read in the data from the feather file
    logger.info(f'Reading data from {data_file}...')
    df = pd.read_feather(data_file)
    df = df.reset_index(drop=True)

    # Define the columns and metrics to analyze
    columns = ['cell_throughput(Mb/s)', 'cell_power(kW)', 'cell_ee(bits/J)', 'cell_se(bits/Hz)']
    metrics = ['mean', 'std']


    logger.info('Starting data analysis...')

    # Calculate the mean and standard deviation for each column
    results = []
    for col in columns:
        col_stats = df.groupby(['serving_cell_id', 'sc_power(dBm)'])[col].agg(metrics)
        col_stats.columns = [f'{col}_{metric}' for metric in metrics]
        results.append(col_stats)

    # Combine the results into a single DataFrame
    stats_df = pd.concat(results, axis=1)

    logger.info('Data analysis complete.')

    # Print the results
    if show_results:
        print(stats_df)
    
    # Return the results
    return stats_df


def main(data_file_path, outfile_path=None):

    # Call the analyze_data function with the specified data file
    summary_stats = analyze_data(data_file_path)

    # Save the results to a tsv file
    logger.info('Saving results to file...')
    path=Path(data_file_path).parent
    outfile_name=str(Path(data_file_path).stem) + '_summary_stats'
    write_dataframe(
          df=summary_stats,
          logger=logger,
          path=outfile_path, 
          outfile=outfile_name,
            file_type='tsv'
        )


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_file_path', 
        type=str, 
        help='Path to feather file',
        default='KISS/_test/data/input/data_analysis/test_s100_p43dBm.fea'
    )
    parser.add_argument(
        '--outfile_path',
        type=str,
        help='Path to output file',
        default='KISS/_test/data/output/data_analysis'
    )
    args = parser.parse_args()

    # Call the main function
    main(data_file_path=args.data_file_path, outfile_path=args.outfile_path)
