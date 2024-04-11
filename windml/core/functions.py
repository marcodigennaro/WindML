import pandas as pd


def select_time_subset(read_df, year=None, month=None, hour=None, date_column='Date_time'):
    """
    Selects a subset of a DataFrame based on year, month, or hour.

    Parameters:
    - read_df: DataFrame containing datetime values.
    - year: Integer representing the year (e.g., 2022) to select. Default is None.
    - month: Integer representing the month (1-12) to select. Default is None.
    - hour: Integer representing the hour (0-23) to select. Default is None.
    - date_column: Name of the column containing the datetime values. Default is 'Date'.

    Returns:
    - DataFrame: Subset of the original DataFrame based on the provided criteria.
    """

    subset = read_df.copy()

    if year is not None:
        subset = subset.loc[subset[date_column].dt.year == year]

    if month is not None:
        subset = subset.loc[subset[date_column].dt.month == month]

    if hour is not None:
        subset = subset.loc[subset[date_column].dt.hour == hour]

    return subset


import os
import time
from memory_profiler import memory_usage
import pandas as pd
import dask.dataframe as dd
import vaex
import modin.pandas as mpd


def read_and_concatenate_csv(folder_path):
    libraries = ['pandas', 'dask', 'vaex', 'modin']
    dtypes = {'Date_time': 'object', 'Date_time_nr': 'int64', 'Wind_turbine_name': 'object'}

    for library in libraries:
        start_time = time.time()
        csv_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]

        if library == 'pandas':
            dataframes = [pd.read_csv(file, dtype=dtypes) for file in csv_files]
            df = pd.concat(dataframes)
        elif library == 'dask':
            dataframes = [dd.read_csv(file, dtype=dtypes) for file in csv_files]
            df = dd.concat(dataframes).compute()
        elif library == 'vaex':
            df = vaex.open_many(csv_files)
        elif library == 'modin':
            dataframes = [mpd.read_csv(file, dtype=dtypes) for file in csv_files]
            df = mpd.concat(dataframes)

        end_time = time.time()
        max_memory = max(memory_usage((lambda: None, ())))
        print(
            f"Library: {library}, Time taken: {end_time - start_time:.2f} seconds, Max memory usage: {max_memory:.2f} MB")

# Example usage (replace 'your_folder_path' with the actual path):
# read_and_concatenate_csv('your_folder_path')
