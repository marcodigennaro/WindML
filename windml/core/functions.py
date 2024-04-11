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


def compare_data_libraries(folder_path):
    libraries = ['pandas', 'dask', 'vaex', 'modin']
    dtypes = {'Date_time': 'object', 'Date_time_nr': 'int64', 'Wind_turbine_name': 'object'}

    # Gather CSV files
    csv_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if
                 file.endswith('.csv') and file.startswith('R')]

    # Log file details
    print(f"Found {len(csv_files)} CSV files.")
    for file in csv_files:
        size = os.path.getsize(file) / (1024 * 1024)  # Size in MB
        print(f"File: {os.path.basename(file)}, Size: {size:.2f} MB")

    for library in libraries:
        start_time = time.time()

        if library == 'pandas':
            dataframes = [pd.read_csv(file, dtype=dtypes) for file in csv_files]
            df = pd.concat(dataframes)
        elif library == 'dask':
            from distributed import Client
            client = Client()
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


def load_one(filename):
    """Load a CSV file and measure the time and memory usage."""
    start_time = time.time()
    dtypes = {'Date_time': 'object', 'Date_time_nr': 'int64', 'Wind_turbine_name': 'object'}

    df = pd.read_csv(filename,
                     dtype=dtypes,
                     parse_dates=['Date_time'],
                     date_format='%Y-%m-%d %H:%M:%S%z'
                     )
    end_time = time.time()
    df['Date_time'] = pd.to_datetime(df['Date_time'], utc=True)

    memory_usage = df.memory_usage(deep=True).sum() / (1024 ** 2)  # Convert bytes to MB
    print(f"Loading time: {end_time - start_time:.2f} seconds.")
    print(f"Memory usage: {memory_usage:.2f} MB.")
    print(f"{len(df)} Lines found.")

    return df


def load_all(folder_path):
    """Load all CSV files and measure the time and memory usage."""
    start_time = time.time()
    dtypes = {'Date_time': 'object', 'Date_time_nr': 'int64', 'Wind_turbine_name': 'object'}

    # Gather CSV files
    csv_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if
                 file.endswith('.csv') and file.startswith('R')]

    dataframes = [pd.read_csv(file, dtype=dtypes, parse_dates=['Date_time'], date_format='%Y-%m-%d %H:%M:%S%z') for file
                  in csv_files]
    df = pd.concat(dataframes)
    end_time = time.time()
    df['Date_time'] = pd.to_datetime(df['Date_time'], utc=True)

    memory_usage = df.memory_usage(deep=True).sum() / (1024 ** 2)  # Convert bytes to MB
    print(f"Loading time: {end_time - start_time:.2f} seconds.")
    print(f"Memory usage: {memory_usage:.2f} MB.")
    print(f"{len(df)} Lines found.")

    return df
