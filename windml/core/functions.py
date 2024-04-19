import os
import time
from memory_profiler import memory_usage
import pandas as pd
import dask.dataframe as dd
import vaex
import modin.pandas as mpd

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


def compare_data_libraries(folder_path):
    """
    Compares the performance of different data processing libraries (pandas, dask, vaex, modin)
    by reading, concatenating, and processing CSV files stored in a specified folder.

    The function logs the execution time and memory usage for each library, providing insights into
    their efficiency for handling multiple CSV files. The function is useful for benchmarking
    the libraries on real-world data processing tasks.

    Parameters:
    - folder_path (str): The path to the directory containing the CSV files. The files should
                         start with 'R' and have a '.csv' extension.

    Notes:
    - This function assumes that the CSV files share a consistent structure suitable for the
      specified dtypes.
    - Requires the installation of pandas, dask, vaex, modin, and memory_profiler.

    Outputs:
    - Console output logging the number of files processed, their sizes, the processing time,
      and maximum memory usage for each library.
    """

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


def load_one(filename, subset_size=False):
    """
        Loads a CSV file into a pandas DataFrame, applies data polishing, and optionally samples a subset
        of the data. It also measures and reports the loading time and memory usage.

        This function is useful for benchmarking data loading and processing performance, and for handling
        large datasets by sampling.

        Parameters:
        - filename (str): The path to the CSV file to be loaded.
        - subset_size (int, optional): If specified, the DataFrame will be sampled to this number of rows
                                       to potentially reduce memory usage and processing time.

        Returns:
        - DataFrame: The processed DataFrame.

        Outputs:
        - Prints the loading time in seconds.
        - Prints the memory usage in megabytes.
        - Prints the number of lines in the final DataFrame.

        Example:
        >>> df = load_one("path/to/data.csv", subset_size=1000)
        >>> print(df.head())
    """

    start_time = time.time()
    dtypes = {'Date_time': 'object', 'Date_time_nr': 'int64', 'Wind_turbine_name': 'object'}

    df = pd.read_csv(filename,
                     dtype=dtypes,
                     parse_dates=['Date_time'],
                     date_format='%Y-%m-%d %H:%M:%S%z'
                     )
    end_time = time.time()

    df = polish_data(df)

    if subset_size:
        # Extract a subset of the database to speed up the calculation
        # Adjust to your machine

        df = df.sample(subset_size)

    used_memory = df.memory_usage(deep=True).sum() / (1024 ** 2)  # Convert bytes to MB
    print(f"Loading time: {end_time - start_time:.2f} seconds.")
    print(f"Memory usage: {used_memory:.2f} MB.")
    print(f"{len(df)} Lines found.")

    return df


def load_all(folder_path):
    """Load all CSV files and measure the time and memory usage."""
    start_time = time.time()
    dtypes = {'Date_time': 'object', 'Date_time_nr': 'int64', 'Wind_turbine_name': 'object'}

    # Gather CSV files
    csv_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if
                 file.endswith('.csv') and file.startswith('R')]

    dataframes = [
        pd.read_csv(file,
                    dtype=dtypes,
                    parse_dates=['Date_time'],
                    date_format='%Y-%m-%d %H:%M:%S%z') for file in csv_files
        ]

    df = pd.concat(dataframes)
    end_time = time.time()

    df = polish_data(df)

    memory_usage = df.memory_usage(deep=True).sum() / (1024 ** 2)  # Convert bytes to MB
    print(f"Loading time: {end_time - start_time:.2f} seconds.")
    print(f"Memory usage: {memory_usage:.2f} MB.")
    print(f"{len(df)} Lines found.")

    return df


def polish_data(df):
    """
    Processes a DataFrame to enhance its usability for time series analysis by setting a datetime
    index and extracting relevant datetime components such as year, month, day of the week, and hour of day.

    This function modifies the DataFrame in-place and adds new columns that represent the temporal
    features of the data, which are useful for downstream analysis and modeling tasks.

    Parameters:
    - df (DataFrame): A pandas DataFrame expected to contain a column named 'Date_time' with date-time information.

    Returns:
    - DataFrame: The modified DataFrame with the 'Date_time' column set as a datetime index and
                 additional columns for 'Year', 'Month', 'DayOfWeek', and 'HourOfDay'.

    Examples:
    >>> df = pd.DataFrame({
    ...     'Date_time': ['2021-01-01 00:00:00', '2021-01-01 01:00:00'],
    ...     'Data': [100, 150]
    ... })
    >>> polished_df = polish_data(df)
    >>> print(polished_df.columns)
    Index(['Data', 'Year', 'Month', 'DayOfWeek', 'HourOfDay'], dtype='object')
    """

    # It's good practice to assign the date time to the index
    df = df.set_index('Date_time')

    # Assign datetime format to column 'Date_time'
    df.index = pd.to_datetime(df.index, utc=True)

    # Extract the year
    df['Year'] = df.index.year

    # Extract the month
    df['Month'] = df.index.month

    # Extract day of the week (Monday=0, Sunday=6)
    df['DayOfWeek'] = df.index.dayofweek

    # Extract time of day
    df['HourOfDay'] = df.index.hour

    return df


def find_highly_correlated_variables(correlation_matrix, target_variable='P_avg', correlation_threshold=0.1):
    """
    Identifies and returns a DataFrame with variable names and their correlation values
    that have correlations beyond a specified threshold with the target variable in the given correlation matrix.

    Parameters:
    - correlation_matrix (pd.DataFrame): A pandas DataFrame representing the correlation matrix.
    - target_variable (str): The name of the column in the correlation matrix for which to find correlations.
    - correlation_threshold (float): The minimum absolute correlation value to consider.

    Returns:
    - pd.DataFrame: A DataFrame containing variable names and their correlation values with the target variable,
                    sorted by the absolute value of the correlation.
    """

    # Filter variables based on the correlation threshold
    filtered_correlations = correlation_matrix[target_variable][
        (correlation_matrix[target_variable].abs() > correlation_threshold)]

    # Remove the target variable to exclude self-correlation
    if target_variable in filtered_correlations:
        filtered_correlations = filtered_correlations.drop(index=target_variable)

    # Create a DataFrame from the filtered correlation series
    result_df = filtered_correlations.abs().sort_values(ascending=False).reset_index()
    result_df.columns = ['Variable', 'Correlation']

    return result_df

