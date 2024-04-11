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
