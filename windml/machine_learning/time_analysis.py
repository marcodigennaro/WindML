import pandas as pd
from statsmodels.tsa.ar_model import AutoReg


def ar_forecast(df, value_col, lags=5, train_size=0.8):
    """
    Performs AR forecasting on time series data with Year and Month columns.
    Trains on a specified percentage of the data and predicts for the test set dates.

    Parameters:
    - df: DataFrame containing the time series data.
    - value_col: String, the name of the column in df that contains the time series values.
    - lags: The number of lagged observations to include in the model.
    - train_size: The proportion of the dataset to include in the train split.

    Returns:
    - forecast_df: A DataFrame containing the forecasted values for the test dates.
    """

    # Create a temporary date column for sorting and forecasting
    df['date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str))

    # Ensure df is sorted by the new date column
    df = df.sort_values('date').reset_index(drop=True)

    # Determine the split point for training and testing
    split_point = int(len(df) * train_size)

    # Split data into training and testing sets
    train = df.iloc[:split_point]
    test = df.iloc[split_point:]

    # Fit an AutoReg model on the training data
    model = AutoReg(train[value_col], lags=lags)
    model_fitted = model.fit()

    # Make forecast for the test period
    start = len(train)
    end = len(df) - 1  # Predict up to the last row in the original DataFrame
    forecast_values = model_fitted.predict(start=start, end=end, dynamic=False)

    # Prepare the forecast DataFrame using test dates
    forecast_df = test[['Year', 'Month']].copy()
    forecast_df[f'pred_{value_col}'] = forecast_values.values  # Ensure forecast values align with test dates

    return forecast_df
