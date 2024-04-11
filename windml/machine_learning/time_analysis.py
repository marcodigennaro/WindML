import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA


def learning_curve_with_grid_search(X, y, model, param_grid, lc_npoints=5, cv_npoints=5,
                                    cv_scoring='neg_mean_absolute_error'):
    """
    Generates a learning curve by training the model on subsets of the data
    and using GridSearchCV to find the best parameters for each subset.

    Parameters:
    - X: Feature matrix.
    - y: Target vector.
    - model: Unfitted machine learning model (to be wrapped in GridSearchCV).
    - param_grid: Dictionary of parameters to search over for GridSearchCV.
    - lc_npoints: Number of points for the learning curve.
    - cv_npoints: Number of folds for cross-validation in GridSearchCV.
    - cv_scoring: Scoring metric for cross-validation in GridSearchCV.

    Returns:
    - Dictionary: Subset size mapped to dictionary of metrics and best parameters.
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    max_size = len(X_train)
    subset_sizes = np.logspace(np.log10(10), np.log10(
        max_size), num=lc_npoints, dtype=int)

    learning_curve_data = {}

    for size in np.unique(subset_sizes):  # Ensure unique sizes due to rounding

        X_train_subset = X_train.sample(size)
        y_train_subset = y_train.sample(size)

        # Use GridSearchCV to find the best model parameters for this subset
        grid_search = GridSearchCV(
            model, param_grid, cv=cv_npoints, scoring=cv_scoring)
        grid_search.fit(X_train_subset, y_train_subset)

        y_pred_train = grid_search.predict(X_train_subset)
        y_pred_test = grid_search.predict(X_test)

        learning_curve_data[size] = {
            'mae': mean_absolute_error(y_test, y_pred_test),
            'mse': mean_squared_error(y_test, y_pred_test),
            'R2': r2_score(y_test, y_pred_test),
            'parameters': grid_search.best_params_,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test
        }

        print(f"Subset size: {size}, Best parameters: {grid_search.best_params_}")

    return learning_curve_data


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
    df['date'] = pd.to_datetime(df['Year'].astype(
        str) + '-' + df['Month'].astype(str))

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
    # Ensure forecast values align with test dates
    forecast_df[f'pred_{value_col}'] = forecast_values.values

    return forecast_df


def arima_forecast(df, value_col, order=(1, 1, 1), train_size=0.8):
    """
    Performs ARIMA forecasting on time series data with Year and Month columns.
    Trains on a specified percentage of the data and predicts for the test set dates.

    Parameters:
    - df: DataFrame containing the time series data.
    - value_col: String, the name of the column in df that contains the time series values.
    - order: Tuple of the form (p,d,q) where p is the order of the AR term,
      d is the degree of differencing, and q is the order of the MA term.
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

    # Fit an ARIMA model on the training data
    model = ARIMA(train[value_col], order=order)
    model_fitted = model.fit()

    # Make forecast for the test period
    forecast_values = model_fitted.forecast(steps=len(test))

    # Prepare the forecast DataFrame using test dates
    forecast_df = test[['Year', 'Month']].copy()
    forecast_df[f'pred_{value_col}'] = forecast_values.values

    return forecast_df
