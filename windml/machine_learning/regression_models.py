import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def learning_curve_with_CV(df, x_list, y_variable, model, param_grid, lc_npoints=6, cv_npoints=5,
                   cv_scoring='neg_mean_absolute_error'):
    """
    Generates a learning curve by training the model on subsets of the data
    and using GridSearchCV to find the best parameters for each subset.

    Parameters:
    - model: Unfitted machine learning model (to be wrapped in GridSearchCV).
    - param_grid: Dictionary of parameters to search over for GridSearchCV.
    - lc_npoints: Number of points for the learning curve.
    - cv_npoints: Number of folds for cross-validation in GridSearchCV.
    - cv_scoring: Scoring metric for cross-validation in GridSearchCV.

    Returns:
    - Dictionary: Subset size mapped to dictionary of metrics and best parameters.
    """

    max_size = len(df)
    subset_sizes = np.logspace(np.log10(10), np.log10(
        max_size), num=lc_npoints + 1, dtype=int)

    learning_curve_data = {}

    for size in np.unique(subset_sizes):  # Ensure unique sizes due to rounding
        subset_df = df.sample(size)

        X = subset_df[x_list]
        y = subset_df[y_variable].values.reshape(-1, 1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Use GridSearchCV to find the best model parameters for this subset
        grid_search = GridSearchCV(
            model, param_grid, cv=cv_npoints, scoring=cv_scoring)
        grid_search.fit(X_train, y_train)

        y_pred_train = grid_search.predict(X_train)
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

        print(f"Subset size: {size}, MAE: {learning_curve_data[size]['mae']}")

    return learning_curve_data

def learning_curve(df, x_list, y_variable, grid_results, lc_npoints=6):
    """
    Generates a learning curve by training the model on subsets of the data
    and using GridSearchCV to find the best parameters for each subset.

    Parameters:
    - model: Unfitted machine learning model (to be wrapped in GridSearchCV).
    - param_grid: Dictionary of parameters to search over for GridSearchCV.
    - lc_npoints: Number of points for the learning curve.
    - cv_npoints: Number of folds for cross-validation in GridSearchCV.
    - cv_scoring: Scoring metric for cross-validation in GridSearchCV.

    Returns:
    - Dictionary: Subset size mapped to dictionary of metrics and best parameters.
    """

    max_size = len(df)
    subset_sizes = np.logspace(np.log10(10), np.log10(
        max_size), num=lc_npoints + 1, dtype=int)

    learning_curve_data = {}

    for size in np.unique(subset_sizes):  # Ensure unique sizes due to rounding
        subset_df = df.sample(size)

        X = subset_df[x_list]
        y = subset_df[y_variable].values.reshape(-1, 1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        grid_results.fit(X_train, y_train)

        y_pred_train = grid_results.predict(X_train)
        y_pred_test = grid_results.predict(X_test)

        learning_curve_data[size] = {
            'mae': mean_absolute_error(y_test, y_pred_test),
            'mse': mean_squared_error(y_test, y_pred_test),
            'R2': r2_score(y_test, y_pred_test),
            'parameters': grid_results.best_params_,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test
        }

        print(f"Subset size: {size}, MAE: {learning_curve_data[size]['mae']}")

    return learning_curve_data

