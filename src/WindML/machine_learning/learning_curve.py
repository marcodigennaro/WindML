from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import laplacian_kernel

from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression


def create_learning_curve_with_grid_search(X, y, model, param_grid, num_points=5):
    """
    Generate learning curve data by training the model on subsets of the data
    and using GridSearchCV to find the best parameters for each subset.

    Parameters:
    - X: Feature matrix.
    - y: Target vector.
    - model: Machine learning model (not yet wrapped in GridSearchCV).
    - param_grid: Dictionary of parameters to search over for GridSearchCV.
    - num_points: Number of points for the learning curve.

    Returns:
    - List of tuples (subset size, MAE, best parameters).
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    max_size = len(X_train)
    print(max_size)
    subset_sizes = np.logspace(np.log10(10), np.log10(max_size), num=num_points, dtype=int)
    print(subset_sizes)

    learning_curve_data = []

    for size in np.unique(subset_sizes):  # Ensure unique sizes due to rounding
        print('size = ', size)

        X_train_subset = X_train.sample(size)
        y_train_subset = y_train.sample(size)

        # Use GridSearchCV to find the best model parameters for this subset
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_absolute_error')
        grid_search.fit(X_train_subset, y_train_subset)

        y_pred = grid_search.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)

        learning_curve_data.append((size, mae, grid_search.best_params_))

        print(f"Subset size: {size}, Best parameters: {grid_search.best_params_}")

    return learning_curve_data


def plot_learning_curve(learning_curve_data):
    subset_sizes, maes, _ = zip(*learning_curve_data)
    plt.figure()
    plt.scatter(subset_sizes, maes, color='blue')
    plt.loglog(subset_sizes, maes, label='MAE with Best Parameters', linestyle='--')
    plt.xlabel('Subset Size')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('Learning Curve with GridSearchCV')
    plt.xscale('log')
    plt.grid(True)
    plt.legend()
    plt.show()
