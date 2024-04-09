import numpy as np
from matplotlib import pyplot as plt
from windrose import WindroseAxes, plot_windrose
from matplotlib import cm


def plot_windrose_subplots(data, *, direction, var, **kwargs):
    """Wrapper function to create subplots per axis"""
    ax = plt.gca()
    ax = WindroseAxes.from_ax(ax=ax)

    wd = data[direction]
    ws = data[var]

    ax.contourf(wd, ws, bins=np.arange(0, 8, 1), cmap=cm.hot)
    ax.contour(wd, ws, bins=np.arange(0, 8, 1), colors="black")
    ax.set_legend()

    # Set the same scale for radial and angular axes
    ax.set_rmax(3500)  # Set the maximum radial value
    ax.set_rticks([700, 1400, 2100, 2800, 3500], labels = ['700', '1400', '2100', '2800', '3500'])  # Set radial ticks
    ax.set_theta_direction(-1)  # Set the direction of the angular axis (counterclockwise)
    ax.set_theta_zero_location('N')  # Set the zero location of the angular axis (north)


def plot_learning_curve(learning_curve_data, error_metric='mae'):
    """
    Plots the learning curve based on the provided learning curve data.

    Parameters:
    - learning_curve_data: Dictionary containing learning curve data.
    - error_metric: Error metric to be plotted. Options: 'mae', 'mse', 'r2'.
    """
    subset_sizes = list(learning_curve_data.keys())
    errors = [learning_curve_data[size][error_metric] for size in subset_sizes]

    plt.figure()
    plt.scatter(subset_sizes, errors, color='blue')
    plt.loglog(subset_sizes, errors, label=f'{error_metric.upper()} with Best Parameters', linestyle='--')
    plt.xlabel('Subset Size')
    plt.ylabel(f'{error_metric.upper()}')
    plt.title('Learning Curve with GridSearchCV')
    plt.xscale('log')
    plt.grid(True)
    plt.legend()
    plt.show()
