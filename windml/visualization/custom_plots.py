import numpy as np
from windrose import WindroseAxes, plot_windrose
from matplotlib import cm

import matplotlib.pyplot as plt
import seaborn as sns


def plot_monthly_evolution(df_energy, value_col):
    """
    Plots the time evolution of a specified value over months and the distribution per month,
    using colormaps for aesthetic adjustments. 'Month' and 'Year' are used for grouping.

    Parameters:
    - df_energy: DataFrame containing the data to plot.
    - value_col: The name of the column in df that represents the value to plot (e.g., energy production).
    """
    # Set up the figure with two subplots
    f, ax = plt.subplots(2, 1, figsize=(8, 8))

    # Use a colormap for the monthly energy production scatter plot
    # Creating a normalized month value for colormap
    norm = plt.Normalize(df_energy['Month'].min(), df_energy['Month'].max())
    smap = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
    df_energy['color'] = df_energy['Month'].apply(lambda x: smap.to_rgba(x))

    # Scatter plot for annual comparison with connected dots
    for year in df_energy['Year'].unique():
        year_data = df_energy[df_energy['Year'] == year]
        ax[0].scatter(year_data['Month'], year_data[value_col],
                      label=year, edgecolor='black')
        ax[0].plot(year_data['Month'], year_data[value_col],
                   linestyle='--', alpha=0.5)

    ax[0].set_title(f'Monthly {value_col} - Annual Comparison')
    ax[0].set_xlabel('Month')
    ax[0].set_ylabel(f'{value_col}')
    ax[0].legend(title='Year')

    # Box plot for monthly distribution using the same colormap concept
    # Since boxplot won't directly use the color mapping, manually assign colors
    month_colors = df_energy.groupby('Month')['color'].first().values
    sns.boxplot(x="Month", y=value_col, data=df_energy,
                ax=ax[1], palette=month_colors)

    ax[1].set_title(f'Monthly {value_col} - Boxplot')
    ax[1].set_xlabel('Month')
    ax[1].set_ylabel(f'{value_col}')

    plt.tight_layout()
    plt.show()
    df_energy.drop(columns='color', inplace=True)


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
    ax.set_rticks([700, 1400, 2100, 2800, 3500], labels=[
                  '700', '1400', '2100', '2800', '3500'])  # Set radial ticks
    # Set the direction of the angular axis (counterclockwise)
    ax.set_theta_direction(-1)
    # Set the zero location of the angular axis (north)
    ax.set_theta_zero_location('N')


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
