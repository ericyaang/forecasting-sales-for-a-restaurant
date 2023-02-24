
from xgboost import plot_importance
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import numpy as np

def setup_plot_style():
    """
    Sets up the plotting style and parameters using matplotlib.
    """
    plt.style.use('seaborn-v0_8')
    
    params = {
        'font.family': 'serif',
        'font.serif':'STIXGeneral',  
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,  
        'axes.labelsize':16,
        'axes.titlesize':24,
        'font.size':20,
        'figure.figsize': [11,11],
    }
    plt.rcParams.update(params)


def plot_predictions(results, y_true, y_pred, time_series_name='xgboost', value_name='log(sales)',
                     plot_size=(16, 12), big_font=22, med_font=16, small_font=14):
    text_str = '\n'.join((f'{k} = {v:.3f}' for k, v in results.items()))
    fig, axes = plt.subplots(1, 1, figsize=plot_size)
    axes.plot(y_true, 'b-o', label=f'Test data for {time_series_name}')
    axes.plot(y_pred, 'r-o', label=f'Forecast data for {time_series_name}')
    axes.legend(loc='upper left', fontsize=med_font)
    axes.set_title(f'Raw and Predicted data trend for {time_series_name}')
    axes.set_ylabel(value_name)
    axes.set_xlabel(y_true.index.name)
    for tick in axes.get_xticklabels() + axes.get_yticklabels():
        tick.set_fontsize(small_font)  
    for label in [axes.title, axes.xaxis.label, axes.yaxis.label]:
        label.set_fontsize(big_font)
    props = dict(boxstyle='round', facecolor='oldlace', alpha=0.5)
    axes.text(0.05, 0.9, text_str, transform=axes.transAxes, fontsize=med_font, 
                verticalalignment='top', bbox=props)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model):
    """
    Plot the feature importance of a trained XGBoost model.
    """

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.tick_params(axis='both')

    plot_importance(model, ax=ax, height=0.8)
    ax.set_title('Feature Importance')
    ax.set_xlabel('Importance Score')
    ax.set_ylabel('Features')
        
    plt.box(False)
    plt.grid(False)
    plt.tight_layout()
    plt.show()

def plot_errors_and_standardized_errors(errors, y_true):
    """
    Plot the residuals and standardized errors with outlier detection.
    """
    # Calculate the z-scores
    z_scores = (errors - errors.mean()) / errors.std()

    # Identify the outliers
    outliers = y_true[z_scores.abs() > 3]
    print(f'Number of outliers: {len(outliers)}')
    for i in range(len(outliers)):
        date = outliers.index.date[i]
        value = outliers.values[i]
        print(f'\nOutliers:')
        print(f">> {date}: {value.round(3)}")

    # Plot the residuals and standardized errors
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot the residuals
    ax1.plot(errors)
    ax1.axhline(y=0, linestyle='--', color='grey')
    ax1.set(title='Residuals', ylabel='Residuals')

    # Plot the standardized errors
    ax2.plot(z_scores)
    ax2.axhline(y=3, color='r', linestyle='--')
    ax2.axhline(y=-3, color='r', linestyle='--')
    ax2.axhline(y=0, linestyle='--', color='grey')
    ax2.set(title='Standardized Residuals', xlabel='Time', ylabel='Z-score')

    plt.tight_layout()
    plt.show()


def plot_qqplot(errors):
    """
    Plot QQ plot to check if the errors are normally distributed.
    """
    from statsmodels.graphics.gofplots import qqplot
    fig, ax = plt.subplots(figsize=(8, 4))
    qqplot(errors, line='s', ax=ax)
    plt.title('QQ Plot')
    plt.show()

def print_top_3_abs_errors(errors):
    """
    Print the top 3 highest absolute error values and their dates.
    """
    abs_errors = np.abs(errors)
    abs_errors_top_3 = np.flip(np.argsort(abs_errors), 0)
    abs_errors_top_3 = abs_errors_top_3[:3]
    abs_errors_values = abs_errors[abs_errors_top_3]

    print("Top 3 highest absolute error values:")
    for i in range(len(abs_errors_values)):
        date = abs_errors_values.index.date[i]
        value = abs_errors_values.values[i]
        print(f"{date}: {value.round(3)}")    


def show_summary(errors):
    n_errors = len(errors)
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    q25_error, q50_error, q75_error = np.quantile(errors, [0.25, 0.5, 0.75])
    min_error, max_error = np.min(errors), np.max(errors)

    print(f"Number of errors: {n_errors}")
    print(f"Mean of errors: {mean_error:.4f}")
    print(f"Standard deviation of errors: {std_error:.4f}")
    print(f"Quantiles of errors: 25%={q25_error:.4f}, 50%={q50_error:.4f}, 75%={q75_error:.4f}")
    print(f"Min/Max errors: {min_error:.4f}/{max_error:.4f}")

def plot_acf_pacf(errors, num_lags=20, figsize=(10, 8)):
    """
    Plots ACF and PACF plots for residual analysis of a time series model.

    Parameters:
    -----------
    errors : array-like
        Array of errors or residuals from the time series model.
    num_lags : int, optional (default=20)
        The number of lags to include in the ACF and PACF plots.
    figsize : tuple, optional (default=(10, 8))
        The size of the plot figure.

    Returns:
    --------
    None
    """
    fig, ax = plt.subplots(nrows=2, figsize=figsize)

    plot_acf(errors, ax=ax[0], lags=num_lags)
    ax[0].set_title('Autocorrelation Function (ACF) from residuals')
    ax[0].set_xlabel('Lag')
    ax[0].set_ylabel('Correlation')

    plot_pacf(errors, ax=ax[1], lags=num_lags, method='ywm')
    ax[1].set_title('Partial Autocorrelation Function (PACF) from residuals')
    ax[1].set_xlabel('Lag')
    ax[1].set_ylabel('Correlation')

    plt.tight_layout()
    plt.show()        

def plot_errors_by_dayofweek(y_true, y_pred):
    """
    Plot the median absolute error grouped by day of the week.
    """
    # Calculate errors by day of week
    errors_by_day = (y_true - y_pred).abs().groupby(y_true.index.dayofweek).median()

    # Plot errors
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(errors_by_day)
    ax.set_xticks(range(7))
    ax.set_xticklabels(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    ax.set(title='Median Absolute Error by Day of Week', xlabel='Day of Week', ylabel='Absolute Error')
    plt.tight_layout()
    plt.show()


def error_analysis(prediction, data, results, model):
    """
    Perform advanced error analysis on the predictions made by the model.

    Args:
    - prediction: the predicted values made by the model
    - data: a dictionary containing the training and testing datasets
    - results: a dictionary containing the results of the model training
    - model: the trained XGBoost model

    Returns:
    - None
    """

    # Set matplotlib config
    setup_plot_style()

    # Create a DataFrame of predictions and true values
    df_preds = data['y_test'].assign(y_pred=prediction)

    # Extract the true values and predicted values as separate variables
    y_true = df_preds['net_sales']
    y_pred = df_preds['y_pred']

    # Calculate the errors (residuals) between the true and predicted values
    errors = y_true - y_pred

    # Plotting the actual versus predicted values
    plot_predictions(results, y_true, y_pred, time_series_name='xgboost', value_name='log(sales)',
                    plot_size=(16, 12), big_font=22, med_font=16, small_font=14)
    
    # Plot feature importance
    plot_feature_importance(model)

    # Show summary statistics of errors
    show_summary(errors)

    # Plot errors and standardized errors
    plot_errors_and_standardized_errors(errors, y_true)

    # Plot autocorrelation and partial autocorrelation plots
    plot_acf_pacf(errors)

    # Plot quantile-quantile plot
    plot_qqplot(errors)

    # Print top 3 absolute errors and plot errors by day of week
    print_top_3_abs_errors(errors)
    plot_errors_by_dayofweek(y_true, y_pred)
