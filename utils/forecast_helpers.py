import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

from utils.plot_time_series import filter_data_country_location


# Function to fit ARIMA and forecast
def forecast_with_arima(data, field, p, d, q, steps):
    values = data[field].dropna()
    train_size = int(len(values) * 0.8)
    train_data = values[:train_size]
    test_data = values[train_size:]

    # Fit ARIMA model
    model = ARIMA(train_data, order=(p, d, q))
    model_fitted = model.fit()

    # Forecast
    forecast = model_fitted.forecast(steps=steps)
    forecast_index = test_data.index[:steps] if len(test_data) >= steps else test_data.index

    mse = mean_squared_error(test_data[:steps], forecast) if len(test_data) >= steps else None
    return train_data, test_data, forecast, forecast_index, mse


# Function to update the plot with ARIMA forecast
def update_plot_with_forecast_arima(data,output, country, location_name, field, p, d, q, steps):
    data_filtered = filter_data_country_location(data, country, location_name)
    y = data_filtered[field]
    x = data_filtered.index  # Use the index for plotting

    try:
        train_data, test_data, forecast, forecast_index, mse = forecast_with_arima(data_filtered, field, p, d, q, steps)
    except Exception as e:
        with output:
            output.clear_output()
            print(f"Error: {e}")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(train_data.index, train_data, label='Training Data', color='blue')
    if len(test_data) > 0:
        plt.plot(test_data.index, test_data, label='Actual Test Data', color='green')
    plt.plot(forecast_index, forecast, label='Forecast', color='red', linestyle='dashed')
    plt.title(f'{field.replace("_", " ").title()} Forecast (ARIMA) - {country}, {location_name}')
    plt.xlabel('Date')
    plt.ylabel(field.replace("_", " ").title())
    plt.legend()
    plt.show()

    if mse is not None:
        print(f"Mean Squared Error: {mse:.4f}")
    else:
        print("Not enough test data to compute MSE.")

# ARIMA model
def arima_forecast(train_data, steps, p, d, q):
    model = ARIMA(train_data, order=(p, d, q))
    model_fitted = model.fit()
    return model_fitted.forecast(steps=steps)
