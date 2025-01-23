import matplotlib.pyplot as plt
import numpy as np
from sympy.abc import alpha

from utils.lstm_helpers import lstm_forecast
from utils.catboost_helpers import catboost_forecast
from utils.sarima_forecast_helpers import sarima_forecast
from utils.forecast_helpers import arima_forecast
from sklearn.metrics import mean_squared_error

def ensemble_forecast(data, field, p, d, q, P, D, Q, s, lags, steps):
    values = data[field].dropna()
    train_size = int(len(values) * 0.8)
    train_data = values[:train_size]
    test_data = values[train_size:]

    # Generate forecasts from each model
    arima_pred = arima_forecast(train_data, steps, p, d, q)
    sarima_pred = sarima_forecast(train_data, steps, p, d, q, P, D, Q, s)
    catboost_pred = catboost_forecast(data, field, lags, steps)
    lstm_pred = lstm_forecast(data, field, lags, steps)

    # Combine forecasts (weighted average)
    ensemble_pred = (
            0.25 * np.array(arima_pred) +
            0.25 * np.array(sarima_pred) +
            0.25 * np.array(catboost_pred) +
            0.25 * np.array(lstm_pred)
    )

    mse = mean_squared_error(test_data[:steps], ensemble_pred) if len(test_data) >= steps else None
    return train_data, test_data, arima_pred, sarima_pred, catboost_pred, lstm_pred, ensemble_pred, mse

def update_plot_with_ensemble(data, output, country, location_name, field, p, d, q, P, D, Q, s, lags, steps):
    data_filtered = data[(data['country'] == country) & (data['location_name'] == location_name)]

    try:
        # Correctly handle all 8 returned values
        train_data, test_data, arima_pred, sarima_pred, catboost_pred, lstm_pred, ensemble_pred, mse = ensemble_forecast(
            data_filtered, field, p, d, q, P, D, Q, s, lags, steps
        )
    except Exception as e:
        with output:
            output.clear_output()
            print(f"Error: {e}")
        return

    # Plot the results
    plt.figure(figsize=(12, 8))
    plt.plot(data_filtered.index[:len(train_data)], train_data, label='Training Data', color='blue')
    plt.plot(data_filtered.index[len(train_data):len(train_data) + len(test_data)], test_data, label='Actual Test Data', color='green', alpha=0.5)
    plt.plot(data_filtered.index[len(train_data):len(train_data) + len(arima_pred)], arima_pred, label='ARIMA Forecast', color='orange', alpha=0.5)
    plt.plot(data_filtered.index[len(train_data):len(train_data) + len(sarima_pred)], sarima_pred, label='SARIMA Forecast', color='purple', alpha=0.5)
    plt.plot(data_filtered.index[len(train_data):len(train_data) + len(catboost_pred)], catboost_pred, label='CatBoost Forecast', color='brown', alpha=0.5)
    plt.plot(data_filtered.index[len(train_data):len(train_data) + len(lstm_pred)], lstm_pred, label='LSTM Forecast', color='pink', alpha=0.5)
    plt.plot(data_filtered.index[len(train_data):len(train_data) + len(ensemble_pred)], ensemble_pred, label='Ensemble Forecast', color='red', linestyle='dashed')
    plt.title(f'{field.replace("_", " ").title()} Ensemble Forecast - {country}, {location_name}')
    plt.xlabel('Date')
    plt.ylabel(field.replace("_", " ").title())
    plt.legend()
    plt.show()

    if mse is not None:
        print(f"Mean Squared Error: {mse:.4f}")
    else:
        print("Not enough test data to compute MSE.")