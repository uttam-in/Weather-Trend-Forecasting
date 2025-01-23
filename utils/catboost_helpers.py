import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Function to prepare lagged features for time series forecasting
def create_lagged_features(data, field, lags):
    df = data[[field]].copy()
    for lag in range(1, lags + 1):
        df[f'{field}_lag_{lag}'] = df[field].shift(lag)
    df = df.dropna()
    return df

# Function to forecast using CatBoost within the test range
def forecast_with_catboost(data, field, lags):
    # Prepare lagged features
    df = create_lagged_features(data, field, lags)
    X = df.drop(columns=[field])
    y = df[field]

    # Split into training and test sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Train the CatBoost model
    model = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.1, silent=True)
    model.fit(X_train, y_train)

    # Predict for the test range
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    return y_train, y_test, predictions, mse

# Function to update the plot with CatBoost forecast
def update_plot_with_forecast_catboost(data,output, country, location_name, field, lags):
    data_filtered = data[(data['country'] == country) & (data['location_name'] == location_name)].copy()
    data_filtered = data_filtered.set_index('last_updated').sort_index()

    try:
        train_data, test_data, predictions, mse = forecast_with_catboost(
            data_filtered, field, lags
        )
    except Exception as e:
        with output:
            output.clear_output()
            print(f"Error: {e}")
        return

    with output:
        output.clear_output(wait=True)
        plt.figure(figsize=(10, 6))
        plt.plot(train_data.index, train_data, label='Training Data', color='blue')
        plt.plot(test_data.index, test_data, label='Actual Test Data', color='green')
        plt.plot(test_data.index, predictions, label='Predicted Test Data', color='orange')
        plt.title(f'{field.replace("_", " ").title()} Forecast (CatBoost) - {country}, {location_name}')
        plt.xlabel('Date')
        plt.ylabel(field.replace("_", " ").title())
        plt.legend()
        plt.grid(True)
        plt.show()

        print(f"Mean Squared Error: {mse:.4f}")


def catboost_forecast(data, field, lags, steps):
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[field].dropna().values.reshape(-1, 1))

    # Prepare lagged features
    df = create_lagged_features(pd.DataFrame(scaled_data, columns=[field]), field, lags)
    X = df.drop(columns=[field])
    y = df[field]

    # Train-Test Split
    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]

    # Train CatBoost
    model = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.1, silent=True)
    model.fit(X_train, y_train)

    # Forecast
    forecast_values = list(y_train[-lags:])
    forecast_results = []
    for _ in range(steps):
        input_features = forecast_values[-lags:]
        prediction = model.predict([input_features])[0]
        forecast_results.append(prediction)
        forecast_values.append(prediction)

    # Inverse transform to original scale
    return scaler.inverse_transform(np.array(forecast_results).reshape(-1, 1)).flatten()