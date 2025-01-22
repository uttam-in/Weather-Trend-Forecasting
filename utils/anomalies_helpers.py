import matplotlib.pyplot as plt
from utils.plot_time_series import filter_data_country_location
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
import numpy as np

# Function to detect anomalies based on the selected model
def detect_anomalies(data, field, model_name):
    values = data[field].dropna().values.reshape(-1, 1)

    if model_name == "IsolationForest":
        model = IsolationForest(contamination=0.05, random_state=42)
        anomalies = model.fit_predict(values)

    elif model_name == "K-Means":
        model = KMeans(n_clusters=2, random_state=42)
        clusters = model.fit_predict(values)
        # Assume the smaller cluster corresponds to anomalies
        anomalies = np.where(clusters == np.argmin(np.bincount(clusters)), -1, 1)

    else:
        raise ValueError("Unsupported model selected.")

    return anomalies

# Function to update the plot with anomalies
def update_plot_with_anomalies(data, output, country, location_name, field, model_name):
    data_filtered = filter_data_country_location(data, country, location_name)
    y = data_filtered[field]
    x = data_filtered['last_updated']

    # Detect anomalies
    try:
        anomalies = detect_anomalies(data_filtered, field, model_name)
    except NotImplementedError as e:
        with output:
            output.clear_output()
            print(e)
        return

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='Data', color='blue')
    plt.scatter(x[anomalies == -1], y[anomalies == -1], color='red', label='Anomalies')
    plt.title(f'{field.replace("_", " ").title()} Time Series with Anomalies - {country}, {location_name}')
    plt.xlabel('Date')
    plt.ylabel(field.replace("_", " ").title())
    plt.legend()
    plt.show()