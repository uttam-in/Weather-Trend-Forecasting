from statsmodels.tools.sm_exceptions import ValueWarning
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing_extensions import dataclass_transform
from utils.plot_time_series import filter_data_country_location
import ipywidgets as widgets
from IPython.display import display
import warnings
from utils.plot_time_series import filter_data_country_location
from utils.anomalies_helpers import update_plot_with_anomalies

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.cluster._kmeans")
warnings.filterwarnings("ignore", category=ValueWarning, module="statsmodels.tsa.base.tsa_model")
warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels.tsa.base.tsa_model")

def get_country_dropdown(data):
    # Create a dropdown widget for selecting the country
    country_dropdown = widgets.Dropdown(
        options=data['country'].unique(),
        description='Country:',
        value='United States of America',
        layout=widgets.Layout(width='200px')  # Set fixed width
    )
    return country_dropdown

def get_location_dropdown(data):
    # Create a dropdown widget for selecting the location
    location_dropdown = widgets.Dropdown(
        options=data[data['country'] == 'United States of America']['location_name'].unique(),
        description='Location:',
        value='Washington Harbor',
        layout=widgets.Layout(width='200px')  # Set fixed width
    )
    return location_dropdown

def get_field_dropdown(data):
    # Create a dropdown widget for selecting the field
    field_dropdown = widgets.Dropdown(
        options=[
            'temperature_celsius', 'temperature_fahrenheit',
            'wind_mph', 'wind_kph', 'wind_degree', 'pressure_mb', 'pressure_in', 'precip_mm', 'precip_in',
            'humidity', 'cloud', 'feels_like_celsius', 'feels_like_fahrenheit', 'visibility_km', 'visibility_miles',
            'uv_index', 'gust_mph', 'gust_kph', 'air_quality_Carbon_Monoxide', 'air_quality_Ozone',
            'air_quality_Nitrogen_dioxide', 'air_quality_Sulphur_dioxide', 'air_quality_PM2.5', 'air_quality_PM10',
            'air_quality_us-epa-index', 'air_quality_gb-defra-index', 'moon_illumination'
        ],
        description='Field:',
        value='temperature_fahrenheit',
        layout=widgets.Layout(width='200px')  # Set fixed width
    )
    return field_dropdown

# Function to update the plot based on the selected country, location, and field
def eda_update_plot(data, country, location_name, field):
    data_filtered = filter_data_country_location(data, country, location_name)
    y = data_filtered[field]
    x = data_filtered['last_updated']

    plt.figure(figsize=(10, 6))  # Adjust the plot size
    plt.plot(x, y, color='blue')
    plt.title(f'{field.replace("_", " ").title()} Time Series - {country}, {location_name}')
    plt.xlabel('Date')
    plt.ylabel(field.replace("_", " ").title())
    plt.show()
