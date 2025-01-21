import matplotlib.pyplot as plt
import matplotlib.dates as mdates  # Ensure dates module is imported
import pandas as pd

def plot_time_series(data, title):
    # Generate visualizations for temperature and precipitation
    plt.figure(figsize=(15, 5))

    # Plot temperature
    plt.subplot(1, 2, 1)
    plt.plot(data['last_updated'], data['temperature_fahrenheit'], color='blue')
    plt.title('Temperature Over Time')
    plt.xlabel('Date (MM-YYYY)')
    plt.ylabel('Temperature')
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%Y'))

    # Plot precipitation
    plt.subplot(1, 2, 2)
    plt.plot(data['last_updated'], data['precip_in'], color='green')
    plt.title('Precipitation Over Time')
    plt.xlabel('Date (MM-YYYY)')
    plt.ylabel('Precipitation (Inches)')
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%Y'))

    # Set the main title
    plt.suptitle(title)

    plt.tight_layout()
    plt.show()

def filter_data_country_location(data, country,location_name):
    data = data[(data['country'] == country) & (data['location_name'] == location_name)]
    return data


# Define the function
def plot_time_series_single(data, country, location_name, columns=['temperature_fahrenheit']):

    # Filter data for the specific country and location
    data = data[(data['country'] == country) & (data['location_name'] == location_name)]

    # Convert 'last_updated' to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(data['last_updated']):
        data['last_updated'] = pd.to_datetime(data['last_updated'])

    # Generate visualizations for multiple columns
    plt.figure(figsize=(15, 5))

    # Iterate through each column and plot
    for column in columns:
        plt.plot(data['last_updated'], data[column], label=column)

    # Formatting the plot
    plt.title(f"Time Series for {country} - {location_name}")
    plt.xlabel('Date (MM-YYYY)')
    plt.ylabel('Values')
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')  # Add x-axis at y=0
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.legend()

    # Set y-axis limits to include all values
    min_value = data[columns].min().min()
    max_value = data[columns].max().max()
    plt.ylim(min(min_value, 0), max_value)  # Ensure y-axis includes negative values

    # Automatically adjust layout
    plt.tight_layout()

    # Display the plot
    plt.show()



