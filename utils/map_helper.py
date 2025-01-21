import folium
from folium.plugins import TimestampedGeoJson
import pandas as pd
import numpy as np

def create_temperature_map(data):
    # Convert 'last_updated' to datetime format
    data['last_updated'] = pd.to_datetime(data['last_updated'])

    # Generate a complete timeline (month and year) covering the data range
    min_date = data['last_updated'].min()
    max_date = data['last_updated'].max()
    timeline = pd.date_range(start=min_date, end=max_date, freq='MS')

    # Create a base map
    m = folium.Map(location=[0, 0], zoom_start=2)  # Adjusted to show the world map

    # Define a function to determine the color based on temperature
    def get_color(temperature):
        if temperature < 50:
            return 'blue'
        elif 50 <= temperature < 70:
            return 'green'
        elif 70 <= temperature < 90:
            return 'orange'
        else:
            return 'red'

    # Add a legend to the map
    def add_legend(map_object):
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: 120px; 
                    background-color: white; z-index:9999; font-size:14px; 
                    border:2px solid grey; border-radius:5px; padding: 10px;">
            <b>Temperature Legend</b><br>
            <i style="background: blue; width: 10px; height: 10px; display: inline-block;"></i> Below 50°F<br>
            <i style="background: green; width: 10px; height: 10px; display: inline-block;"></i> 50°F - 70°F<br>
            <i style="background: orange; width: 10px; height: 10px; display: inline-block;"></i> 70°F - 90°F<br>
            <i style="background: red; width: 10px; height: 10px; display: inline-block;"></i> Above 90°F
        </div>
        '''
        map_object.get_root().html.add_child(folium.Element(legend_html))

    # Group by regions (assume 'region' column identifies the region)
    region_groups = data.groupby('location_name')

    # Create GeoJSON features for each region and fill missing months
    features = []
    for region, group in region_groups:
        # Select only numeric columns before resampling
        numeric_columns = group.select_dtypes(include=[np.number])
        group = group.set_index('last_updated')
        resampled_group = group[numeric_columns.columns].resample('MS').mean().reset_index()
        resampled_group['location_name'] = region  # Add region name back

        for _, row in resampled_group.iterrows():
            avg_lat = row['latitude'] if not np.isnan(row['latitude']) else 0
            avg_lon = row['longitude'] if not np.isnan(row['longitude']) else 0
            avg_temperature = row['temperature_fahrenheit'] if not np.isnan(row['temperature_fahrenheit']) else 0
            color = get_color(avg_temperature)

            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [[
                        [avg_lon - 0.5, avg_lat - 0.5],
                        [avg_lon - 0.5, avg_lat + 0.5],
                        [avg_lon + 0.5, avg_lat + 0.5],
                        [avg_lon + 0.5, avg_lat - 0.5],
                        [avg_lon - 0.5, avg_lat - 0.5]
                    ]]  # Approximated as a square region for simplicity
                },
                'properties': {
                    'time': row['last_updated'].strftime('%Y-%m'),  # Format to show month and year
                    'style': {
                        'fillColor': color,
                        'fillOpacity': 0.6,
                        'stroke': 'true',
                        'color': color
                    },
                    'popup': f"Average Temperature: {avg_temperature:.2f}°F\nLocation: {region}"
                }
            }
            features.append(feature)

    # Add the TimestampedGeoJson layer
    TimestampedGeoJson({
        'type': 'FeatureCollection',
        'features': features
    }, period='P1M', add_last_point=True).add_to(m)  # Adjusted to show month and year as the time slider

    # Add the legend to the map
    add_legend(m)

    # Display the map in the Jupyter Notebook
    return m