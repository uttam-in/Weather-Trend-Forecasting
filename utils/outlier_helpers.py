import pandas as pd
# find the count of outliers in each numerical column below 25th percentile and above 75th percentile

def find_outliers(data):
    results = []

    # Iterate through all numerical columns in the dataset
    for column in data.select_dtypes(include=['number']).columns:
        # Calculate the 25th and 75th percentiles
        q25 = data[column].quantile(0.25)
        q75 = data[column].quantile(0.75)

        # Identify outliers
        outliers_above_75 = (data[column] > q75).sum()
        outliers_below_25 = (data[column] < q25).sum()

        # Append results for the current column
        results.append({
            'column_name': column,
            '# < 25 percentile': outliers_below_25,
            '# > 75 percentile': outliers_above_75
        })

    outliers_df = pd.DataFrame(results)
    return outliers_df