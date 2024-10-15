import numpy as np
import pandas as pd
import time
import math
import os
import json

## random state
rst = 42

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import preprocessing

from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns

import requests
import osmnx as ox
import networkx as nx

import geopandas as gpd
from geopy.distance import great_circle
from geopy.exc import GeocoderTimedOut

import folium
from folium.plugins import HeatMap, MarkerCluster
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.prepared import prep

#########################################################################################################################
#########################################################################################################################

def calculate_distance_gc(row):
    pickup = (row['pickup_latitude'], row['pickup_longitude'])
    dropoff = (row['dropoff_latitude'], row['dropoff_longitude'])

    try:
        distance = great_circle(pickup, dropoff).km  # or .miles for miles
    except GeocoderTimedOut:
        distance = None

    return distance

def calculate_speed(trip_duration, distance):
    # Convert duration to hours (duration is in seconds)
    trip_duration_hours = trip_duration / 3600
    if trip_duration_hours > 0:
        return distance / trip_duration_hours
    else:
        return 0

def cal_road_distance_osmnx(df): 
    G = ox.graph_from_place('New York, United States', network_type='drive')

    def road_distance_osmnx(row):
        pickup_latitude = row['pickup_latitude']
        pickup_longitude = row['pickup_longitude']
        dropoff_latitude = row['dropoff_latitude']
        dropoff_longitude = row['dropoff_longitude']

        pickup_node = ox.distance.nearest_nodes(G, pickup_longitude, pickup_latitude)
        dropoff_node = ox.distance.nearest_nodes(G, dropoff_longitude, dropoff_latitude)

        try:
            distance = nx.shortest_path_length(G, pickup_node, dropoff_node, weight='length') / 1000  # Convert to km
        except nx.NetworkXNoPath:
            distance = None

        return distance

    df['distance_osmnx'] = df.apply(road_distance_osmnx, axis=1)

    return df

def cal_road_distance_osrm(df):
    osrm_url = 'http://router.project-osrm.org/route/v1/driving/{},{};{},{}'
    total_rows = len(df)

    def road_distance_osrm(row):
        pickup_latitude = row['pickup_latitude']
        pickup_longitude = row['pickup_longitude']
        dropoff_latitude = row['dropoff_latitude']
        dropoff_longitude = row['dropoff_longitude']

        if pd.isna(pickup_latitude) or pd.isna(pickup_longitude) or pd.isna(dropoff_latitude) or pd.isna(dropoff_longitude):
            return None  # Return None if any coordinate is missing

        url = osrm_url.format(pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude)

        try:
            response = requests.get(url)
            data = response.json()

            if response.status_code == 200 and 'routes' in data and data['routes']:
                distance = data['routes'][0]['distance'] / 1000  # Convert to km
            else:
                print(f"Error in response: {data}")
                distance = None
        except Exception as e:
            print(f"Error occurred: {e}")
            distance = None

        return distance

    for index, row in df.iterrows():
        df.at[index, 'distance_osrm'] = road_distance_osrm(row)

        if index % 100 == 0:
            print(f"Processing row {index} of {total_rows}...")

    return df

def cal_road_distance_osrm_local(df):
    osrm_url = "http://localhost:5001/route/v1/driving/{},{};{},{}?overview=false"
    total_rows = len(df)

    def road_distance_osrm(row):
        pickup_latitude = row['pickup_latitude']
        pickup_longitude = row['pickup_longitude']
        dropoff_latitude = row['dropoff_latitude']
        dropoff_longitude = row['dropoff_longitude']

        if pd.isna(pickup_latitude) or pd.isna(pickup_longitude) or pd.isna(dropoff_latitude) or pd.isna(dropoff_longitude):
            return None  # Return None if any coordinate is missing

        url = osrm_url.format(pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude)

        try:
            response = requests.get(url)
            data = response.json()

            if response.status_code == 200 and 'routes' in data and data['routes']:
                distance = data['routes'][0]['distance'] / 1000  # Convert to km
            else:
                print(f"Error in response: {data}")
                distance = None
        except Exception as e:
            print(f"Error occurred: {e}")
            distance = None

        return distance

    for index, row in df.iterrows():
        df.at[index, 'distance_osrm'] = road_distance_osrm(row)

        if index % 100 == 0:
            print(f"Processing row {index} of {total_rows}...")

    return df

def datetime_transform(df):

    for datetime_col in ['pickup_datetime', 'dropoff_datetime']:
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df[f'{datetime_col}_year'] = df[datetime_col].dt.year
        df[f'{datetime_col}_month'] = df[datetime_col].dt.month
        df[f'{datetime_col}_hour'] = df[datetime_col].dt.hour
        df[f'{datetime_col}_weekday'] = df[datetime_col].dt.day_name()
        df[f'{datetime_col}_period'] = pd.cut(df[f'{datetime_col}_hour'], 
                                              bins=[0, 6, 12, 16, 20, 24], 
                                              labels=['Early Morning', 'Morning', 'Afternoon', 'Evening', 'Night'], 
                                              right=False, 
                                              include_lowest=True)

    df.drop(columns=['pickup_datetime', 'dropoff_datetime'], inplace=True)

    return df

def add_plot(plot1, plot2, name1, name2, filename='plot.png', x_limits=None, y_limits=None):

    # Derive title from the filename (remove extension)
    title = os.path.splitext(filename)[0]

    fig, ax = plt.subplots(figsize=(7, 7))

    plot1.plot(ax=ax, color='blue', label=name1)
    plot2.plot(ax=ax, color='red', label=name2)

    if x_limits is not None:
        ax.set_xlim(x_limits)
    if y_limits is not None:
        ax.set_ylim(y_limits)

    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.show()

def plot_vendor_trip_duration(df):
    df_hour = df.copy()
    
    df_hour['trip_duration_hours'] = df_hour['trip_duration'] / 3600
    
    vendor_trip_duration = df_hour.groupby('vendor_id')['trip_duration_hours'].sum().reset_index()
    
    plt.figure(figsize=(8, 6))
    bar_plot = sns.barplot(x='vendor_id', y='trip_duration_hours', data=vendor_trip_duration, palette='viridis')
    
    plt.xlabel('Vendor ID')
    plt.ylabel('Total Trip Duration (hours)')
    plt.title('Total Trip Duration by Vendor in 2016 (in Hours)')
    
    for p in bar_plot.patches:
        bar_plot.annotate(format(p.get_height(), '.0f'), 
                          (p.get_x() + p.get_width() / 2., p.get_height()), 
                          ha='center', va='center', 
                          xytext=(0, 9), 
                          textcoords='offset points')
    
    plt.show()

def plot_taxi_trips_per_vendor(df):
    taxi_trips = df.groupby('vendor_id')['id'].count().reset_index()
    taxi_trips.rename(columns={'id': 'trip_count'}, inplace=True)
    
    plt.figure(figsize=(8, 6))
    bar_plot = sns.barplot(x='vendor_id', y='trip_count', data=taxi_trips, palette='magma')
    
    plt.title('Number of Taxi Trips per Vendor in 2016')
    plt.ylabel('Number of Trips')
    plt.xlabel('Vendor ID')
    
    for p in bar_plot.patches:
        bar_plot.annotate(format(p.get_height(), '.0f'),
                          (p.get_x() + p.get_width() / 2., p.get_height()),
                          ha='center', va='center',
                          xytext=(0, 9),
                          textcoords='offset points')
    
    plt.show()

def plot_trip_counts_by_hour(df):
    pickup_counts = df.groupby('pickup_datetime_hour')['id'].count().reset_index()
    pickup_counts.rename(columns={'id': 'pickup_count'}, inplace=True)

    dropoff_counts = df.groupby('dropoff_datetime_hour')['id'].count().reset_index()
    dropoff_counts.rename(columns={'id': 'dropoff_count'}, inplace=True)

    combined_counts = pd.merge(pickup_counts, dropoff_counts, 
                                left_on='pickup_datetime_hour', 
                                right_on='dropoff_datetime_hour', 
                                how='outer').fillna(0)

    combined_counts = combined_counts.melt(id_vars='pickup_datetime_hour', 
                                           value_vars=['pickup_count', 'dropoff_count'], 
                                           var_name='trip_type', 
                                           value_name='count')

    order = list(range(24))
    labels = [f'{hour}:00' for hour in order]

    combined_counts['ordered_labels'] = pd.Categorical(combined_counts['pickup_datetime_hour'], 
                                                       categories=order, 
                                                       ordered=True)

    palette = {'pickup_count': '#FF5733', 'dropoff_count': '#33C1FF'}

    combined_counts['trip_type'] = combined_counts['trip_type'].replace({'pickup_count': 'Pickup', 'dropoff_count': 'Drop-off'})

    palette = {'Pickup': '#FF5733', 'Drop-off': '#33C1FF'}

    plt.figure(figsize=(12, 6))
    bar_plot = sns.barplot(x='ordered_labels', y='count', hue='trip_type', data=combined_counts, palette=palette)

    bar_plot.set_xticklabels(labels, rotation=45)

    for p in bar_plot.patches:
        bar_plot.annotate(format(p.get_height(), '.0f'),
                          (p.get_x() + p.get_width() / 2., p.get_height()),
                          ha='center', va='center',
                          xytext=(0, 5),
                          textcoords='offset points')

    plt.title('Total Taxi Trips by Hour of Day (2016)')
    plt.xlabel('Hour of Day')
    plt.ylabel('Total Trips Count')
    plt.legend(title='Trip Type')
    plt.show()
    
def plot_taxi_trips_by_time_of_day(df):
    pickup_counts = df.groupby('pickup_datetime_period')['id'].count().reset_index()
    pickup_counts.rename(columns={'id': 'pickup_count'}, inplace=True)

    dropoff_counts = df.groupby('dropoff_datetime_period')['id'].count().reset_index()
    dropoff_counts.rename(columns={'id': 'dropoff_count'}, inplace=True)

    combined_counts = pd.merge(pickup_counts, dropoff_counts, 
                               left_on='pickup_datetime_period', 
                               right_on='dropoff_datetime_period', 
                               how='outer').fillna(0)

    combined_counts = combined_counts.melt(id_vars='pickup_datetime_period', 
                                           value_vars=['pickup_count', 'dropoff_count'], 
                                           var_name='trip_type', 
                                           value_name='count')

    combined_counts['trip_type'].replace({'pickup_count': 'Pickup', 'dropoff_count': 'Drop-off'}, inplace=True)

    order = ['Early Morning', 'Morning', 'Afternoon', 'Evening', 'Night']
    labels = [
        'Early Morning\n(00:00-06:00)',
        'Morning\n(06:00-12:00)',
        'Afternoon\n(12:00-16:00)',
        'Evening\n(16:00-20:00)',
        'Night\n(20:00-24:00)'
    ]

    combined_counts['ordered_labels'] = pd.Categorical(combined_counts['pickup_datetime_period'], 
                                                       categories=order, 
                                                       ordered=True)

    palette = {'Pickup': '#009688', 'Drop-off': '#FF9800'}

    plt.figure(figsize=(10, 6))
    bar_plot = sns.barplot(x='ordered_labels', y='count', hue='trip_type', data=combined_counts, palette=palette)

    bar_plot.set_xticklabels(labels)

    for p in bar_plot.patches:
        bar_plot.annotate(format(p.get_height(), '.0f'),
                          (p.get_x() + p.get_width() / 2., p.get_height()),
                          ha='center', va='center',
                          xytext=(0, 5),
                          textcoords='offset points')

    plt.title('Total Taxi Trips by Time of Day (2016)')
    plt.xlabel('Time of Day')
    plt.ylabel('Total Trips Count')
    plt.legend(title='Trip Type')
    plt.show()

def plot_pickup_by_weekday(df):
    pickup_counts = df.groupby('pickup_datetime_weekday')['id'].count().reset_index()
    pickup_counts.rename(columns={'id': 'pickup_count'}, inplace=True)

    order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pickup_counts['pickup_datetime_weekday'] = pd.Categorical(pickup_counts['pickup_datetime_weekday'], 
                                                             categories=order, 
                                                             ordered=True)

    plt.figure(figsize=(10, 6))
    bar_plot = sns.barplot(x='pickup_datetime_weekday', y='pickup_count', data=pickup_counts, color='#D32F2F')

    bar_plot.set_xticklabels(order)

    for p in bar_plot.patches:
        bar_plot.annotate(format(p.get_height(), '.0f'),
                          (p.get_x() + p.get_width() / 2., p.get_height()),
                          ha='center', va='center',
                          xytext=(0, 5),
                          textcoords='offset points')

    plt.title('Total Taxi Trips by Weekday (2016) - Based on Pickup Data')
    plt.xlabel('Weekday')
    plt.ylabel('Total Trips Count')
    plt.show()
    
def plot_pickup_by_month(df):
    pickup_counts = df.groupby('pickup_datetime_month')['id'].count().reset_index()
    pickup_counts.rename(columns={'id': 'pickup_count'}, inplace=True)
    pickup_counts = pickup_counts[pickup_counts['pickup_datetime_month'] <= 6]
    
    order = [1, 2, 3, 4, 5, 6]
    labels = ['January', 'February', 'March', 'April', 'May', 'June']
    
    pickup_counts['ordered_labels'] = pd.Categorical(pickup_counts['pickup_datetime_month'], 
                                                     categories=order, 
                                                     ordered=True)
    
    plt.figure(figsize=(10, 6))
    bar_plot = sns.barplot(x='ordered_labels', y='pickup_count', data=pickup_counts, color='#FF7043')
    bar_plot.set_xticklabels(labels)
    
    for p in bar_plot.patches:
        bar_plot.annotate(format(p.get_height(), '.0f'),
                          (p.get_x() + p.get_width() / 2., p.get_height()),
                          ha='center', va='center',
                          xytext=(0, 5),
                          textcoords='offset points')
    
    plt.title('Total Taxi Trips by Month (2016) - Based on Pickup Data')
    plt.xlabel('Month')
    plt.ylabel('Total Trips Count')
    plt.show()

def plot_passenger_trip_duration(df):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['passenger_count'], df['trip_duration'], alpha=0.6)
    plt.title('Relationship between Number of Passengers and Trip Duration')
    plt.xlabel('Passenger Count')
    plt.ylabel('Trip Duration (seconds)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def plot_passenger_vendor(df):
    trip_counts = df.groupby(['passenger_count', 'vendor_id']).size().reset_index(name='trip_count')
    trip_counts_pivot = trip_counts.pivot(index='passenger_count', columns='vendor_id', values='trip_count').fillna(0)
    plt.figure(figsize=(10, 6))
    bar_plot = trip_counts_pivot.plot(kind='bar', width=0.8, figsize=(10, 6))
    plt.xlabel('Number of Passengers')
    plt.ylabel('Total Trips Count')
    plt.title('Relation Between Number of Passengers and Available Vendors')
    plt.legend(['Vendor 1', 'Vendor 2'])
    plt.xticks(rotation=0)
    labels = [str(int(label)) for label in trip_counts_pivot.index]
    bar_plot.set_xticklabels(labels)
    for p in bar_plot.patches:
        bar_plot.annotate(format(p.get_height(), '.0f'),
                          (p.get_x() + p.get_width() / 2., p.get_height()),
                          ha='center', va='center',
                          xytext=(0, 5),
                          textcoords='offset points')
    plt.tight_layout()
    plt.show()

def create_heatmap(df, lat_col, lon_col, map_title):
    # Original center of the map at New York City (latitude=40.7128, longitude=-74.0060)
    map_center = [40.7128, -74.0060]

    heatmap_map = folium.Map(location=map_center, zoom_start=11)  # default = 12

    heatmap_data = [[row[lat_col], row[lon_col]] for _, row in df.iterrows()]

    print("in progress ...")

    heatmap_layer = folium.FeatureGroup(name='HeatMap', show=True)
    HeatMap(heatmap_data).add_to(heatmap_layer)
    heatmap_layer.add_to(heatmap_map)

    marker_cluster_layer = folium.FeatureGroup(name='Markers', show=True)
    marker_cluster = MarkerCluster().add_to(marker_cluster_layer)

    for _, row in df.iterrows():
        folium.Marker(location=[row[lat_col], row[lon_col]]).add_to(marker_cluster)

    marker_cluster_layer.add_to(heatmap_map)

    folium.LayerControl(collapsed=False).add_to(heatmap_map)

    heatmap_map.save(f"{map_title}.html")

    print("done!")
    
    return heatmap_map
 
def plot_store_and_fwd_trip_counts(df):
    store_and_fwd_trip_counts = df.groupby('store_and_fwd_flag')['trip_duration'].count().reset_index()
    store_and_fwd_trip_counts.columns = ['store_and_fwd_flag', 'Count']

    total_count = store_and_fwd_trip_counts['Count'].sum()
    store_and_fwd_trip_counts['Percentage'] = (store_and_fwd_trip_counts['Count'] / total_count) * 100

    fig, ax = plt.subplots()
    colors = ['skyblue', 'lightgreen']
    bars = ax.bar(store_and_fwd_trip_counts['store_and_fwd_flag'], store_and_fwd_trip_counts['Count'], color=colors)

    for bar, count, percentage in zip(bars, store_and_fwd_trip_counts['Count'], store_and_fwd_trip_counts['Percentage']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{count} ({percentage:.1f}%)', 
                ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('store_and_fwd_flag')
    ax.set_ylabel('Count')
    ax.set_title('Count of Trips by Store and Forward Flag')

    plt.show()

def plot_store_and_fwd_trip_duration(df):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='store_and_fwd_flag', y='trip_duration', data=df)
    plt.title('Effect of Store-and-Forward Flag on Taxi Trip Duration')
    plt.xlabel('Store and Forward Flag')
    plt.ylabel('Trip Duration (seconds)')
    plt.show()

def sample_clusters(df, cluster_col, n):
    sampled_df = df.groupby(cluster_col).apply(lambda x: x.sample(n=min(n, len(x)))).reset_index(drop=True)
    return sampled_df
    
def create_cluster_map(df, cluster_col, lat_col, lon_col, map_title='Cluster Map'):

    # Center the map at New York City
    nyc_lat = 40.7128
    nyc_lon = -74.0060
    
    folium_map = folium.Map(location=[nyc_lat, nyc_lon], zoom_start=11)
    
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen']

    cluster_group = folium.FeatureGroup(name='Clusters', show=True).add_to(folium_map)
    grouped = df.groupby(cluster_col)
    
    for cluster, group in grouped:
        for lat, lon in zip(group[lat_col], group[lon_col]):
            folium.CircleMarker(
                location=(lat, lon),
                radius=3,
                color=colors[cluster % len(colors)],
                fill=True,
                fill_opacity=0.7
            ).add_to(cluster_group)
    
    folium.LayerControl().add_to(folium_map)
    output_file = f"{map_title.replace(' ', '_').lower()}.html"
    folium_map.save(output_file)
    print(f"Map saved as {output_file}")
    
    return folium_map    
    
def qqplot(df_old, df_new, column_old, column_new):

    original_data = df_old[column_old]
    transformed_data = df_new[column_new]
    
    # Original data
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    sns.histplot(original_data, bins=30, kde=True)
    plt.title('Original Data Distribution')
    plt.xlabel(column_old)
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    stats.probplot(original_data, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Original Data')

    plt.tight_layout()
    plt.show()

    original_skewness = original_data.skew()
    print(f'Original Skewness: {original_skewness:.2f}')
    
    # Transformed data
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    sns.histplot(transformed_data, bins=30, kde=True)
    plt.title('Transformed Data Distribution')
    plt.xlabel(column_new)
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    stats.probplot(transformed_data, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Transformed Data')

    plt.tight_layout()
    plt.show()

    transformed_skewness = transformed_data.skew()
    print(f'Transformed Skewness: {transformed_skewness:.2f}')

# Outliers subplot
def style_axis(ax, ylabel, title=None):
    ax.set_ylabel(ylabel, fontsize=14)
    if title is not None:
        ax.set_title(title, fontsize=16)
    ax.grid(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    
def outliers_graph(df, cols, df2=None, col2=None, name1="Data 1", name2="Data 2"):
    # Filter numeric columns only
    numeric_cols = [col for col in cols if pd.api.types.is_numeric_dtype(df[col])]
    
    # Do the same for df2 if provided
    if df2 is not None and col2 is not None:
        numeric_col2 = [col for col in col2 if pd.api.types.is_numeric_dtype(df2[col])]
    else:
        numeric_col2 = []

    # Check if there are any numeric columns to plot
    if len(numeric_cols) == 0:
        print("No numeric columns to plot in the first dataset.")
        return
    if df2 is not None and len(numeric_col2) == 0:
        print("No numeric columns to plot in the second dataset.")
        return

    sns.set_style("whitegrid")
    custom_palette = sns.color_palette("muted", max(len(numeric_cols), 2))
    ncols = min(len(numeric_cols), 4)
    nrows = math.ceil(len(numeric_cols) / ncols)

    if df2 is not None and col2 is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))
        sns.boxplot(y=df[numeric_cols[0]], ax=ax1, color=custom_palette[0])
        sns.boxplot(y=df2[numeric_col2[0]], ax=ax2, color=custom_palette[1])

        style_axis(ax1, numeric_cols[0], f"{name1}")
        style_axis(ax2, numeric_col2[0], f"{name2}")

    else:
        # Plot for a single dataset
        if len(numeric_cols) == 1:
            fig, ax = plt.subplots(figsize=(4, 6))
            sns.boxplot(y=df[numeric_cols[0]], ax=ax, color=custom_palette[0])
            style_axis(ax, numeric_cols[0])
        else:
            fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows))
            ax = ax.flatten()

            for i in range(len(numeric_cols)):
                sns.boxplot(y=df[numeric_cols[i]], ax=ax[i], color=custom_palette[i])
                style_axis(ax[i], numeric_cols[i])

    # Remove unused subplots if any
    for j in range(len(numeric_cols), nrows * ncols):
        fig.delaxes(ax[j])

    fig.tight_layout(w_pad=5.0)
    plt.show()
    
def num_corr(df):
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    correlation = df.corr(numeric_only=True)
    mask = np.zeros_like(correlation, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(correlation, annot=True, mask=mask, cmap='coolwarm', annot_kws={"size": 11})
    sns.despine(left=True, bottom=True)
    plt.grid(False)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.title("Correlation Matrix", fontsize=15, fontweight='bold')  

def correlation_strength(corr):
    if corr >= 0.7 or corr <= -0.7:
        return 'Strong'
    elif 0.3 < corr < 0.7 or -0.7 < corr < -0.3:
        return 'Moderate'
    else:
        return 'Weak'
    
def get_ranked_correlations(df):

    correlation_matrix = df.corr(numeric_only=True)

    # Melt the correlation matrix
    tidy_correlation = correlation_matrix.reset_index().melt(id_vars='index')
    tidy_correlation.columns = ['Variable 1', 'Variable 2', 'Correlation']

    # Remove self-correlations (where Variable 1 == Variable 2)
    tidy_correlation = tidy_correlation[tidy_correlation['Variable 1'] != tidy_correlation['Variable 2']]

    # Ensure unique pairs (Variable 1, Variable 2) and (Variable 2, Variable 1) are treated the same
    tidy_correlation['Pair'] = tidy_correlation[['Variable 1', 'Variable 2']].apply(lambda x: tuple(sorted(x)), axis=1)

    # Drop duplicate pairs
    tidy_correlation = tidy_correlation.drop_duplicates(subset='Pair').drop(columns='Pair')

    # Rank the correlations based on absolute value
    tidy_correlation['AbsCorrelation'] = tidy_correlation['Correlation'].abs()

    # Sort by absolute value of correlation
    tidy_correlation = tidy_correlation.sort_values(by='AbsCorrelation', ascending=False)

    # Add rank based on absolute value of correlation (1 is highest, so negative and positive are treated equally strong)
    tidy_correlation['Rank'] = tidy_correlation['AbsCorrelation'].rank(ascending=False, method='min')

    # Add correlation strength
    tidy_correlation['Strength'] = tidy_correlation['Correlation'].apply(correlation_strength)

    # Drop the temporary column
    tidy_correlation = tidy_correlation.drop(columns=['AbsCorrelation'])

    # Sort by correlation for final output (positive/negative correlation ordering)
    tidy_correlation = tidy_correlation.sort_values(by='Rank', ascending=True)

    return tidy_correlation
    
