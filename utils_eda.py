import numpy as np
import pandas as pd
import time
import math
from math import sqrt
import os
import json

from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.metrics import mean_squared_log_error
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.base import BaseEstimator, TransformerMixin

from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns

import requests
import osmnx as ox
import networkx as nx

from geopy.distance import great_circle, geodesic
from geopy.exc import GeocoderTimedOut

import folium
from folium.plugins import HeatMap, MarkerCluster

#########################################################################################################################

def calculate_distance_gc(row):
    pickup = (row['pickup_latitude'], row['pickup_longitude'])
    dropoff = (row['dropoff_latitude'], row['dropoff_longitude'])

    try:
        distance = great_circle(pickup, dropoff).km  # or .miles for miles
    except GeocoderTimedOut:
        distance = None

    return distance

def to_speed(df, distance_column='distance_osrm', speed_column='speed_osrm', time_column='trip_duration'):
    df_speed = df.copy()
    
    # Convert trip_duration to hours
    df_speed['trip_duration_hour'] = df_speed[time_column] / 3600
    
    # Avoid division by zero by replacing 0 with a small value
    df_speed['trip_duration_hour'] = df_speed['trip_duration_hour'].replace(0, 1e-6)
    
    # Calculate speed based on the provided distance column
    df[speed_column] = df[distance_column] / df_speed['trip_duration_hour']
    
    return df

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

def distance_from_NYC_center(df):
    lat_col='pickup'
    lon_col='dropoff'
    map_center=[40.7128, -74.0060]

    def calculate_distance_from_NYC_center(row):
        pickup_distance = geodesic(map_center, (row[f"{lat_col}_latitude"], row[f"{lat_col}_longitude"])).km
        dropoff_distance = geodesic(map_center, (row[f"{lon_col}_latitude"], row[f"{lon_col}_longitude"])).km
        return pickup_distance, dropoff_distance

    df_nyc = df.copy()
    distances = df_nyc.apply(calculate_distance_from_NYC_center, axis=1)
    df_nyc[['pickup_dist_NYC_center', 'dropoff_dist_NYC_center']] = pd.DataFrame(distances.tolist(), index=df_nyc.index)
    
    return df_nyc

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

class OutlierMapper(BaseEstimator, TransformerMixin):
    def __init__(self, lat_col="pickup", lon_col="dropoff", map_title="nyc_outliers_map", radius_km=50,
                 csv_dir="csv_eda", generate_map=True):
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.map_title = map_title
        self.radius_km = radius_km
        self.csv_dir = csv_dir
        self.generate_map = generate_map
        self.csv_filename = f"{self.map_title.replace(' ', '_').lower()}.csv"
        self.map_center = [40.7128, -74.0060]  # NYC center

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print("--- Identifying Outliers on the Map ---")
        
        X_map = X.copy()
        y_map = y.copy() if y is not None else None
        initial_count = len(X_map)

        # Ensure required columns are present
        if 'pickup_dist_NYC_center' not in X_map.columns or 'dropoff_dist_NYC_center' not in X_map.columns:
            raise ValueError("X must contain 'pickup_dist_NYC_center' and 'dropoff_dist_NYC_center' columns.")

        # Identify outliers based on radius
        if y_map is not None:   
            df_outliers = pd.concat([X_map, y_map], axis=1)
            df_outliers['is_outlier'] = (df_outliers['pickup_dist_NYC_center'] > self.radius_km) | (df_outliers['dropoff_dist_NYC_center'] > self.radius_km)
            outliers = df_outliers[df_outliers['is_outlier']]
        else:
            X_map['is_outlier'] = (X_map['pickup_dist_NYC_center'] > self.radius_km) | (X_map['dropoff_dist_NYC_center'] > self.radius_km)
            outliers = X_map[X_map['is_outlier']]

        # Save outliers to CSV
        os.makedirs(self.csv_dir, exist_ok=True)
        csv_path = os.path.join(self.csv_dir, self.csv_filename)
        outliers.to_csv(csv_path, index=False)
        print(f"Outliers saved to '{csv_path}'")

        # Filter out outliers
        if y_map is not None: 
            X_cleaned = df_outliers[~df_outliers['is_outlier']].drop(columns=['is_outlier', 'trip_duration']).reset_index(drop=True)
            y_map = y_map[~df_outliers['is_outlier']].reset_index(drop=True)
        else: 
            X_cleaned = X_map[~X_map['is_outlier']].drop(columns=['is_outlier']).reset_index(drop=True)

        # Generate the map if enabled
        if self.generate_map:
            self._generate_map(outliers)

        # Drop distance columns after filtering
        X_cleaned.drop(columns=['pickup_dist_NYC_center', 'dropoff_dist_NYC_center'], inplace=True)

        removed_count = initial_count - len(X_cleaned)
        print(f"Removed {removed_count} ({removed_count / initial_count * 100:.2f}%) records outside NYC boundaries.")

        return (X_cleaned, y_map)

    def _generate_map(self, outliers):
        nyc_map = folium.Map(location=self.map_center, zoom_start=10)
        folium.Circle(
            location=self.map_center,
            radius=self.radius_km * 1000,  # Convert to meters
            color="blue",
            fill=True,
            fill_opacity=0.1,
            popup="NYC Boundary (50 km radius)"
        ).add_to(nyc_map)

        for _, row in outliers.iterrows():
            if row['pickup_dist_NYC_center'] > self.radius_km:
                pickup_coords = (row[f"{self.lat_col}_latitude"], row[f"{self.lon_col}_longitude"])
                folium.Marker(
                    location=pickup_coords,
                    icon=folium.Icon(color="red"),
                    popup=f"Outlier ID: {row['id']} (Pickup)"
                ).add_to(nyc_map)
            
            if row['dropoff_dist_NYC_center'] > self.radius_km:
                dropoff_coords = (row[f"{self.lon_col}_latitude"], row[f"{self.lon_col}_longitude"])
                folium.Marker(
                    location=dropoff_coords,
                    icon=folium.Icon(color="red"),
                    popup=f"Outlier ID: {row['id']} (Dropoff)"
                ).add_to(nyc_map)

        os.makedirs("html", exist_ok=True)
        file_name = f"html/{self.map_title.replace(' ', '_').lower()}.html"
        nyc_map.save(file_name)
        print(f"Map saved as '{file_name}'")

def elbow_method(coords, title='Elbow Method', k_range=range(1, 11)):
    inertia = []

    for k in k_range:
        print(f'In progress... k = {k}')
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(coords)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertia, marker='o')
    plt.title(f'{title}')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')    
    plt.xticks(k_range)    
    plt.show()

def slovin(population):
    sample = math.ceil(len(population) / (1 + len(population) * 0.01**2))
    print("Sample size based on Slovin's calculation:", sample)
    return sample

def sample_clusters_random(df, cluster_col, n):
    sampled_df = df.groupby(cluster_col).apply(lambda x: x.sample(n=min(n, len(x)), random_state = 42)).reset_index(drop=True)
    return sampled_df   
    
def plot_cluster_map(df, cluster_type='pickup', output_file='html/cluster_map.html'):
    map_center = [40.7128, -74.0060]  # NYC coordinates
    zoom_start = 10

    m = folium.Map(location=map_center, zoom_start=zoom_start)

    radius_m = 50000  # radius in meters (50 km)
    folium.Circle(
        location=map_center,
        radius=radius_m,
        color="blue",
        popup="NYC Boundary (50 km radius)"
    ).add_to(m)

    if cluster_type == 'pickup':
        lon_col = 'pickup_longitude'
        lat_col = 'pickup_latitude'
        cluster_col = 'pickup_cluster'
        colors = {0: 'orange', 1: 'darkblue', 2: 'darkgreen', 3: 'darkred'}
    elif cluster_type == 'dropoff':
        lon_col = 'dropoff_longitude'
        lat_col = 'dropoff_latitude'
        cluster_col = 'dropoff_cluster'
        colors = {0: 'black', 1: 'teal', 2: 'purple'}
    else:
        raise ValueError("cluster_type must be either 'pickup' or 'dropoff'")

    marker_group = folium.FeatureGroup(name='Markers')
    for i, row in df.iterrows():
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=3,
            color=colors[row[cluster_col]],
            fill=True,
            fill_color=colors[row[cluster_col]],
            fill_opacity=0.1,
            opacity=0.7
        ).add_to(marker_group)

    marker_group.add_to(m)
    folium.LayerControl().add_to(m)

    m.save(output_file)
    print(f"Map has been saved to {output_file}")

def plot_cluster(df, cluster_type='pickup'):
    if cluster_type == 'pickup':
        longitude = 'pickup_longitude'
        latitude = 'pickup_latitude'
        cluster = 'pickup_cluster'
        title = 'Pickup Points by Cluster'
        colors = {0: 'orange', 1: 'darkblue', 2: 'darkgreen', 3: 'darkred'}
        marker = 'o'
    elif cluster_type == 'dropoff':
        longitude = 'dropoff_longitude'
        latitude = 'dropoff_latitude'
        cluster = 'dropoff_cluster'
        title = 'Dropoff Points by Cluster'
        colors = {0: 'black', 1: 'teal', 2: 'purple'}
        marker = 'o'
    else:
        raise ValueError("Invalid cluster_type. Choose 'pickup' or 'dropoff'.")

    color_list = df[cluster].map(colors)

    plt.figure(figsize=(7, 7))
    
    plt.scatter(df[longitude], df[latitude], 
                c=color_list, label=f'{cluster_type.capitalize()} Points', 
                marker=marker, alpha=0.3)
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(title)
    
    handles = [plt.Line2D([0], [0], marker=marker, color='w', label=f'Cluster {i}', 
                           markerfacecolor=colors[i], markersize=10) for i in colors.keys()]
    plt.legend(handles=handles, title='Clusters', loc="upper left")
    
    # Add accurate circular boundary around New York City
    nyc_lat, nyc_lon = 40.7128, -74.0060
    radius_km = 50  # Radius in kilometers
    
    num_points = 100
    angles = np.linspace(0, 2 * np.pi, num_points)
    latitudes = nyc_lat + (radius_km / 111) * np.sin(angles)
    longitudes = nyc_lon + (radius_km / (111 * np.cos(np.radians(nyc_lat)))) * np.cos(angles)
    
    plt.plot(longitudes, latitudes, color='blue', linestyle='-', linewidth=2)

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
    plt.xticks(fontsize=11, rotation=45)
    plt.yticks(fontsize=11, rotation=45)
    plt.title("Correlation Matrix", fontsize=16, fontweight='bold')

def correlation_strength(corr):
    if corr >= 0.7 or corr <= -0.7:
        return 'Strong'
    elif 0.3 < corr < 0.7 or -0.7 < corr < -0.3:
        return 'Moderate'
    else:
        return 'Weak'
    
def get_ranked_correlations(df):
    correlation_matrix = df.corr(numeric_only=True)

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
    
def calc_vif(df, target_variable):
    numeric_data = df.select_dtypes(include="number")
    
    if target_variable in numeric_data.columns:
        numeric_data = numeric_data.drop(columns=[target_variable])
    
    vif = pd.DataFrame()
    vif["Features"] = numeric_data.columns
    vif["VIF"] = [variance_inflation_factor(numeric_data.values, i) for i in range(numeric_data.shape[1])]

    vif = vif.sort_values(by="VIF", ascending=False).reset_index(drop=True)

    return vif    
    
#########################################################################################################################

def plot_trip_duration_vs_distance(df):
    df_dist = df.copy()
    df_dist['trip_duration_hours'] = df_dist['trip_duration'] / 3600
    
    plt.figure(figsize=(8, 6))
    plt.scatter(df_dist['distance_osrm'], df_dist['trip_duration_hours'], alpha=0.3)
    plt.title("Relationship between Trip Duration and Distance", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Distance (km)")
    plt.ylabel("Trip Duration (hours)")
    plt.grid(True)
    plt.show()

def plot_vendor_trip_duration(df):
    df_hour = df.copy()
    df_hour['trip_duration_hours'] = df_hour['trip_duration'] / 3600
    
    vendor_trip_duration = df_hour.groupby('vendor_id')['trip_duration_hours'].sum().reset_index()
    
    plt.figure(figsize=(8, 6))
    bar_plot = sns.barplot(x='vendor_id', y='trip_duration_hours', data=vendor_trip_duration, palette='viridis')
    plt.xlabel('Vendor ID')
    plt.ylabel('Total Trip Duration (hours)')
    plt.title('Total Trip Duration by Vendor in 2016 (in Hours)', fontsize=14, fontweight='bold', pad=15)
    
    for p in bar_plot.patches:
        bar_plot.annotate(format(p.get_height(), '.0f'), 
                          (p.get_x() + p.get_width() / 2., p.get_height()), 
                          ha='center', va='center', 
                          xytext=(0, 9), 
                          textcoords='offset points')
        
    max_y = vendor_trip_duration['trip_duration_hours'].max()
    padding = max_y * 0.1
    plt.ylim(0, max_y + padding)
    
    plt.show()
    
def plot_total_distance_by_vendor(df):
    total_distance_by_vendor = df.groupby('vendor_id')['distance_osrm'].sum()

    plt.figure(figsize=(8,6))
    ax = total_distance_by_vendor.plot(kind='bar', color=['blue', 'orange'])
    plt.title('Total Distance by Vendor in 2016', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Vendor ID')
    plt.ylabel('Total Distance (km)')
    plt.xticks(rotation=0)

    for p in ax.patches:
        ax.annotate(f'{p.get_height():.0f}',
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 9), 
                    textcoords='offset points')
    
    max_y = total_distance_by_vendor.max()
    padding = max_y * 0.1
    plt.ylim(0, max_y + padding)
    
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

    plt.figure(figsize=(15, 6))
    bar_plot = sns.barplot(x='ordered_labels', y='count', hue='trip_type', data=combined_counts, palette=palette)

    bar_plot.set_xticklabels(labels, rotation=0)

    for p in bar_plot.patches:
        bar_plot.annotate(format(p.get_height(), '.0f'),
                          (p.get_x() + p.get_width() / 2., p.get_height()),
                          ha='center', va='center',
                          xytext=(0, 5),
                          textcoords='offset points')

    max_y = combined_counts['count'].max()
    padding = max_y * 0.1
    plt.ylim(0, max_y + padding)
        
    plt.title('Total Taxi Trips by Hour of Day (2016)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Hour of Day')
    plt.ylabel('Total Trips Count')
    plt.legend(title='Trip Type', loc='upper left')
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

    max_y = combined_counts['count'].max()
    padding = max_y * 0.1
    plt.ylim(0, max_y + padding)
        
    plt.title('Total Taxi Trips by Time of Day (2016)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Time of Day')
    plt.ylabel('Total Trips Count')
    plt.legend(title='Trip Type', loc='upper left')
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

    max_y = pickup_counts['pickup_count'].max()
    padding = max_y * 0.1
    plt.ylim(0, max_y + padding)
        
    plt.title('Total Taxi Trips by Weekday (2016) - Based on Pickup Data', fontsize=14, fontweight='bold', pad=15)
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
    
    max_y = pickup_counts['pickup_count'].max()
    padding = max_y * 0.1
    plt.ylim(0, max_y + padding)
    
    plt.title('Total Taxi Trips by Month (2016) - Based on Pickup Data', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Month')
    plt.ylabel('Total Trips Count')
    plt.show()  
        
def plot_average_distance_covered_by_hour(df):
    pickup_distances = df.groupby('pickup_datetime_hour')['distance_osrm'].mean().reset_index()
    pickup_distances.rename(columns={'distance_osrm': 'pickup_distance'}, inplace=True)

    dropoff_distances = df.groupby('dropoff_datetime_hour')['distance_osrm'].mean().reset_index()
    dropoff_distances.rename(columns={'distance_osrm': 'dropoff_distance'}, inplace=True)

    combined_distances = pd.merge(pickup_distances, dropoff_distances, 
                                  left_on='pickup_datetime_hour', 
                                  right_on='dropoff_datetime_hour', 
                                  how='outer').fillna(0)

    combined_distances = combined_distances.melt(id_vars='pickup_datetime_hour', 
                                                 value_vars=['pickup_distance', 'dropoff_distance'], 
                                                 var_name='trip_type', 
                                                 value_name='distance')

    combined_distances['trip_type'] = combined_distances['trip_type'].replace({'pickup_distance': 'Pickup', 'dropoff_distance': 'Drop-off'})

    order = list(range(24))
    labels = [f'{hour}:00' for hour in order]

    combined_distances['ordered_labels'] = pd.Categorical(combined_distances['pickup_datetime_hour'], 
                                                          categories=order, 
                                                          ordered=True)

    palette = {'Pickup': '#4CAF50', 'Drop-off': '#FFC107'}

    plt.figure(figsize=(15, 6))
    bar_plot = sns.barplot(x='ordered_labels', y='distance', hue='trip_type', data=combined_distances, palette=palette)

    bar_plot.set_xticklabels(labels, rotation=0)

    for p in bar_plot.patches:
        bar_plot.annotate(format(p.get_height(), '.2f'),
                          (p.get_x() + p.get_width() / 2., p.get_height()),
                          ha='center', va='center',
                          xytext=(0, 5),
                          textcoords='offset points')

    max_y = combined_distances['distance'].max()
    padding = max_y * 0.1
    plt.ylim(0, max_y + padding)
        
    plt.title('Average Distance by Hour of Day (2016)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Distance (km)')
    plt.legend(title='Trip Type', loc='upper right')
    plt.show()

def plot_average_speed_by_hour(df):
    pickup_speeds = df.groupby('pickup_datetime_hour')['speed_osrm'].mean().reset_index()
    pickup_speeds.rename(columns={'speed_osrm': 'pickup_speed'}, inplace=True)

    dropoff_speeds = df.groupby('dropoff_datetime_hour')['speed_osrm'].mean().reset_index()
    dropoff_speeds.rename(columns={'speed_osrm': 'dropoff_speed'}, inplace=True)

    combined_speeds = pd.merge(pickup_speeds, dropoff_speeds, 
                               left_on='pickup_datetime_hour', 
                               right_on='dropoff_datetime_hour', 
                               how='outer').fillna(0)

    combined_speeds = combined_speeds.melt(id_vars='pickup_datetime_hour', 
                                           value_vars=['pickup_speed', 'dropoff_speed'], 
                                           var_name='trip_type', 
                                           value_name='speed')

    combined_speeds['trip_type'] = combined_speeds['trip_type'].replace({'pickup_speed': 'Pickup', 'dropoff_speed': 'Drop-off'})

    order = list(range(24))
    labels = [f'{hour}:00' for hour in order]

    combined_speeds['ordered_labels'] = pd.Categorical(combined_speeds['pickup_datetime_hour'], 
                                                       categories=order, 
                                                       ordered=True)

    palette = {'Pickup': 'purple', 'Drop-off': 'orange'}

    plt.figure(figsize=(15, 6))
    bar_plot = sns.barplot(x='ordered_labels', y='speed', hue='trip_type', data=combined_speeds, palette=palette)

    bar_plot.set_xticklabels(labels, rotation=0)

    for p in bar_plot.patches:
        bar_plot.annotate(format(p.get_height(), '.2f'),
                          (p.get_x() + p.get_width() / 2., p.get_height()),
                          ha='center', va='center',
                          xytext=(0, 5),
                          textcoords='offset points')

    max_y = combined_speeds['speed'].max()
    padding = max_y * 0.1
    plt.ylim(0, max_y + padding)
        
    plt.title('Average Speed by Hour of Day (2016)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Speed (km/h)')
    plt.legend(title='Trip Type', loc='upper right')
    plt.show()

def plot_passenger_trip_duration(df):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['passenger_count'], df['trip_duration'], alpha=0.3)
    plt.title('Relationship between Number of Passengers and Trip Duration', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Passenger Count')
    plt.ylabel('Trip Duration (seconds)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()   
    
def plot_passenger_vendor(df):
    trip_counts = df.groupby(['passenger_count', 'vendor_id']).size().reset_index(name='trip_count')
    trip_counts_pivot = trip_counts.pivot(index='passenger_count', columns='vendor_id', values='trip_count').fillna(0)

    bar_plot = trip_counts_pivot.plot(kind='bar', width=0.8, figsize=(12, 6))
    
    plt.xlabel('Number of Passengers')
    plt.ylabel('Total Trips Count')
    plt.title('Relation Between Number of Passengers and Available Vendors', fontsize=14, fontweight='bold', pad=15)
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

    max_y = trip_counts_pivot.max().max()
    padding = max_y * 0.1
    plt.ylim(0, max_y + padding)
        
    plt.tight_layout()
    plt.show()

def create_heatmap(df, lat_col, lon_col, map_title):
    # Map center set to New York City coordinates
    map_center = [40.7128, -74.0060]

    heatmap_map = folium.Map(location=map_center, zoom_start=10, title=map_title)

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

    radius_m = 50000  # radius in meters (50 km)
    folium.Circle(
        location=map_center,
        radius=radius_m,
        color="blue",
        popup="NYC Boundary (50 km radius)"
    ).add_to(heatmap_map)

    # Add layer control to toggle heatmap and marker layers
    folium.LayerControl(collapsed=False).add_to(heatmap_map)

    heatmap_map.save(f"html/{map_title}.html")

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

    max_y = store_and_fwd_trip_counts['Count'].max()
    padding = max_y * 0.1
    ax.set_ylim(0, max_y + padding)
        
    ax.set_xlabel('store_and_fwd_flag')
    ax.set_ylabel('Count')
    ax.set_title('Count of Trips by Store and Forward Flag', fontsize=14, fontweight='bold', pad=15)

    plt.show()

def plot_store_and_fwd_trip_duration(df):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='store_and_fwd_flag', y='trip_duration', data=df)
    plt.title('Effect of Store-and-Forward Flag on Taxi Trip Duration', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Store and Forward Flag')
    plt.ylabel('Trip Duration (seconds)')
    plt.show()

def plot_store_and_fwd_trip_distance(df):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='store_and_fwd_flag', y='distance_osrm', data=df, palette={'N': 'skyblue', 'Y': 'lightcoral'})
    plt.title('Effect of Store-and-Forward Flag on Taxi Trip Distance', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Store and Forward Flag')
    plt.ylabel('Distance (km)')    
    plt.show()
 