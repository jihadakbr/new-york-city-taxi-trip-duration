import numpy as np
import pandas as pd
import time
import math
from math import sqrt
import os
import json
import pickle
import joblib

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_log_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_log_error, make_scorer
from sklearn.model_selection import learning_curve, RepeatedKFold, KFold, cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.impute import SimpleImputer

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns

import requests
import osmnx as ox
import networkx as nx

import geopandas as gpd
from geopy.distance import great_circle, geodesic
from geopy.exc import GeocoderTimedOut

import folium
from folium.plugins import HeatMap, MarkerCluster
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.prepared import prep

###################################################################################################################################
###################################################################################################################################

class RemoveDuplicates(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y):
        print(">>>> Starting the process of removing duplicates ...")
        
        X_copy = X.copy()
        y_copy = y.copy()

        # Combine X and y into a single DataFrame
        df_copy = pd.concat([X_copy, y_copy], axis=1)

        # Check for duplicates
        duplicates = df_copy.duplicated().sum()
        if duplicates > 0:
            print(f"Found {duplicates} duplicates. Dropping them.")
            # Drop duplicates
            df_copy = df_copy.drop_duplicates()
        else:
            print("No duplicates found.")

        # Separate the data again into X and y
        X_copy = df_copy.iloc[:, :-1]  # All columns except the last one
        y_copy = df_copy.iloc[:, -1]  # Only the last column

        # Return the cleaned data
        return X_copy, y_copy

class ToDataTypes(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.expected_dtypes = {
            'id': 'object',
            'vendor_id': 'object',
            'pickup_datetime': 'datetime64[ns]',
            'dropoff_datetime': 'datetime64[ns]',
            'passenger_count': 'int64',
            'pickup_longitude': 'float64',
            'pickup_latitude': 'float64',
            'dropoff_longitude': 'float64',
            'dropoff_latitude': 'float64',
            'store_and_fwd_flag': 'object',
            'distance_osrm': 'float64',
            'pickup_dist_NYC_center': 'float64',
            'dropoff_dist_NYC_center': 'float64',
        }

    def fit(self, X, y=None):
        # Stateless transformer, no fitting required
        return self

    def transform(self, X, y):
        print(">>>> Starting data type conversion process...")
        
        X_copy = X.copy()
        y_copy = y.copy() if y is not None else None

        changed_columns = []

        # Check and convert X_copy columns
        for col, expected_dtype in self.expected_dtypes.items():
            if col in X_copy.columns:
                original_dtype = X_copy[col].dtype
                if col in ['pickup_datetime', 'dropoff_datetime']:
                    if not pd.api.types.is_datetime64_any_dtype(X_copy[col]):
                        X_copy[col] = pd.to_datetime(X_copy[col], errors='coerce')
                        changed_columns.append((col, original_dtype, 'datetime64[ns]'))
                elif X_copy[col].dtype != expected_dtype:
                    X_copy[col] = X_copy[col].astype(expected_dtype, errors='ignore')
                    changed_columns.append((col, original_dtype, expected_dtype))

        # Check and convert y_copy
        if y_copy is not None:
            original_dtype_y = y_copy.dtype
            if y_copy.dtype != 'int64':
                y_copy = y_copy.astype('int64', errors='ignore')
                changed_columns.append(('trip_duration', original_dtype_y, 'int64'))

        # Print changed columns with their original and converted types
        if changed_columns:
            for col, original_dtype, new_dtype in changed_columns:
                print(f"Column '{col}' changed from {original_dtype} to {new_dtype}")
        else:
            print("No columns were changed.")

        return X_copy, y_copy

class DateTimeBreak(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass  # Stateless transformer

    def fit(self, X, y=None):
        return self  # No fitting required

    def transform(self, X):
        print(">>>> Starting datetime feature extraction...")
        X_copy = X.copy()

        # Extract datetime features
        X_copy['pickup_month'] = X_copy['pickup_datetime'].dt.month.astype(str)
        X_copy['pickup_day'] = X_copy['pickup_datetime'].dt.day.astype(str)
        X_copy['pickup_day_of_week'] = X_copy['pickup_datetime'].dt.dayofweek.astype(str)
        X_copy['pickup_hour'] = X_copy['pickup_datetime'].dt.hour.astype(str)

        print("Extracted features: ['pickup_month', 'pickup_day', 'pickup_day_of_week', 'pickup_hour']")

        # Drop unnecessary columns
        X_copy.drop(['pickup_datetime', 'dropoff_datetime'], axis=1, inplace=True, errors='ignore')

        print("Dropped columns: ['pickup_datetime', 'dropoff_datetime']")

        return X_copy

class MissValInput(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.is_fitted = False
        self.medians = {}
        self.modes = {}

    def fit(self, X, y):
        X_copy = X.copy()

        # Handle missing values for columns in X with more than 5% and less than 30%
        for column in X_copy.columns:
            missing_percentage = X_copy[column].isnull().mean() * 100
            missing_count = X_copy[column].isnull().sum()

            if 5 < missing_percentage < 30:
                if pd.api.types.is_numeric_dtype(X_copy[column]):
                    median_value = X_copy[column].median()
                    print(f"Imputed {missing_count} missing values ({missing_percentage:.2f}%) in column '{column}' with median value {median_value}.")
                    self.medians[column] = median_value
                else:
                    mode_value = X_copy[column].mode()[0]
                    print(f"Imputed {missing_count} missing values ({missing_percentage:.2f}%) in column '{column}' with mode value '{mode_value}'.")
                    self.modes[column] = mode_value

        self.is_fitted = True
        return self

    def transform(self, X, y):
        print(">>>> Starting missing value imputation...")
        if not self.is_fitted:
            raise ValueError("The fit method must be called before transform.")

        X_copy = X.copy()
        y_copy = y.copy()

        initial_length = len(X_copy)

        # Handle missing values for `trip_duration` (target variable)
        if y_copy.isnull().any():
            missing_count = y_copy.isnull().sum()
            print(f"Dropped {missing_count} entries ({missing_count / len(y_copy) * 100:.2f}%) from 'trip_duration' due to missing values.")
            valid_indices = y_copy.notnull()
            X_copy = X_copy[valid_indices]
            y_copy = y_copy[valid_indices]

        # Handle missing values for all columns in X
        for column in X_copy.columns:
            missing_percentage = X_copy[column].isnull().mean() * 100
            missing_count = X_copy[column].isnull().sum()

            if 0 < missing_percentage <= 5:
                print(f"Dropped {missing_count} entries ({missing_percentage:.2f}%) from column '{column}' due to missing values.")
                valid_indices = X_copy[column].notnull()
                X_copy = X_copy[valid_indices]
                y_copy = y_copy[valid_indices]
            elif missing_percentage >= 30:
                print(f"Dropped column '{column}' due to more than 30% missing values.")
                X_copy = X_copy.drop(columns=[column])

        # Apply missing value imputation for X
        for column, median in self.medians.items():
            X_copy[column].fillna(median, inplace=True)
        for column, mode in self.modes.items():
            X_copy[column].fillna(mode, inplace=True)

        removed_data = initial_length - len(X_copy)
        removed_data_perc = (removed_data / initial_length) * 100

        print(f"Initial data length: {initial_length}")
        print(f"Removed data: {removed_data} ({removed_data_perc:.2f}%)")
        print(f"Final data length: {len(X_copy)}")

        return X_copy, y_copy

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X, y)

class SpeedDeriv(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Initialize the transformer (no hyperparameters in this case)
        pass
    
    def fit(self, X, y=None):
        # The fit method does nothing for a stateless transformer
        return self
    
    def transform(self, X, y):
        print(">>>> Starting speed derivation...")
        # Create copies of X and y (without resetting the index)
        X_copy = X.copy()
        y_copy = y.copy()
        
        # Replace 0 values in y_copy with a very small number to avoid division by zero
        y_copy = y_copy.replace(0, 1e-6)
        
        # Create the new 'trip_duration_hour' and 'speed_osrm' features
        X_copy['trip_duration_hour'] = y_copy / 3600
        X_copy['speed_osrm'] = X_copy['distance_osrm'] / X_copy['trip_duration_hour']
        
        # Drop the 'trip_duration_hour' feature after creating speed_osrm
        X_copy.drop('trip_duration_hour', axis=1, inplace=True)
        
        # Print confirmation
        print("Feature 'speed_osrm' has been created.")
        
        # Return the transformed data
        return X_copy, y_copy

class FeatureRestriction(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # No fitting required as it's a stateless transformer
        return self

    def transform(self, X, y):
        print(">>>> Starting features restriction ...")
        # Creating copies of the input data
        X_copy = X.copy()
        y_copy = y.copy()

        # Count initial length of the dataset
        initial_length = len(X_copy)
        print(f"The dataset size: {initial_length} rows")
        
        # Print initial max and min trip_duration
        print(f"trip_duration (old) -> [min, max]: [{y_copy.min()}, {y_copy.max()}]")
        
        # Restrict trip_duration
        y_copy = y_copy[(y_copy <= 86400) & (y_copy >= 60)]
        print(f"trip_duration (new) -> [min, max]: [{y_copy.min()}, {y_copy.max()}]")
        
        # Synchronize X_copy with y_copy
        X_copy = X_copy.loc[y_copy.index]

        # Print initial max and min distance_osrm
        print(f"distance_osrm (old) -> [min, max]: [{X_copy['distance_osrm'].min()}, {X_copy['distance_osrm'].max()}]")

        # Restrict distance_osrm
        X_copy = X_copy[(X_copy['distance_osrm'] <= 100) & (X_copy['distance_osrm'] >= 0.1)]
        print(f"distance_osrm (new) -> [min, max]: [{X_copy['distance_osrm'].min()}, {X_copy['distance_osrm'].max()}]")

        # Synchronize y_copy with X_copy
        y_copy = y_copy.loc[X_copy.index]

        # Check if 'speed_osrm' column exists
        if 'speed_osrm' in X_copy.columns:
            # Print initial max and min speed_osrm
            print(f"speed_osrm (old) -> [min, max]: [{X_copy['speed_osrm'].min():.4f}, {X_copy['speed_osrm'].max():.4f}]")

            # Restrict speed_osrm
            X_copy = X_copy[(X_copy['speed_osrm'] <= 130) & (X_copy['speed_osrm'] >= 3)]
            print(f"speed_osrm (new) -> [min, max]: [{X_copy['speed_osrm'].min():.4f}, {X_copy['speed_osrm'].max():.4f}]")

            # Synchronize y_copy with X_copy
            y_copy = y_copy.loc[X_copy.index]
        else:
            print("speed_osrm column not found, skipping restriction on 'speed_osrm'.")

        # Print initial max and min passenger_count
        print(f"passenger_count (old) -> [min, max]: [{X_copy['passenger_count'].min()}, {X_copy['passenger_count'].max()}]")

        # Restrict passenger_count
        X_copy = X_copy[X_copy['passenger_count'] != 0]
        print(f"passenger_count (new) -> [min, max]: [{X_copy['passenger_count'].min()}, {X_copy['passenger_count'].max()}]")

        # Synchronize y_copy with X_copy
        y_copy = y_copy.loc[X_copy.index]

        # Calculate total removed data
        final_length = len(X_copy)
        total_removed = initial_length - final_length
        print(f"Total removed data: {total_removed} ({(total_removed / initial_length) * 100:.2f}%)")

        # Return the cleaned versions of X_copy and y_copy
        return X_copy, y_copy

class OutlierMapper(BaseEstimator, TransformerMixin):
    def __init__(self, lat_col="pickup", lon_col="dropoff", map_title="nyc_outliers_map", radius_km=50,
                 csv_dir="csv_ml", html_dir="html_ml", generate_map=True):
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.map_title = map_title
        self.radius_km = radius_km
        self.csv_dir = csv_dir
        self.html_dir = html_dir
        self.generate_map = generate_map
        self.csv_filename = f"{self.map_title.replace(' ', '_').lower()}.csv"
        self.map_center = [40.7128, -74.0060]  # NYC center

    def fit(self, X, y=None):
        return self

    def transform(self, X, y):
        print(">>>> Starting New York City map restriction ...")
        
        X_map = X.copy()
        y_map = y.copy()
        initial_count = len(X_map)

        # Ensure required columns are present
        if 'pickup_dist_NYC_center' not in X_map.columns or 'dropoff_dist_NYC_center' not in X_map.columns:
            raise ValueError("X must contain 'pickup_dist_NYC_center' and 'dropoff_dist_NYC_center' columns.")
 
        df_outliers = pd.concat([X_map, y_map], axis=1)
        df_outliers['is_outlier'] = (df_outliers['pickup_dist_NYC_center'] > self.radius_km) | (df_outliers['dropoff_dist_NYC_center'] > self.radius_km)
        outliers = df_outliers[df_outliers['is_outlier']]

        # Save outliers to CSV
        os.makedirs(self.csv_dir, exist_ok=True)
        csv_path = os.path.join(self.csv_dir, self.csv_filename)
        outliers.to_csv(csv_path, index=False)
        print(f"Outliers saved to '{csv_path}'")

        # Filter out outliers
        X_cleaned = df_outliers[~df_outliers['is_outlier']].drop(columns=['is_outlier', 'trip_duration'])
        y_map = y_map.loc[X_cleaned.index]

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

        os.makedirs(self.html_dir, exist_ok=True)
        file_name = f"{self.html_dir}/{self.map_title.replace(' ', '_').lower()}.html"
        nyc_map.save(file_name)
        print(f"Map saved as '{file_name}'")

class PickDropCluster(BaseEstimator, TransformerMixin):
    def __init__(self, optimal_k_pickup=4, optimal_k_dropoff=3):
        self.optimal_k_pickup = optimal_k_pickup
        self.optimal_k_dropoff = optimal_k_dropoff
        self.is_fitted = False
        self.kmeans_pickup = None
        self.kmeans_dropoff = None

    def fit(self, X, y=None):
        # Create copies of X
        X_copy = X.copy()

        # Extract pickup and dropoff coordinates
        pickup_coords = X_copy[['pickup_longitude', 'pickup_latitude']]
        dropoff_coords = X_copy[['dropoff_longitude', 'dropoff_latitude']]

        # Fit KMeans for pickup and dropoff clusters
        self.kmeans_pickup = KMeans(n_clusters=self.optimal_k_pickup, random_state=42, n_init=10)
        self.kmeans_pickup.fit(pickup_coords)

        self.kmeans_dropoff = KMeans(n_clusters=self.optimal_k_dropoff, random_state=42, n_init=10)
        self.kmeans_dropoff.fit(dropoff_coords)

        # Mark the transformer as fitted
        self.is_fitted = True
        
        return self

    def transform(self, X, y=None):
        print(">>>> Starting to create cluster features ...")
        # Check if the transformer has been fitted
        if not self.is_fitted:
            raise ValueError("The fit method must be called before transform.")

        # Create a copy of X
        X_copy = X.copy()

        # Extract pickup and dropoff coordinates
        pickup_coords = X_copy[['pickup_longitude', 'pickup_latitude']]
        dropoff_coords = X_copy[['dropoff_longitude', 'dropoff_latitude']]

        # Predict and assign cluster labels
        X_copy['pickup_cluster'] = self.kmeans_pickup.predict(pickup_coords).astype(str)

        X_copy['dropoff_cluster'] = self.kmeans_dropoff.predict(dropoff_coords).astype(str)

        # Drop original coordinate columns
        X_copy = X_copy.drop(['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'], axis=1)

        print("New features 'pickup_cluster' and 'dropoff_cluster' have been created.")
        print("Dropped the columns 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', and 'dropoff_latitude'")
        return X_copy, y

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)

class SkewnessTransform(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # No fitting process needed, but we need to check the skewness in the transform method.
        return self

    def transform(self, X, y):
        print(">>>> Starting to transform the skew features ...")
        # Create a copy of X and y to avoid modifying the original data
        X_copy = X.copy()
        y_copy = y.copy()

        # Check and print the skewness for each column in X_copy and y_copy before transformation
        skewness_x_before = X_copy.skew(numeric_only=True)
        skewness_y_before = y_copy.skew(numeric_only=True)
        
        print("\nSkewness of X columns before transformation:")
        print(skewness_x_before)
        print(f"Skewness of y before transformation: {skewness_y_before}")

        # Skewness threshold for transformation
        skew_threshold = (-2, 2)

        # Process only numeric columns in X_copy
        transformed_columns = []  # Keep track of columns that were transformed

        for column in X_copy.select_dtypes(include=[np.number]).columns:
            skew = skewness_x_before[column]
            if skew < skew_threshold[0] or skew > skew_threshold[1]:
                print(f"\nColumn '{column}' has skewness {skew} which is outside the range {skew_threshold}")
                
                # Get the minimum value of the column
                min_value = X_copy[column].min()
                print(f"Min value of column '{column}': {min_value}")

                # Custom formula: handling positive, negative, zero values
                X_copy[column] = np.log1p(np.maximum(X_copy[column] + np.abs(X_copy[column]), 0))
                print(f"Applied custom log transformation formula to '{column}'.")
                            
                # Rename the column by prefixing 'log_' and drop the original column
                X_copy.rename(columns={column: f'log_{column}'}, inplace=True)
                transformed_columns.append(column)  # Track the transformed column
        
        # Apply skewness transformation to y_copy if necessary
        skew_y = skewness_y_before
        if skew_y < skew_threshold[0] or skew_y > skew_threshold[1]:
            print(f"\nTarget variable 'y' has skewness {skew_y} which is outside the range {skew_threshold}")

            # Get the minimum value of y_copy
            min_value_y = y_copy.min()
            print(f"Min value of target variable 'y': {min_value_y}")

            # Custom formula: handling positive, negative, zero values
            y_copy = np.log1p(np.maximum(y_copy + np.abs(y_copy), 0))
            print(f"Applied custom log transformation formula to 'y'.")
            
            # Rename the target variable by prefixing 'log_' and drop the original 'y'
            y_copy.name = f'log_{y_copy.name}'
            transformed_columns.append('y')  # Track the transformed target variable

        # Check if transformation was done; if not, print a message
        if not transformed_columns:
            print("No skewness transformation was applied. All columns are within the acceptable skewness range (-2 <= skew <= 2).")

        # Check if transformation was done; if not, skip printing the skewness after transformation
        if transformed_columns:
            skewness_x_after = X_copy.skew(numeric_only=True)
            skewness_y_after = y_copy.skew(numeric_only=True)
        
            print("\nSkewness of X columns after transformation:")
            print(skewness_x_after)
            print(f"Skewness of y after transformation: {skewness_y_after}")

            # Check if skewness is still outside the range after transformation, and create an alert if it is
            alert_columns = []

            for column in X_copy.select_dtypes(include=[np.number]).columns:
                if skewness_x_after[column] < skew_threshold[0] or skewness_x_after[column] > skew_threshold[1]:
                    alert_columns.append(f"Column '{column}'")
            
            if skewness_y_after < skew_threshold[0] or skewness_y_after > skew_threshold[1]:
                alert_columns.append("Target variable 'y'")

            if alert_columns:
                print(f"ALERT: The following columns still have skewness outside the range {-2} <= skew <= 2 after transformation:")
                for alert in alert_columns:
                    print(alert)

        # Return the transformed X_copy and y_copy
        return X_copy, y_copy

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

class CorrStrength(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # This transformer is stateless, so no fitting is required
        return self

    def transform(self, X, y):
        print(">>>> Starting to check correlation strength ...")
        # Create copies of X and y
        X_copy = X.copy()
        y_copy = y.copy()

        # Combine X_copy and y_copy to analyze correlations
        df_concat = pd.concat([X_copy, y_copy.rename('trip_duration')], axis=1)

        # Rank correlations
        ranked_correlations = get_ranked_correlations(df_concat)
        print("Ranked correlations:\n", ranked_correlations)

        # Calculate the correlations between features
        correlations = X_copy.corr(numeric_only=True)

        # Identify strongly correlated features
        drop_columns = set()
        for col in correlations.columns:
            for row in correlations.index:
                if row != col and abs(correlations.loc[row, col]) >= 0.7:
                    # Identify which column to drop (keep leftmost column by default)
                    drop_columns.add(col)

        # Drop the identified columns and print the action
        if drop_columns:
            print(f"Dropping columns due to strong feature-feature correlations: {list(drop_columns)}")
            X_copy = X_copy.drop(columns=list(drop_columns))
        else:
            print("There is no strong feature-feature correlations")

        # Return the cleaned X_copy and y_copy
        return X_copy, y_copy

class VIFStrength(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=10):
        self.threshold = threshold

    def fit(self, X, y=None):
        return self  # No fitting is needed in this stateless transformer.

    def transform(self, X):
        print(">>>> Starting to check VIF strength ...")
        X_copy = X.copy()

        # Select only numeric columns for VIF calculation
        numeric_cols = X_copy.select_dtypes(include=[np.number]).columns

        # Calculate VIF for each numeric column
        vif_data = pd.DataFrame()
        vif_data['feature'] = numeric_cols
        vif_data['VIF'] = [variance_inflation_factor(X_copy[numeric_cols].values, i) for i in range(len(numeric_cols))]

        print(vif_data)
        
        # Find features with VIF greater than the threshold
        high_vif_features = vif_data[vif_data['VIF'] > self.threshold]

        # If there are features with VIF > threshold, drop the one with the highest VIF
        if not high_vif_features.empty:
            max_vif_feature = high_vif_features.loc[high_vif_features['VIF'].idxmax(), 'feature']
            print(f"Dropping column '{max_vif_feature}' with VIF: {vif_data.loc[vif_data['feature'] == max_vif_feature, 'VIF'].values[0]}")

            # Drop the column with the highest VIF
            X_copy = X_copy.drop(columns=[max_vif_feature])

            # Recalculate the numeric columns for VIF after dropping the high VIF column
            numeric_cols = X_copy.select_dtypes(include=[np.number]).columns

            # Calculate VIF again for remaining columns
            vif_data = pd.DataFrame()
            vif_data['feature'] = numeric_cols
            vif_data['VIF'] = [variance_inflation_factor(X_copy[numeric_cols].values, i) for i in range(len(numeric_cols))]

            # Print the new VIF values for the remaining columns
            print("\nRemaining columns after dropping the column with high VIF:")
            print(vif_data)

            # Check if any columns still have VIF > threshold and print a warning
            high_vif_features_after_drop = vif_data[vif_data['VIF'] > self.threshold]
            if not high_vif_features_after_drop.empty:
                print("\nAlert: After dropping the column, the following columns still have VIF > 10:")
                print(high_vif_features_after_drop)
        else:
            print("There are no features with VIF > 10.")

        # Return cleaned data with dropped columns and unchanged target (y)
        return X_copy

class FeatureEncoding(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Initialization (no parameters for now)
        pass

    def fit(self, X, y=None):
        # No fitting is needed for this stateless transformer
        return self

    def transform(self, X):
        print(">>>> Starting to encode the features ...")
        # Create a copy of X
        X_copy = X.copy()
        
        print("Starting transformations...")

        # Step 1: Drop 'id' column
        if 'id' in X_copy.columns:
            print("Dropping 'id' column...")
            X_copy = X_copy.drop(columns=['id'])

        # Step 2: Dummy encoding for categorical variables
        categorical_columns = ['vendor_id', 'store_and_fwd_flag', 'pickup_cluster', 'dropoff_cluster']
        for col in categorical_columns:
            if col in X_copy.columns:
                print(f"Performing dummy encoding on '{col}' column...")
                X_copy[col] = X_copy[col].astype(str)  # Cast to string to ensure consistent dtype
                X_copy = pd.get_dummies(X_copy, columns=[col], drop_first=True)
        
        # Step 3: Cyclical encoding for date/time-related variables
        cyclical_columns = ['pickup_month', 'pickup_day', 'pickup_day_of_week', 'pickup_hour']
        for col in cyclical_columns:
            if col in X_copy.columns:
                print(f"Performing cyclical encoding on '{col}' column...")
                X_copy[col] = X_copy[col].astype('int64')
                X_copy = self._cyclical_encode(X_copy, col)

        # Step 4: Drop original columns after encoding (only if they exist)
        print("Dropping original columns after encoding...")
        # Only drop the original columns if they still exist in X_copy
        columns_to_drop = categorical_columns + cyclical_columns
        columns_to_drop = [col for col in columns_to_drop if col in X_copy.columns]
        
        X_copy = X_copy.drop(columns=columns_to_drop)
        
        print("Transformation completed.")
        return X_copy
    
    def _cyclical_encode(self, X, col):
        # Convert values to radians (assuming that the values are in a typical range like 1-12 for months)
        max_value = X[col].max()
        X[f'{col}_sin'] = np.sin(2 * np.pi * X[col] / max_value)
        X[f'{col}_cos'] = np.cos(2 * np.pi * X[col] / max_value)
        print(f"Cyclical encoding for '{col}' completed.")
        return X

# Function to calculate RMSLE
def rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 1, None)  # Clip predictions to avoid log(0)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

# Function to evaluate models and optionally save them, and return results
def evaluate_models(models, X_train, y_train, X_test, y_test, save_models=False, save_path='./'):
    results = []
    
    for name, model in models:
        # Record start time for fitting
        start_time = time.time()
        
        # Fit the model
        model.fit(X_train, y_train)
        
        # Record end time for fitting
        fit_time = time.time() - start_time
        
        # Predict on the test set
        y_pred = model.predict(X_test)
        
        # Compute RMSLE
        rmsle_score = rmsle(y_test, y_pred)

        # Print result immediately after evaluating the model
        print(f'{name} - RMSLE: {rmsle_score:.4f}, Fit time: {fit_time:.4f} seconds')
        
        # Prepare result for this model in the desired format
        result = f'{name} - RMSLE: {rmsle_score:.4f}, Fit time: {fit_time:.4f} seconds'
        results.append(result)
        
        # Optionally save the model to a .pkl file
        if save_models:
            model_filename = f'{save_path}{name}_model.pkl'
            joblib.dump(model, model_filename)
            print(f'{name} model saved as {model_filename}')
    
    # Return the collected results as a single string (with each result on a new line)
    return '\n'.join(results)


        