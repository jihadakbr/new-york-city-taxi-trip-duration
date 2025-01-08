# NYC Taxi Trip Duration Prediction

This project aims to predict taxi trip durations in New York City using 2016 data from the NYC Taxi and Limousine Commission. The project showcases advanced data analysis, feature engineering, geospatial visualizations, and machine learning workflows, leveraging business-aligned metrics to achieve optimal predictions.

---

## Table of Contents

- [Dataset Overview](#dataset-overview)
- [Objectives](#objectives)
- [Approach](#approach)
- [Key Features and Insights](#key-features-and-insights)
- [Tools and Libraries](#tools-and-libraries)
- [Results](#results)
- [How to Use](#how-to-use)
- [Future Improvements](#future-improvements)
- [Contact](#contact)

---

## Dataset Overview

The dataset contains detailed trip records from 2016 provided by the [NYC Taxi and Limousine Commission](https://www.kaggle.com/competitions/nyc-taxi-trip-duration). It includes the following features:

| **Column**            | **Description**                                                                 |
|------------------------|---------------------------------------------------------------------------------|
| `id`                  | Unique identifier for each trip.                                               |
| `vendor_id`           | Code for the vendor providing the trip.                                        |
| `pickup_datetime`     | Timestamp for when the trip started.                                           |
| `dropoff_datetime`    | Timestamp for when the trip ended.                                             |
| `passenger_count`     | Number of passengers in the taxi.                                              |
| `pickup_longitude`    | Longitude of the pickup location.                                              |
| `pickup_latitude`     | Latitude of the pickup location.                                               |
| `dropoff_longitude`   | Longitude of the dropoff location.                                             |
| `dropoff_latitude`    | Latitude of the dropoff location.                                              |
| `store_and_fwd_flag`  | Whether the trip data was stored and forwarded due to connectivity issues.      |
| `trip_duration`       | Duration of the trip in seconds (target variable).                             |

Dataset size: **1,458,644 rows** with no missing values.

Details about the dataset can be found on the [Kaggle competition page](https://www.kaggle.com/competitions/nyc-taxi-trip-duration).

---

## Objectives

1. Predict NYC taxi trip durations with high accuracy to support strategic decision-making.
2. Explore and visualize geographic and temporal trip patterns.
3. Develop modular and scalable machine learning workflows.

---

## Approach

### 1. Data Preprocessing
- Cleaned and processed raw data to handle outliers and inconsistencies.
- Applied scaling and encoding to numeric and categorical features.

### 2. Feature Engineering
- Engineered features like trip distance (haversine and Manhattan), average speed, and pickup/dropoff clusters.
- Optimized geospatial features by restricting coordinates to valid NYC boundaries.

### 3. Exploratory Data Analysis (EDA)
- Conducted temporal and spatial EDA to uncover patterns in trip durations.
- Created heatmaps and geospatial visualizations for key pickup and dropoff points.

### 4. Clustering
- Applied K-Means clustering to identify natural pickup and dropoff clusters.
- Visualized clusters on NYC maps to reveal geographic trip patterns and inform decision-making.

### 5. Machine Learning Workflow
- Designed reusable transformer components using object-oriented programming (OOP), ensuring compatibility with scikit-learn pipelines.
- Evaluated various models, including LightGBM, using RMSLE as the performance metric.
- Tuned hyperparameters and optimized the final LightGBM model for best performance.

---

## Key Features and Insights

- **Feature Engineering**: Enhanced the dataset with meaningful features like distance, speed, and time-based patterns.
- **Geospatial Analysis**: 
  - Heatmaps of high-density pickup/dropoff points.
  - Cluster visualizations for geographic insights.
- **Modular Pipelines**: Reusable components for data preprocessing, ensuring efficient and scalable workflows.

---

## Tools and Libraries

- **Programming Language**: Python
- **Key Libraries**:
  - Data Processing: Pandas, NumPy
  - Geospatial Analysis: Geopy, Folium, Matplotlib, Seaborn
  - Machine Learning: Scikit-learn, LightGBM
  - Clustering: K-Means (from Scikit-learn)

---

## Results

- **Final Model**: LightGBM Regressor
- **Performance Metric**: RMSLE (Root Mean Squared Logarithmic Error)
- **Achieved Score**: 0.55 RMSLE
- **Business Alignment**: Prioritized relative error over absolute error to meet business objectives.

![dropoff cluster](https://raw.githubusercontent.com/jihadakbr/new-york-city-taxi-trip-duration/refs/heads/main/img/dropoff_cluster.png)

![dropoff heatmap marker](https://raw.githubusercontent.com/jihadakbr/new-york-city-taxi-trip-duration/refs/heads/main/img/dropoff_heatmap_marker.png)

![dropoff plot](https://raw.githubusercontent.com/jihadakbr/new-york-city-taxi-trip-duration/refs/heads/main/img/dropoff_plot.png)

![elbow dropoff](https://raw.githubusercontent.com/jihadakbr/new-york-city-taxi-trip-duration/refs/heads/main/img/elbow_dropoff.png)

![rsmle score](https://raw.githubusercontent.com/jihadakbr/new-york-city-taxi-trip-duration/refs/heads/main/img/rsmle_score.png)

---

## How to Use

1. Clone the repository:
```
git clone https://github.com/your-username/nyc-taxi-trip-duration.git
cd nyc-taxi-trip-duration
```
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Run the notebook or scripts for preprocessing, analysis, and modeling:
```
jupyter notebook
```
4. Follow the instructions in the notebooks to replicate the workflow.
