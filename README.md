# NYC Taxi Trip Duration Prediction

This project predicts taxi trip durations in New York City using 2016 data from the NYC Taxi and Limousine Commission. The workflow is divided into multiple Jupyter Notebooks, each focusing on a specific phase of the analysis and modeling process. The project demonstrates advanced data analysis, feature engineering, geospatial visualizations, and machine learning techniques.

---

## Table of Contents

- [Dataset Overview](#dataset-overview)
- [Objectives](#objectives)
- [Project Workflow](#project-workflow)
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

More details can be found on the [Kaggle competition page](https://www.kaggle.com/competitions/nyc-taxi-trip-duration).

---

## Objectives

1. Explore and analyze data to uncover trends and patterns.
2. Build a baseline machine learning model for trip duration prediction.
3. Optimize the model for improved performance using advanced techniques.
4. Develop reusable and modular machine learning workflows.

---

## Project Workflow

The project is divided into four main notebooks, each focusing on a specific phase:

1. **[New York City Taxi Trip Duration - EDA.ipynb](https://github.com/jihadakbr/new-york-city-taxi-trip-duration/blob/main/New%20York%20City%20Taxi%20Trip%20Duration%20-%20EDA.ipynb)**  
   - Objective: Perform exploratory data analysis (EDA).
   - Includes geospatial and temporal visualizations, feature correlations, and insights into trip patterns.

2. **[New York City Taxi Trip Duration - ML - 1 - Baseline.ipynb](https://github.com/jihadakbr/new-york-city-taxi-trip-duration/blob/main/New%20York%20City%20Taxi%20Trip%20Duration%20-%20ML%20-%201%20-%20Baseline.ipynb)**  
   - Objective: Build baseline machine learning models.
   - Compares simple regression models to establish a performance benchmark.

3. **[New York City Taxi Trip Duration - ML - 2 - Model Selection.ipynb](https://github.com/jihadakbr/new-york-city-taxi-trip-duration/blob/main/New%20York%20City%20Taxi%20Trip%20Duration%20-%20ML%20-%202%20-%20Model%20Selection.ipynb)**  
   - Objective: Select the best-performing models.
   - Compares advanced models such as Random Forest, Gradient Boosting, LightGBM, etc.

4. **[New York City Taxi Trip Duration - ML - 3 - Model Tuning.ipynb](https://github.com/jihadakbr/new-york-city-taxi-trip-duration/blob/main/New%20York%20City%20Taxi%20Trip%20Duration%20-%20ML%20-%203%20-%20Model%20Tuning.ipynb)**  
   - Objective: Tune hyperparameters of the selected model.
   - Uses techniques like grid search for optimization.

---

## Key Features and Insights

- **Feature Engineering**: Enhanced the dataset with meaningful features like distance, speed, and time-based patterns.
- **Geospatial Analysis**: Heatmaps of key pickup/dropoff locations and clustering of trip patterns.
- **Scalable Workflows**: Implemented modular pipelines using scikit-learn and reusable transformer components.

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


# Future Improvements

1. Integrate external data sources like real-time weather and traffic information.
2. Experiment with deep learning models (e.g., LSTMs or Transformers) to capture temporal dependencies.
3. Build an interactive dashboard to visualize predictions and geospatial patterns.

# Contact
For questions or collaborations, feel free to reach out:

- Email: [jihadakbr@gmail.com](mailto:jihadakbr@gmail.com)
- LinkedIn: [linkedin.com/in/jihadakbr](https://www.linkedin.com/in/jihadakbr)
- Portfolio: [jihadakbr.github.io](https://jihadakbr.github.io/)
