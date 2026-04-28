import pandas as pd
import numpy as np
import time
from sklearn.metrics import root_mean_squared_error
import os

# Suppress TensorFlow INFO/WARNING startup logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# We don't have a GPU, so this supresses warnings related to not finding a GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Disable oneDNN optimizations for more reproducible CPU results
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense

# =====================================================================
# Parameters and Variables:
# dataset_file_path (string): The name and location of the CSV file for Sensor 1.
# dataset_file_path_2 (string): The name and location of the CSV file for Sensor 2.
# air_quality_data (DataFrame): The Pandas table that holds our main dataset.
# sensor2_data (DataFrame): The Pandas table that holds the second sensor dataset.
# clean_control_data (DataFrame): The data with zero missing values.
# =====================================================================

# Load the main sensor (Aotizhongxin)
dataset_file_path = "data/air-quality-master-tz-stripped.csv" #REPLACE WITH PATH TO AIR QUALITY CSV
clean_control_data = pd.read_csv(dataset_file_path) #clean data to compare with the ground truth
# air_quality_data["time"]=pd.to_datetime(air_quality_data["time"])

print("Original total rows:", len(clean_control_data))


def temporal_interpolate_by_zip(df, feature, method, **kwargs):
    """#Spline interpolation itself cannot interpolate edge cases (ex. temporally cannot fill missing values which are the first or last timestep)
    to combat this we use nearest neighbors interpolation to fill edge cases by casting interpolate again, for global interpolation functions and
    data without these edge cases, this does nothing. We decided to use this instead of restricting the dropped values to exclude the first and last 
    features since that seems to unfarily benefit spline interpolation where this limitation does exist."""
    return (
        df
        .sort_values(["zip", "time"])
        .groupby("zip", group_keys=False)[feature]
        .transform(
            lambda s: (
                s.interpolate(method=method, **kwargs)
                 .interpolate(method="nearest") #Spline interpolation helper
                 .ffill()
                 .bfill()
            )
        )
    )


def average_impute_by_zip(df, feature):
    """
    Fill missing values using the mean of that feature within each ZIP code.
    """
    return (
        df
        .groupby("zip")[feature]
        .transform(lambda s: s.fillna(s.mean()))
    )


# NOW WE FILL MISSING VALUES USING NEAREST NEIGHBOR SO THIS IS UNNEEDED, REVERTED TO SIMPLY CALLING root_mean_squared_error
# def safe_rmse(real_values, guessed_values):
#     """
#     Computes RMSE only on positions where interpolation produced a value.

#     This avoids crashing when spline/polynomial interpolation leaves edge NaNs.
#     """
#     valid_mask = real_values.notna() & guessed_values.notna()

#     if valid_mask.sum() == 0:
#         return np.nan

#     return root_mean_squared_error(
#         real_values[valid_mask],
#         guessed_values[valid_mask]
#     )


def run_all_methods(test_data, real_values, hole_indexes, test_name):
    print("\n" + "=" * 50)
    print("RUNNING INTERPOLATION FOR:", test_name)
    print("=" * 50)

    # Make sure each ZIP code is temporally ordered before interpolation
    test_data = test_data.sort_values(["zip", "time"]).copy()
    data_filled_newton_all_features = test_data.copy()
    results = {}

    for feature in interpolation_features:
        print(f"\n--- Feature: {feature} ---")

        results[feature] = {}

        # ==============================================================
        # Method 1: Average Imputation by ZIP
        # ==============================================================
        start_time = time.time()

        data_filled_with_average = test_data.copy()
        data_filled_with_average[feature] = average_impute_by_zip(
            data_filled_with_average,
            feature
        )

        runtime_seconds = time.time() - start_time

        average_guessed_values = data_filled_with_average.loc[hole_indexes, feature]
        rmse_average = root_mean_squared_error(real_values[feature], average_guessed_values)

        results[feature]["average_by_zip"] = rmse_average

        print(
            f"Average Imputation by ZIP Runtime: "
            f"{runtime_seconds:.4f}s | RMSE: {rmse_average:.4f}"
        )

        # ==============================================================
        # Method 2: 2nd Degree Spline Interpolation by ZIP
        # ==============================================================
        start_time = time.time()

        data_filled_with_spline_2 = test_data.copy()
        data_filled_with_spline_2[feature] = temporal_interpolate_by_zip(
            data_filled_with_spline_2,
            feature,
            method="spline",
            order=2
        )

        runtime_seconds = time.time() - start_time

        spline_guessed_values = data_filled_with_spline_2.loc[hole_indexes, feature]
        rmse_spline = root_mean_squared_error(real_values[feature], spline_guessed_values)

        results[feature]["spline_2_by_zip"] = rmse_spline

        print(
            f"2nd Degree Spline by ZIP Runtime: "
            f"{runtime_seconds:.4f}s | RMSE: {rmse_spline:.4f}"
        )

        # ==============================================================
        # Method 3: Nearest Neighbors Interpolation by ZIP
        # ==============================================================
        start_time = time.time()

        data_filled_nearest = test_data.copy()
        data_filled_nearest[feature] = temporal_interpolate_by_zip(
            data_filled_nearest,
            feature,
            method="nearest"
        )

        runtime_seconds = time.time() - start_time

        nearest_guessed_values = data_filled_nearest.loc[hole_indexes, feature]
        rmse_nearest = root_mean_squared_error(real_values[feature], nearest_guessed_values)

        results[feature]["nearest_by_zip"] = rmse_nearest

        print(
            f"Nearest Neighbors by ZIP Runtime: "
            f"{runtime_seconds:.4f}s | RMSE: {rmse_nearest:.4f}"
        )

        # ==============================================================
        # Methods 4, 5, 6: Spline Interpolation by ZIP, Degrees 3, 4, 5
        # ==============================================================
        degrees_to_test = [3, 4, 5]

        for current_degree in degrees_to_test:
            start_time = time.time()

            data_filled_spline = test_data.copy()
            data_filled_spline[feature] = temporal_interpolate_by_zip(
                data_filled_spline,
                feature,
                method="spline",
                order=current_degree
            )

            runtime_seconds = time.time() - start_time

            spline_guessed_values = data_filled_spline.loc[hole_indexes, feature]
            rmse_spline_deg = root_mean_squared_error(real_values[feature], spline_guessed_values)

            results[feature][f"spline_{current_degree}_by_zip"] = rmse_spline_deg

            print(
                f"{current_degree} Degree Spline by ZIP Runtime: "
                f"{runtime_seconds:.4f}s | RMSE: {rmse_spline_deg:.4f}"
            )

        # ==============================================================
        # Method 7: Newton / Polynomial Interpolation by ZIP
        # ==============================================================
        start_time = time.time()

        data_filled_newton = test_data.copy()
        data_filled_newton[feature] = temporal_interpolate_by_zip(
            data_filled_newton,
            feature,
            method="polynomial",
            order=3
        )

        # Also save this feature into the all-feature Newton dataframe
        data_filled_newton_all_features[feature] = data_filled_newton[feature]

        runtime_seconds = time.time() - start_time

        newton_guessed_values = data_filled_newton.loc[hole_indexes, feature]
        rmse_newton = root_mean_squared_error(real_values[feature], newton_guessed_values)

        results[feature]["newton_polynomial_3_by_zip"] = rmse_newton

        print(
            f"Newton / Polynomial by ZIP Runtime: "
            f"{runtime_seconds:.4f}s | RMSE: {rmse_newton:.4f}"
        )

    return results, data_filled_newton_all_features


# =====================================================================
# The purpose of this loop is to test 10%, 20%, 30%, and 40% missing data
# =====================================================================
percentages_to_test = [0.10]
total_clean_rows = len(clean_control_data)
# column_position = clean_control_data.columns.get_loc('us_aqi')
interpolation_features = ["us_aqi","pm10","pm2_5","carbon_monoxide",
            "nitrogen_dioxide","sulphur_dioxide","ozone","uv_index_clear_sky","uv_index","dust","aerosol_optical_depth"] #Features included in dataset
num_features = len(interpolation_features)
exempt_columns = []

saved_newton_data_for_nn = None 

for percent in percentages_to_test:
    print(f"\n\n********** STARTING TEST FOR {int(percent * 100)}% MISSING DATA **********")
    
    # -----------------------------------------------------------------
    # Create Random Holes
    # -----------------------------------------------------------------
    test_data_random_holes = clean_control_data.copy()
    unique_timesteps = clean_control_data["time"].drop_duplicates().to_numpy()
    number_of_random_timesteps=int(len(unique_timesteps) * percent)
    random_timesteps = np.random.choice(unique_timesteps, number_of_random_timesteps, replace=False)
    random_row_indexes=clean_control_data.index[clean_control_data["time"].isin(random_timesteps)].to_numpy()
    test_data_random_holes.loc[random_row_indexes, interpolation_features] = np.nan
    real_values_random = clean_control_data.loc[random_row_indexes, interpolation_features]    
    # -----------------------------------------------------------------
    # Create 24-Hour Block Holes
    # -----------------------------------------------------------------
    test_data_block_holes = clean_control_data.copy()
    block_size = 24
    unique_timesteps_sorted = np.sort(unique_timesteps)
    number_of_blocks = int((len(unique_timesteps) * percent) / block_size)

    block_timesteps = []
    for _ in range(number_of_blocks):
        start_index = np.random.randint(0, len(unique_timesteps_sorted) - block_size)
        for i in range(block_size):
            block_timesteps.extend(unique_timesteps_sorted[start_index:start_index+block_size])

    block_timesteps = list(set(block_timesteps))
    block_row_indexes = clean_control_data.index[
        clean_control_data["time"].isin(block_timesteps)
    ].to_numpy()

    test_data_block_holes.loc[block_row_indexes, interpolation_features] = np.nan

    real_values_blocks = clean_control_data.loc[block_row_indexes, interpolation_features]

    # -----------------------------------------------------------------
    # Run the Methods (Now passing Sensor 2 data as well)
    # -----------------------------------------------------------------
    newton_random_results, newton_random_filled_df = run_all_methods(test_data_random_holes, real_values_random, random_row_indexes, f"{int(percent * 100)}% RANDOM HOLES")
    newton_block_results, newton_block_filled_df = run_all_methods(test_data_block_holes, real_values_blocks, block_row_indexes, f"{int(percent * 100)}% 24-HOUR BLOCK HOLES")
    
    if percent == 0.10:
        saved_newton_data_for_nn = newton_random_filled_df

# =====================================================================
# NEURAL NETWORK SECTION (WITH WEATHER DATA)
# =====================================================================

# =====================================================================
# NEURAL NETWORK SECTION
# Works with multi-ZIP time-series dataset
# =====================================================================

target_col = "us_aqi"
nn_features=["us_aqi","pm10","pm2_5","carbon_monoxide",
            "nitrogen_dioxide","sulphur_dioxide","ozone","uv_index_clear_sky","uv_index","dust","aerosol_optical_depth","latitude","longitude"]
window_size = 24
num_features = len(nn_features)

# Make sure the data is sorted correctly
clean_control_data = clean_control_data.sort_values(["zip", "time"]).reset_index(drop=True)

# Optional but recommended: make sure time is datetime
clean_control_data["time"] = pd.to_datetime(clean_control_data["time"])


def create_multivariate_time_windows_by_zip(df, features, target_col, window_size):
    """
    Creates Conv1D/LSTM-ready time windows separately for each ZIP code.

    X shape:
        (num_windows, window_size, num_features)

    y shape:
        (num_windows,)

    Each y value is the target_col value immediately after the window.
    """

    X_windows = []
    y_answers = []

    target_index = features.index(target_col)

    for zip_code, zip_group in df.groupby("zip"):
        zip_group = zip_group.sort_values("time")

        data_array = zip_group[features].values

        if len(data_array) <= window_size:
            continue

        for current_step in range(len(data_array) - window_size):
            X_windows.append(data_array[current_step : current_step + window_size])
            y_answers.append(data_array[current_step + window_size][target_index])

    return np.array(X_windows), np.array(y_answers)


def train_test_split_time_windows_by_zip(df, features, target_col, window_size, train_fraction=0.8):
    """
    Splits each ZIP code's time series into train/test first,
    then creates windows separately.

    This avoids creating training windows that contain future test data.
    """

    X_train_all = []
    y_train_all = []
    X_test_all = []
    y_test_all = []

    target_index = features.index(target_col)

    for zip_code, zip_group in df.groupby("zip"):
        zip_group = zip_group.sort_values("time")

        data_array = zip_group[features].values

        if len(data_array) <= window_size + 1:
            continue

        split_index = int(len(data_array) * train_fraction)

        train_array = data_array[:split_index]
        test_array = data_array[split_index:]

        # Need enough rows to create at least one window
        if len(train_array) > window_size:
            for current_step in range(len(train_array) - window_size):
                X_train_all.append(train_array[current_step : current_step + window_size])
                y_train_all.append(train_array[current_step + window_size][target_index])

        if len(test_array) > window_size:
            for current_step in range(len(test_array) - window_size):
                X_test_all.append(test_array[current_step : current_step + window_size])
                y_test_all.append(test_array[current_step + window_size][target_index])

    return (
        np.array(X_train_all),
        np.array(y_train_all),
        np.array(X_test_all),
        np.array(y_test_all)
    )


def build_convlstm_model(window_size, num_features):
    model = Sequential()
    model.add(tf.keras.Input(shape=(window_size, num_features)))
    model.add(Conv1D(filters=64, kernel_size=2, activation="relu"))
    model.add(LSTM(50, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")

    return model

# =====================================================================
# Train and Test Baseline Model
# =====================================================================

X_train, y_train, X_test, y_test = train_test_split_time_windows_by_zip(
    clean_control_data,
    nn_features,
    target_col,
    window_size,
    train_fraction=0.8
)

baseline_model = build_convlstm_model(window_size, num_features)

print("\nData is prepared and the Convolutional LSTM model is built successfully!")

print("\n--- Training Baseline Model (No Missing Values) ---")
num_training_rounds = 3

baseline_model.fit(X_train, y_train, epochs=num_training_rounds, verbose=0)

baseline_predictions = baseline_model.predict(X_test, verbose=0).flatten()

baseline_prediction_rmse = root_mean_squared_error(y_test, baseline_predictions)

print("Baseline Model Prediction RMSE:", baseline_prediction_rmse)

# =====================================================================
# Train and Test Newton Model
# =====================================================================

print("\n--- Training Model with Newton Interpolation Data (10% Random Holes) ---")

saved_newton_data_for_nn = saved_newton_data_for_nn.sort_values(["zip", "time"]).reset_index(drop=True)
saved_newton_data_for_nn["time"] = pd.to_datetime(saved_newton_data_for_nn["time"])

X_train_newton, y_train_newton, X_test_newton, y_test_newton = train_test_split_time_windows_by_zip(
    saved_newton_data_for_nn,
    nn_features,
    target_col,
    window_size,
    train_fraction=0.8
)

newton_model = build_convlstm_model(window_size, num_features)

newton_model.fit(X_train_newton, y_train_newton, epochs=num_training_rounds, verbose=0)

newton_predictions = newton_model.predict(X_test_newton, verbose=0).flatten()

newton_prediction_rmse = root_mean_squared_error(y_test_newton, newton_predictions)

print("Newton Model Prediction RMSE:", newton_prediction_rmse)