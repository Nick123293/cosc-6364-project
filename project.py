import os
# Suppress TensorFlow INFO/WARNING startup logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# Disable oneDNN optimizations for more reproducible CPU results
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import time
import random
import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error
import json #for logging

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense

#for reproducibility
GLOBAL_SEED = 42

random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
tf.keras.utils.set_random_seed(GLOBAL_SEED)
tf.config.experimental.enable_op_determinism()

rng = np.random.default_rng(GLOBAL_SEED)

#global variables
dataset_file_path = "data/air-quality-master-tz-stripped.csv" #CHANGE TO YOUR DATA FILEPATH

log_file_path = "experiment_log.csv"

percentages_to_test = [0.10, 0.20] #percentages of removal for interpolation

block_size = 8 #Block size for removing blocks for interpolation

target_col = "us_aqi"

interpolation_features = [
    "us_aqi",
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "uv_index_clear_sky",
    "uv_index",
    "dust",
    "aerosol_optical_depth",
]

nn_features = interpolation_features + ["latitude", "longitude"]# latitude and longitude are kept because they can help the NN learn location-specific behavior.

PREDICTION_TARGETS = ["us_aqi"] #Currently just one prediction target, this is here so we can change to multiple if we'd like

window_size = 24
train_fraction = 0.8
num_training_rounds = 3


clean_control_data = pd.read_csv(dataset_file_path) #our dataset requires no cleaning, so we can just read the csv as clean

log_file_path = "experiment_log.json" #Log file path, change to whatever you'd like

experiment_log = {
    "baseline": {
        "prediction_rmse": {}
    },
    "interpolation_methods": {}
}


def percent_key(missing_percent):
    """
    Converts 0.1 to '0.10' so JSON keys are consistent.
    """
    return f"{missing_percent:.2f}"


def ensure_method_pattern_percent(method, missing_pattern, missing_percent):
    """
    Ensures the nested JSON structure exists:

    interpolation_methods
        -> method
            -> missing_pattern
                -> missing_percent
    """
    p_key = percent_key(missing_percent)

    experiment_log["interpolation_methods"].setdefault(method, {})
    experiment_log["interpolation_methods"][method].setdefault(missing_pattern, {})
    experiment_log["interpolation_methods"][method][missing_pattern].setdefault(p_key, {})

    return experiment_log["interpolation_methods"][method][missing_pattern][p_key]


def log_baseline_prediction(feature, prediction_rmse, runtime_seconds=None):
    experiment_log["baseline"]["prediction_rmse"][feature] = {
        "value": float(prediction_rmse),
        "runtime_seconds": None if runtime_seconds is None else float(runtime_seconds)
    }


def log_interpolation_rmse(
    method,
    missing_pattern,
    missing_percent,
    feature,
    interpolation_rmse,
    runtime_seconds=None,
    num_missing_timesteps=None,
    num_missing_rows=None,
):
    percent_block = ensure_method_pattern_percent(
        method,
        missing_pattern,
        missing_percent
    )

    percent_block.setdefault("_metadata", {})

    if num_missing_timesteps is not None:
        percent_block["_metadata"]["num_missing_timesteps"] = int(num_missing_timesteps)

    if num_missing_rows is not None:
        percent_block["_metadata"]["num_missing_rows"] = int(num_missing_rows)

    percent_block.setdefault(feature, {})

    percent_block[feature]["interpolation_rmse"] = {
        "value": float(interpolation_rmse),
        "runtime_seconds": None if runtime_seconds is None else float(runtime_seconds)
    }


def log_prediction_rmse(
    method,
    missing_pattern,
    missing_percent,
    feature,
    prediction_rmse,
    runtime_seconds=None,
    num_missing_timesteps=None,
    num_missing_rows=None,
):
    percent_block = ensure_method_pattern_percent(
        method,
        missing_pattern,
        missing_percent
    )

    percent_block.setdefault("_metadata", {})

    if num_missing_timesteps is not None:
        percent_block["_metadata"]["num_missing_timesteps"] = int(num_missing_timesteps)

    if num_missing_rows is not None:
        percent_block["_metadata"]["num_missing_rows"] = int(num_missing_rows)

    percent_block.setdefault(feature, {})

    percent_block[feature]["prediction_rmse"] = {
        "value": float(prediction_rmse),
        "runtime_seconds": None if runtime_seconds is None else float(runtime_seconds)
    }


def save_log():
    with open(log_file_path, "w") as f:
        json.dump(experiment_log, f, indent=4)

    print(f"\nSaved experiment log to: {log_file_path}")


def finite_rmse(real_values, guessed_values):
    """
    Computes RMSE while ignoring non-finite values.
    This protects logging from NaN/inf predictions during unstable interpolation.
    """

    real_values = np.asarray(real_values, dtype=float).flatten()
    guessed_values = np.asarray(guessed_values, dtype=float).flatten()

    valid_mask = np.isfinite(real_values) & np.isfinite(guessed_values)

    if valid_mask.sum() == 0:
        return np.nan

    return root_mean_squared_error(
        real_values[valid_mask],
        guessed_values[valid_mask],
    )

def interpolate_one_zip_series(series, method, fill_edges=True, **kwargs):
    """
    Interpolate one ZIP code's time series.

    We reset the index to 0,1,2,... before interpolation so spline/polynomial
    methods use local temporal positions instead of the original dataframe index.
    """
    original_index = series.index

    local_series = pd.Series(
        series.to_numpy(dtype=float),
        index=np.arange(len(series), dtype=float),
    )

    filled = local_series.interpolate(method=method, **kwargs)

    if fill_edges:
        #To make sure interpolation can fill edge cases
        filled = filled.interpolate(method="nearest").ffill().bfill()

    filled.index = original_index

    return filled


def temporal_interpolate_by_zip(df, feature, method, fill_edges=True, **kwargs):
    #Interpolation groud by zip code (for temporal interpolation)
    df_sorted = df.sort_values(["zip", "time"]).copy()

    filled = (
        df_sorted
        .groupby("zip", group_keys=False)[feature]
        .apply(
            lambda s: interpolate_one_zip_series(
                s,
                method=method,
                fill_edges=fill_edges,
                **kwargs,
            )
        )
    )

    return filled.reindex(df.index)


def average_impute_by_zip(df, feature):
    #Imputation grouped by zip code (for temporal imputation)
    return (
        df
        .groupby("zip")[feature]
        .transform(lambda s: s.fillna(s.mean()))
    )


def newton_divided_differences(x, y):
    x = np.asarray(x, dtype=float)
    coef = np.asarray(y, dtype=float).copy()

    n = len(coef)

    for j in range(1, n):
        denominator = x[j:n] - x[0:n - j]
        coef[j:n] = (coef[j:n] - coef[j - 1:n - 1]) / denominator

    return coef


def newton_evaluate(x_known, coef, x_eval):
    x_known = np.asarray(x_known, dtype=float)
    x_eval = np.asarray(x_eval, dtype=float)

    result = np.full_like(x_eval, coef[-1], dtype=float)

    for k in range(len(coef) - 2, -1, -1):
        result = result * (x_eval - x_known[k]) + coef[k]

    return result


def local_newton_interpolate_by_zip(df, feature, degree=3):
    df_sorted = df.sort_values(["zip", "time"]).copy()
    filled_series = df_sorted[feature].copy()

    num_points_needed = degree + 1

    for zip_code, zip_group in df_sorted.groupby("zip"):
        values = zip_group[feature].astype(float)

        known_mask = values.notna()
        missing_mask = values.isna()

        if missing_mask.sum() == 0:
            continue

        if known_mask.sum() < num_points_needed:
            fallback = values.interpolate(method="nearest").ffill().bfill()
            filled_series.loc[zip_group.index] = fallback
            continue

        x_all = np.arange(len(zip_group), dtype=float)

        known_positions = x_all[known_mask.to_numpy()]
        known_values = values.loc[known_mask].to_numpy(dtype=float)

        missing_positions = x_all[missing_mask.to_numpy()]
        missing_indexes = zip_group.index[missing_mask]

        for x_missing, row_index in zip(missing_positions, missing_indexes):
            nearest_indices = np.argsort(
                np.abs(known_positions - x_missing)
            )[:num_points_needed]

            x_known_local = known_positions[nearest_indices]
            y_known_local = known_values[nearest_indices]

            order = np.argsort(x_known_local)
            x_known_local = x_known_local[order]
            y_known_local = y_known_local[order]

            coef = newton_divided_differences(x_known_local, y_known_local)

            y_missing = newton_evaluate(
                x_known_local,
                coef,
                np.array([x_missing]),
            )[0]

            filled_series.loc[row_index] = y_missing

    # Final fallback in case anything remains missing.
    filled_series = (
        filled_series
        .groupby(df_sorted["zip"], group_keys=False)
        .apply(lambda s: s.interpolate(method="nearest").ffill().bfill())
    )

    return filled_series.reindex(df.index)


# =====================================================================
# Interpolation Method Registry
# =====================================================================

INTERPOLATION_METHODS = [
    {
        "name": "average_by_zip",
        "type": "average",
    },
    {
        "name": "nearest_by_zip",
        "type": "pandas_interpolate",
        "method": "nearest",
        "kwargs": {},
    },
    {
        "name": "spline_2_by_zip",
        "type": "pandas_interpolate",
        "method": "spline",
        "kwargs": {"order": 2},
    },
    # {
    #     "name": "spline_3_by_zip",
    #     "type": "pandas_interpolate",
    #     "method": "spline",
    #     "kwargs": {"order": 3},
    # },
    # {
    #     "name": "spline_4_by_zip",
    #     "type": "pandas_interpolate",
    #     "method": "spline",
    #     "kwargs": {"order": 4},
    # },
    # {
    #     "name": "spline_5_by_zip",
    #     "type": "pandas_interpolate",
    #     "method": "spline",
    #     "kwargs": {"order": 5},
    # },
    {
        "name": "local_newton_degree_3_by_zip",
        "type": "local_newton",
        "degree": 3,
    },
]


def fill_feature_with_method(df, feature, method_spec):
    """
    Fill one feature using one interpolation method.
    """

    method_type = method_spec["type"]

    if method_type == "average":
        return average_impute_by_zip(df, feature)

    if method_type == "pandas_interpolate":
        return temporal_interpolate_by_zip(
            df,
            feature,
            method=method_spec["method"],
            fill_edges=True,
            **method_spec.get("kwargs", {}),
        )

    if method_type == "local_newton":
        return local_newton_interpolate_by_zip(
            df,
            feature,
            degree=method_spec["degree"],
        )

    raise ValueError(f"Unknown interpolation method type: {method_type}")


def run_all_interpolation_methods(
    test_data,
    real_values,
    hole_indexes,
    missing_pattern,
    missing_percent,
    num_missing_timesteps,
):
    """
    Runs all interpolation methods.

    Returns:
        filled_datasets:
            dict mapping method_name -> fully interpolated dataframe
    """

    print("\n" + "=" * 70)
    print(
        f"RUNNING INTERPOLATION: "
        f"pattern={missing_pattern}, missing={int(missing_percent * 100)}%"
    )
    print("=" * 70)

    # Do not reset index here; hole_indexes are original index labels.
    test_data = test_data.sort_values(["zip", "time"]).copy()

    filled_datasets = {
        method_spec["name"]: test_data.copy()
        for method_spec in INTERPOLATION_METHODS
    }

    num_missing_rows = len(hole_indexes)

    for feature in interpolation_features:
        print(f"\n--- Interpolating Feature: {feature} ---")

        for method_spec in INTERPOLATION_METHODS:
            method_name = method_spec["name"]

            start_time = time.time()

            filled_feature = fill_feature_with_method(
                test_data,
                feature,
                method_spec,
            )

            runtime_seconds = time.time() - start_time

            filled_datasets[method_name][feature] = filled_feature

            guessed_values = filled_datasets[method_name].loc[hole_indexes, feature]

            rmse_value = finite_rmse(
                real_values[feature],
                guessed_values,
            )

            log_interpolation_rmse(
                method=method_name,
                missing_pattern=missing_pattern,
                missing_percent=missing_percent,
                feature=feature,
                interpolation_rmse=rmse_value,
                runtime_seconds=runtime_seconds,
                num_missing_timesteps=num_missing_timesteps,
                num_missing_rows=num_missing_rows,
            )

            print(
                f"{method_name:30s} | "
                f"Runtime: {runtime_seconds:.4f}s | "
                f"RMSE: {rmse_value:.4f}"
            )

    return filled_datasets


# =====================================================================
# Missing Data Creation
# =====================================================================

def create_random_timestep_holes(clean_df, percent):
    """
    Randomly select a percent of global timesteps and remove all interpolation
    features for every ZIP code at those timesteps.
    """

    test_data = clean_df.copy()

    unique_timesteps = clean_df["time"].drop_duplicates().to_numpy()

    number_of_random_timesteps = int(len(unique_timesteps) * percent)

    random_timesteps = rng.choice(
        unique_timesteps,
        number_of_random_timesteps,
        replace=False,
    )

    row_indexes = clean_df.index[
        clean_df["time"].isin(random_timesteps)
    ].to_numpy()

    test_data.loc[row_indexes, interpolation_features] = np.nan

    real_values = clean_df.loc[row_indexes, interpolation_features]

    return test_data, real_values, row_indexes, len(random_timesteps)


def create_block_timestep_holes(clean_df, percent, block_size):
    """
    Select random contiguous timestep blocks and remove all interpolation
    features for every ZIP code at those timesteps.

    Note:
        Blocks may overlap, so actual missing percent can be slightly lower
        than requested.
    """

    test_data = clean_df.copy()

    unique_timesteps = clean_df["time"].drop_duplicates().to_numpy()
    unique_timesteps_sorted = np.sort(unique_timesteps)

    number_of_blocks = int((len(unique_timesteps_sorted) * percent) / block_size)

    block_timesteps = []

    for _ in range(number_of_blocks):
        start_index = rng.integers(
            0,
            len(unique_timesteps_sorted) - block_size,
        )

        block_timesteps.extend(
            unique_timesteps_sorted[start_index:start_index + block_size]
        )

    block_timesteps = np.array(sorted(set(block_timesteps)))

    row_indexes = clean_df.index[
        clean_df["time"].isin(block_timesteps)
    ].to_numpy()

    test_data.loc[row_indexes, interpolation_features] = np.nan

    real_values = clean_df.loc[row_indexes, interpolation_features]

    actual_percent = len(block_timesteps) / len(unique_timesteps_sorted)

    print(
        f"Requested block missing percent: {percent:.2f}, "
        f"actual timestep missing percent: {actual_percent:.4f}"
    )

    return test_data, real_values, row_indexes, len(block_timesteps)


# =====================================================================
# Neural Network Helpers
# =====================================================================

def train_test_split_time_windows_by_zip(
    df,
    features,
    target_col,
    window_size,
    train_fraction=0.8,
):
    """
    Splits each ZIP code's time series into train/test first,
    then creates windows separately.

    X shape:
        (num_windows, window_size, num_features)

    y shape:
        (num_windows,)
    """

    X_train_all = []
    y_train_all = []
    X_test_all = []
    y_test_all = []

    target_index = features.index(target_col)

    df = df.sort_values(["zip", "time"]).copy()

    for zip_code, zip_group in df.groupby("zip"):
        zip_group = zip_group.sort_values("time")

        data_array = zip_group[features].to_numpy(dtype=float)

        if len(data_array) <= window_size + 1:
            continue

        split_index = int(len(data_array) * train_fraction)

        train_array = data_array[:split_index]
        test_array = data_array[split_index:]

        if len(train_array) > window_size:
            for current_step in range(len(train_array) - window_size):
                X_train_all.append(
                    train_array[current_step:current_step + window_size]
                )
                y_train_all.append(
                    train_array[current_step + window_size][target_index]
                )

        if len(test_array) > window_size:
            for current_step in range(len(test_array) - window_size):
                X_test_all.append(
                    test_array[current_step:current_step + window_size]
                )
                y_test_all.append(
                    test_array[current_step + window_size][target_index]
                )

    return (
        np.asarray(X_train_all, dtype=float),
        np.asarray(y_train_all, dtype=float),
        np.asarray(X_test_all, dtype=float),
        np.asarray(y_test_all, dtype=float),
    )


def build_convlstm_model(window_size, num_features):
    model = Sequential()
    model.add(tf.keras.Input(shape=(window_size, num_features)))
    model.add(Conv1D(filters=64, kernel_size=2, activation="relu"))
    model.add(LSTM(50, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")

    return model


def train_and_evaluate_nn(
    df,
    method_name,
    missing_pattern,
    missing_percent,
    num_missing_timesteps,
    num_missing_rows,
    prediction_target,
):
    """
    Trains and evaluates Conv1D + LSTM on one dataframe and one prediction target.
    Logs prediction RMSE.
    """

    if prediction_target not in nn_features:
        raise ValueError(
            f"Prediction target '{prediction_target}' must be in nn_features."
        )

    df = df.sort_values(["zip", "time"]).copy()

    X_train, y_train, X_test, y_test = train_test_split_time_windows_by_zip(
        df=df,
        features=nn_features,
        target_col=prediction_target,
        window_size=window_size,
        train_fraction=train_fraction,
    )

    if len(X_train) == 0 or len(X_test) == 0:
        print(
            f"Skipping NN for method={method_name}, target={prediction_target}: "
            f"not enough windows."
        )
        return np.nan

    if not np.isfinite(X_train).all():
        raise ValueError(
            f"X_train contains NaN/inf for method={method_name}, "
            f"target={prediction_target}"
        )

    if not np.isfinite(X_test).all():
        raise ValueError(
            f"X_test contains NaN/inf for method={method_name}, "
            f"target={prediction_target}"
        )

    if not np.isfinite(y_train).all():
        raise ValueError(
            f"y_train contains NaN/inf for method={method_name}, "
            f"target={prediction_target}"
        )

    if not np.isfinite(y_test).all():
        raise ValueError(
            f"y_test contains NaN/inf for method={method_name}, "
            f"target={prediction_target}"
        )

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(GLOBAL_SEED)

    model = build_convlstm_model(
        window_size=window_size,
        num_features=len(nn_features),
    )

    start_time = time.time()

    model.fit(
        X_train,
        y_train,
        epochs=num_training_rounds,
        verbose=0,
    )

    predictions = model.predict(X_test, verbose=0).flatten()

    runtime_seconds = time.time() - start_time

    prediction_rmse = finite_rmse(y_test, predictions)

    if method_name == "clean_baseline":
        log_baseline_prediction(
            feature=prediction_target,
            prediction_rmse=prediction_rmse,
            runtime_seconds=runtime_seconds,
        )
    else:
        log_prediction_rmse(
            method=method_name,
            missing_pattern=missing_pattern,
            missing_percent=missing_percent,
            feature=prediction_target,
            prediction_rmse=prediction_rmse,
            runtime_seconds=runtime_seconds,
            num_missing_timesteps=num_missing_timesteps,
            num_missing_rows=num_missing_rows,
        )

    print(
        f"NN Prediction | "
        f"method={method_name:30s} | "
        f"target={prediction_target:20s} | "
        f"RMSE={prediction_rmse:.4f} | "
        f"Runtime={runtime_seconds:.4f}s"
    )

    return prediction_rmse


# =====================================================================
# Baseline Neural Network on Clean Dataset
# =====================================================================

print("\n" + "=" * 70)
print("TRAINING BASELINE NN ON CLEAN DATASET")
print("=" * 70)

train_and_evaluate_nn(
    df=clean_control_data,
    method_name="clean_baseline",
    missing_pattern="none",
    missing_percent=0.0,
    num_missing_timesteps=0,
    num_missing_rows=0,
    prediction_target=target_col,
)


# =====================================================================
# Main Experiment Loop
# =====================================================================

all_filled_datasets = {}

for percent in percentages_to_test:
    print(
        f"\n\n********** STARTING TEST FOR "
        f"{int(percent * 100)}% MISSING DATA **********"
    )

    # -----------------------------------------------------------------
    # Random timestep holes
    # -----------------------------------------------------------------

    (
        test_data_random_holes,
        real_values_random,
        random_row_indexes,
        num_random_missing_timesteps,
    ) = create_random_timestep_holes(clean_control_data, percent)

    random_filled_datasets = run_all_interpolation_methods(
        test_data=test_data_random_holes,
        real_values=real_values_random,
        hole_indexes=random_row_indexes,
        missing_pattern="random_timesteps",
        missing_percent=percent,
        num_missing_timesteps=num_random_missing_timesteps,
    )

    all_filled_datasets[("random_timesteps", percent)] = random_filled_datasets

    # Train/test NN for each random-hole interpolated dataset
    for method_name, filled_df in random_filled_datasets.items():
        train_and_evaluate_nn(
            df=filled_df,
            method_name=method_name,
            missing_pattern="random_timesteps",
            missing_percent=percent,
            num_missing_timesteps=num_random_missing_timesteps,
            num_missing_rows=len(random_row_indexes),
            prediction_target=target_col,
        )

    # -----------------------------------------------------------------
    # Block timestep holes
    # -----------------------------------------------------------------

    (
        test_data_block_holes,
        real_values_blocks,
        block_row_indexes,
        num_block_missing_timesteps,
    ) = create_block_timestep_holes(
        clean_df=clean_control_data,
        percent=percent,
        block_size=block_size,
    )

    block_filled_datasets = run_all_interpolation_methods(
        test_data=test_data_block_holes,
        real_values=real_values_blocks,
        hole_indexes=block_row_indexes,
        missing_pattern=f"block_timesteps_{block_size}",
        missing_percent=percent,
        num_missing_timesteps=num_block_missing_timesteps,
    )

    all_filled_datasets[(f"block_timesteps_{block_size}", percent)] = block_filled_datasets

    # Train/test NN for each block-hole interpolated dataset
    for method_name, filled_df in block_filled_datasets.items():
        train_and_evaluate_nn(
            df=filled_df,
            method_name=method_name,
            missing_pattern=f"block_timesteps_{block_size}",
            missing_percent=percent,
            num_missing_timesteps=num_block_missing_timesteps,
            num_missing_rows=len(block_row_indexes),
            prediction_target=target_col,
        )


save_log()

print("\nExperiment complete.")