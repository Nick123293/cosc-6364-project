# README: Running `project.py`

This file explains how to configure and run `project.py`.

## 1. Global Configuration Variables

The main inputs to the script are defined as global variables near the top of `project.py`, around lines 24–69.

You can change the following values:

| Variable | Approx. Line | Description |
|---|---:|---|
| `GLOBAL_SEED` | 24 | Sets the random seed for reproducibility. |
| `dataset_file_path` | 34 | File path to the input dataset. |
| `percentages_to_test` | 36 | Missing-data percentages to test. |
| `block_size` | 38 | Block size for block-based data removal. |
| `target_col` | 40 | Prediction target for the neural network. |
| `interpolation_features` | 42–54 | Features that will be interpolated. |
| `window_size` | 58 | Window size used by the LSTM. |
| `train_fraction` | 59 | Train/test split used for the LSTM. |
| `num_training_rounds` | 60 | Number of epochs used to train the neural network. |
| `log_file_path` | 67 | Output file path for the experiment log. |

## 2. Required Shape Data for Spatial Interpolation

Spatial interpolation requires shape data so the script can determine which ZIP codes are closest to each other.

The required shape files are the `houston_filtered*` files included in the ZIP folder.

**All `houston_filtered*` files must be in the same directory as `project.py` for spatial interpolation to work correctly.**

These files are used by the GeoPandas-based spatial interpolation methods.

## 3. Choosing Interpolation Methods

The main variable that determines which interpolation methods are tested is `INTERPOLATION_METHODS`.

This variable is located around lines 422–511.

`INTERPOLATION_METHODS` is a list of dictionaries. Each dictionary describes one interpolation method that the script should test.

For each method listed in `INTERPOLATION_METHODS`, the script will:

1. Run that interpolation method for every missing-data percentage.
2. Train and test the neural network using the filled dataset.
3. Save interpolation RMSE, prediction RMSE, and runtime information to the output log file.

## 4. Supported Dictionary Keys

The supported keys inside each interpolation method dictionary are:

- `name`
- `type`
- `method`
- `degree`
- `kwargs`

Not every key is required for every method. Some keys are only used for specific `type` values.

For example, `method` is only used when `type` is `pandas_interpolate`.

The `name` key can be assigned any descriptive value. It is only used for logging and identifying results in the output file.

## 5. Supported Interpolation Method Types

The currently supported `type` values are:

- `average`
- `pandas_interpolate`
- `local_newton`
- `geopandas_idw`
- `geopandas_spatial_spline`

The method specifications are passed into the function `fill_feature_with_method`.

This function reads the `type` and other method information to determine which interpolation function should be called.

## 6. Method Type Requirements

### 6.1 `average`

The `average` type performs average-based imputation.

Required keys:

    {
        "name": "average_by_zip",
        "type": "average"
    }

No additional information is required.

### 6.2 `geopandas_idw`

The `geopandas_idw` type performs spatial interpolation using inverse distance weighting.

Required keys:

    {
        "name": "spatial_idw_geopandas",
        "type": "geopandas_idw"
    }

No additional information is required beyond `name` and `type`.

### 6.3 `pandas_interpolate`

The `pandas_interpolate` type uses Pandas interpolation methods.

Required keys:

    {
        "name": "spline_3_by_zip",
        "type": "pandas_interpolate",
        "method": "spline",
        "kwargs": {"order": 3}
    }

The `method` key determines which Pandas interpolation method is used.

Internally, this runs Pandas interpolation in the form `interpolate(method=method, **kwargs)`.

Therefore, any interpolation method supported by Pandas can be used here.

The required `kwargs` depend on the selected Pandas interpolation method.

For example, spline interpolation requires an `order` argument:

    "kwargs": {"order": 3}

This specifies the degree of the spline polynomials.

The other Pandas interpolation methods used in this project, such as `pchip`, `akima`, and `linear`, do not require additional keyword arguments, so their `kwargs` value can be empty:

    "kwargs": {}

For more information about supported Pandas interpolation methods, see:

https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html

### 6.4 `local_newton`

The `local_newton` type uses the project’s implementation of local Newton interpolation.

Required keys:

    {
        "name": "local_newton_degree_3_by_zip",
        "type": "local_newton",
        "degree": 3
    }

The `degree` key specifies the degree of the local Newton polynomial.

For example, `degree: 3` means the method will use a degree-3 local Newton interpolation polynomial.

### 6.5 `geopandas_spatial_spline`

The `geopandas_spatial_spline` type uses SciPy’s `RBFInterpolator` from `scipy.interpolate`.

This method performs spatial interpolation using ZIP-code centroid coordinates.

Required keys:

    {
        "name": "spatial_spline_thin_plate",
        "type": "geopandas_spatial_spline",
        "kwargs": {
            "kernel": "thin_plate_spline",
            "smoothing": 0.0,
            "neighbors": None
        }
    }

The `kwargs` dictionary must include:

- `kernel`
- `smoothing`
- `neighbors`

#### `kernel`

The only tested kernel is `thin_plate_spline`.

Other kernels may work, but they have not been tested in this project.

#### `smoothing`

For true interpolation, use `smoothing: 0.0`.

If `smoothing` is not zero, the method is allowed to not exactly pass through the known data points. This can produce a smoother approximation, but it is no longer exact interpolation.

No smoothing values other than `0.0` were tested in this project.

#### `neighbors`

The `neighbors` argument controls whether the spatial spline is global or local.

If `neighbors` is set to `None`, then the method creates one global spatial curve or surface using all known spatial points in the dataset for each timestamp.

If `neighbors` is given an integer value, such as `20`, then the method creates a local spatial curve or surface using only the 20 nearest known spatial points.

Example:

    {
        "name": "spatial_spline_thin_plate_neighbors_20",
        "type": "geopandas_spatial_spline",
        "kwargs": {
            "kernel": "thin_plate_spline",
            "smoothing": 0.0,
            "neighbors": 20
        }
    }

This uses a local thin-plate spline interpolator based on the nearest 20 spatial points.

## 7. Example `INTERPOLATION_METHODS` Entries

A temporal spline interpolation method can be added like this:

    {
        "name": "spline_3_by_zip",
        "type": "pandas_interpolate",
        "method": "spline",
        "kwargs": {"order": 3}
    }

A spatial thin-plate spline interpolation method can be added like this:

    {
        "name": "spatial_spline_thin_plate",
        "type": "geopandas_spatial_spline",
        "kwargs": {
            "kernel": "thin_plate_spline",
            "smoothing": 0.0,
            "neighbors": None
        }
    }

## 8. Output

After the script finishes, results are saved to the path specified by `log_file_path`.

The output log contains interpolation RMSE, neural-network prediction RMSE, and runtime information for each interpolation method and missing-data percentage.
