#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_prediction_rmse_from_json(json_path, target_feature="us_aqi"):

    with open(json_path, "r") as f:
        log_data = json.load(f)

    rows = []

    # ------------------------------------------------------------
    # Load baseline
    # ------------------------------------------------------------
    baseline_rmse = None

    baseline_block = (
        log_data
        .get("baseline", {})
        .get("prediction_rmse", {})
        .get(target_feature, {})
    )

    if "value" in baseline_block:
        baseline_rmse = baseline_block["value"]

    # ------------------------------------------------------------
    # Load interpolation-method prediction RMSEs
    # ------------------------------------------------------------
    interpolation_methods = log_data.get("interpolation_methods", {})

    for method_name, method_data in interpolation_methods.items():
        for missing_pattern, pattern_data in method_data.items():
            for missing_percent, percent_data in pattern_data.items():

                if target_feature not in percent_data:
                    continue

                feature_data = percent_data[target_feature]

                if "prediction_rmse" not in feature_data:
                    continue

                prediction_rmse = feature_data["prediction_rmse"]["value"]

                rows.append(
                    {
                        "method": method_name,
                        "missing_pattern": missing_pattern,
                        "missing_percent": float(missing_percent),
                        "prediction_rmse": prediction_rmse,
                    }
                )

    df = pd.DataFrame(rows)

    return df, baseline_rmse


def plot_prediction_rmse(df, baseline_rmse, output_path, target_feature="us_aqi"):
    """
    Creates one plot per missing pattern.

    Each plot compares interpolation methods across missing percentages.
    """

    if df.empty:
        raise ValueError("No prediction RMSE values found in the JSON file.")

    missing_patterns = sorted(df["missing_pattern"].unique())

    for missing_pattern in missing_patterns:
        pattern_df = df[df["missing_pattern"] == missing_pattern].copy()

        plt.figure(figsize=(12, 7))

        for method_name, method_df in pattern_df.groupby("method"):
            method_df = method_df.sort_values("missing_percent")

            plt.plot(
                method_df["missing_percent"] * 100,
                method_df["prediction_rmse"],
                marker="o",
                label=method_name,
            )

        if baseline_rmse is not None:
            plt.axhline(
                y=baseline_rmse,
                linestyle="--",
                label=f"clean_baseline ({baseline_rmse:.4f})",
            )

        plt.xlabel("Missing Data Percentage (%)")
        plt.ylabel(f"Prediction RMSE for {target_feature}")
        plt.title(f"NN Prediction RMSE by Interpolation Method\nPattern: {missing_pattern}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_file = Path(output_path)

        if len(missing_patterns) > 1:
            stem = output_file.stem
            suffix = output_file.suffix
            pattern_output = output_file.with_name(f"{stem}_{missing_pattern}{suffix}")
        else:
            pattern_output = output_file

        plt.savefig(pattern_output, dpi=300)
        print(f"Saved plot to: {pattern_output}")

        plt.close()


def plot_prediction_rmse_bar(df, baseline_rmse, output_path, target_feature="us_aqi"):
    """
    Creates a grouped bar chart.

    X-axis:
        missing pattern + missing percentage

    Bars:
        interpolation methods
    """

    if df.empty:
        raise ValueError("No prediction RMSE values found in the JSON file.")

    plot_df = df.copy()
    plot_df["case"] = (
        plot_df["missing_pattern"]
        + "\n"
        + (plot_df["missing_percent"] * 100).astype(int).astype(str)
        + "%"
    )

    pivot_df = plot_df.pivot_table(
        index="case",
        columns="method",
        values="prediction_rmse",
        aggfunc="mean",
    )

    ax = pivot_df.plot(
        kind="bar",
        figsize=(14, 7),
        rot=0,
    )

    if baseline_rmse is not None:
        ax.axhline(
            y=baseline_rmse,
            linestyle="--",
            label=f"clean_baseline ({baseline_rmse:.4f})",
        )

    ax.set_xlabel("Missing Pattern and Missing Percentage")
    ax.set_ylabel(f"Prediction RMSE for {target_feature}")
    ax.set_title(f"NN Prediction RMSE Comparison for {target_feature}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved bar plot to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot NN prediction RMSE from nested experiment_log.json."
    )

    parser.add_argument(
        "--json",
        default="experiment_log.json",
        help="Path to experiment_log.json.",
    )

    parser.add_argument(
        "--target-feature",
        default="us_aqi",
        help="Feature whose prediction RMSE should be plotted. Default: us_aqi.",
    )

    parser.add_argument(
        "--output",
        default="prediction_rmse_comparison.png",
        help="Output plot path.",
    )

    parser.add_argument(
        "--plot-type",
        choices=["line", "bar"],
        default="line",
        help="Type of plot to create: line or bar. Default: line.",
    )

    args = parser.parse_args()

    df, baseline_rmse = load_prediction_rmse_from_json(
        json_path=args.json,
        target_feature=args.target_feature,
    )

    print("\nLoaded prediction RMSE values:")
    print(df)

    if baseline_rmse is not None:
        print(f"\nBaseline RMSE for {args.target_feature}: {baseline_rmse:.4f}")
    else:
        print(f"\nNo baseline RMSE found for {args.target_feature}.")

    if args.plot_type == "line":
        plot_prediction_rmse(
            df=df,
            baseline_rmse=baseline_rmse,
            output_path=args.output,
            target_feature=args.target_feature,
        )
    else:
        plot_prediction_rmse_bar(
            df=df,
            baseline_rmse=baseline_rmse,
            output_path=args.output,
            target_feature=args.target_feature,
        )


if __name__ == "__main__":
    main()