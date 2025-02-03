import os

os.chdir(os.environ["SAM_REPO_PATH"])
import warnings

warnings.filterwarnings(
    "ignore",
    message="It is advised to add a data point where survival_rate becomes 0 at the highest concentration.",
)


import numpy as np
import seaborn as sns
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import argparse
import sys
import pandas as pd

sys.path.append("docs/generation_scripts/")

from img_creation.lc_increase_data import (
    calculate_lc_trajectories,
    gen_dose_response_frame,
    gen_experiment_res_frame,
    gen_mean_curves,
    STRESSES,
)

from img_creation.e_fac_mod import overwrite_examples_with_efac
import json
from sam import read_data


def gen_dot_plotly(input_df, norm_by_effect_range=False):
    y_title = "Increase of Toxicant Sensitivity"

    if norm_by_effect_range:
        input_df = input_df.copy()
        input_df["true_10_frac"] /= input_df.effect_range
        input_df["true_50_frac"] /= input_df.effect_range
        input_df["sam_10_frac"] /= input_df.effect_range
        input_df["sam_50_frac"] /= input_df.effect_range

        y_title = "Increase of Toxicant Sensitivity / Effect Range"

    assert (
        len(input_df.days.unique()) == 1
    ), f"Found these days - must be unique!{input_df.days.unique()}"

    fig = make_subplots(rows=1, cols=2)

    color_mapping = {
        "Measurements": "blue",
        "Predictions": "green",
    }

    name_to_id = []

    def add_points(y_col, col):
        for _, row in input_df.iterrows():
            # Measurements
            fig.add_trace(
                go.Scatter(
                    x=[row.stress_level],
                    y=[row[y_col]],
                    mode="markers",
                    name=row.Name,
                    hovertext=(
                        f"<br><b>Name</b>: {row.Name}"
                        f"<br><b>Experiment</b>: {row.experiment_name}"
                        f"<br><b>Main Stressor</b>: {row.chemical}"
                        f"<br><b>Additional Stressor</b>: {row.stress_name}"
                        f"<br><b>Duration</b>: {row.days}"
                        f"<br><b>Organism</b>: {row.organism}"
                    ),
                    showlegend=False,
                    marker=dict(color=color_mapping["Measurements"]),
                ),
                row=1,
                col=col,
            )
            name_to_id.append(f"{row.Name}_{y_col}")

            # Predictions
            fig.add_trace(
                go.Scatter(
                    x=[row.stress_level],
                    y=[row[y_col.replace("true", "sam")]],
                    mode="markers",
                    name=row.Name,
                    hovertext=(
                        f"<br><b>Name</b>: {row.Name}"
                        f"<br><b>Experiment</b>: {row.experiment_name}"
                        f"<br><b>Main Stressor</b>: {row.chemical}"
                        f"<br><b>Additional Stressor</b>: {row.stress_name}"
                        f"<br><b>Duration</b>: {row.days}"
                        f"<br><b>Organism</b>: {row.organism}"
                    ),
                    showlegend=False,
                    marker=dict(color=color_mapping["Predictions"]),
                ),
                row=1,
                col=col,
            )
            name_to_id.append(f"{row.Name}_{y_col}")

    # Add measurement and prediction points for LC10 and LC50
    add_points("true_10_frac", 1)
    add_points("true_50_frac", 2)

    # ------------------------------------------------------
    # Add color legend items for "Measurements" and "Predictions" (points)
    # ------------------------------------------------------
    for name, color in color_mapping.items():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",  # Markers for the color legend
                marker=dict(color=color),
                name=name,
            )
        )
        name_to_id.append(f"color_{name}")

    # ------------------------------------------------------
    # Add dashed regression lines (log-based) for "all data", always visible
    # ------------------------------------------------------
    x_data = input_df["stress_level"].values

    # For LC10 subplot
    y_data_true_10 = input_df["true_10_frac"].values
    slope_10_m, intercept_10_m = np.polyfit(x_data, np.log10(y_data_true_10), 1)
    x_line = np.linspace(x_data.min(), x_data.max(), 100)
    y_line = 10 ** (slope_10_m * x_line + intercept_10_m)

    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            line=dict(color="blue", dash="dash"),
            name="Measurements (Regression)",
            showlegend=True,  # Only show legend on left
        ),
        row=1,
        col=1,
    )
    name_to_id.append("reg_meas_10")

    y_data_pred_10 = input_df["sam_10_frac"].values
    slope_10_p, intercept_10_p = np.polyfit(x_data, np.log10(y_data_pred_10), 1)
    y_line = 10 ** (slope_10_p * x_line + intercept_10_p)

    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            line=dict(color="green", dash="dash"),
            name="Predictions (Regression)",
            showlegend=True,  # Only show legend on left
        ),
        row=1,
        col=1,
    )
    name_to_id.append("reg_pred_10")

    # For LC50 subplot
    y_data_true_50 = input_df["true_50_frac"].values
    slope_50_m, intercept_50_m = np.polyfit(x_data, np.log10(y_data_true_50), 1)
    y_line = 10 ** (slope_50_m * x_line + intercept_50_m)

    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            line=dict(color="blue", dash="dash"),
            name="Measurements (Regression)",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    name_to_id.append("reg_meas_50")

    y_data_pred_50 = input_df["sam_50_frac"].values
    slope_50_p, intercept_50_p = np.polyfit(x_data, np.log10(y_data_pred_50), 1)
    y_line = 10 ** (slope_50_p * x_line + intercept_50_p)

    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            line=dict(color="green", dash="dash"),
            name="Predictions (Regression)",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    name_to_id.append("reg_pred_50")

    # ------------------------------------------------------
    # Dropdown logic
    # ------------------------------------------------------
    cleaner = {
        "organism": "Organism",
        "experiment_name": "Experiment",
    }

    def get_visible(df, name):
        valid = {
            # regression lines
            "reg_meas_10",
            "reg_pred_10",
            "reg_meas_50",
            "reg_pred_50",
            # color legend items
            "color_Measurements",
            "color_Predictions",
        }

        for n in df.Name.values:
            valid.add(f"{n}_true_10_frac")
            valid.add(f"{n}_true_50_frac")

        for v in valid:
            assert v in name_to_id, f"'{v}' not found in name_to_id!"

        return [i in valid for i in name_to_id]

    # Sanity check for "All"
    assert len(get_visible(input_df, "all")) == len(fig.data)

    buttons = [
        dict(
            label="All",
            method="update",
            args=[{"visible": get_visible(input_df, "all")}],
        )
    ]

    for key, label_name in cleaner.items():
        for val, frame in input_df.groupby(key):
            group_label = f"{label_name} = {val}"
            buttons.append(
                dict(
                    label=group_label,
                    method="update",
                    args=[{"visible": get_visible(frame, group_label)}],
                )
            )

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
            ),
        ],
        annotations=[
            dict(
                x=0.2,
                y=1.1,
                xref="paper",
                yref="paper",
                text="LC 10",
                showarrow=False,
                font=dict(size=16),
            ),
            dict(
                x=0.8,
                y=1.1,
                xref="paper",
                yref="paper",
                text="LC 50",
                showarrow=False,
                font=dict(size=16),
            ),
        ],
    )

    # Increase vertical space to avoid squished appearance
    fig.update_layout(height=800)

    fig.update_xaxes(title_text="Environmental Stress", row=1, col=1)
    fig.update_yaxes(title_text=y_title, type="log", row=1, col=1)
    fig.update_xaxes(title_text="Environmental Stress", row=1, col=2)
    fig.update_yaxes(title_text=y_title, type="log", row=1, col=2)

    return fig


def generate_html_page(
    df: pd.DataFrame,
    norm_by_effect_range: bool = True,
    days: list[int] = [2, 7, 14, 21],
    additional_site_info: str = "",
) -> str:
    """
    Generate a single HTML string containing:
      - A general explanation of the plots and formulas.
      - Any additional info (e.g. e-factor details, outlier lists) as <pre> text.
      - Separate headings and embedded Plotly figures for each day in `days`.

    Returns the entire HTML document as a string, ready to be saved or served.
    """

    page_sections = []

    # 1) Add an overview at the top
    page_sections.append(
        """
<h2>Overview & Explanation</h2>
<p>
  This page shows how the LC<sub>10</sub> (left graph) and LC<sub>50</sub> (right graph)
  are increased by environmental co-stressors (compared to a control).
  The blue points are measured data,
  the green points are our model predictions.
  Dashed lines represent a regression (in log-space) for all data combined.
</p>
"""
    )

    # 2) Insert any extra string the user wants, wrapped in <pre> so newlines render
    if additional_site_info.strip():
        page_sections.append(
            f"<h3>Additional Info</h3>\n<pre>{additional_site_info}</pre>"
        )
    page_sections.append("<hr>")
    # 3) Plot each day’s figure
    import plotly.io as pio

    for day in days:
        subset = df.query("days == @day")
        if subset.empty:
            page_sections.append(f"<h3>Day {day}</h3>\n<p>No data available.</p>")
            continue

        page_sections.append(f"<h3>Day {day}</h3>")

        fig = gen_dot_plotly(subset, norm_by_effect_range=norm_by_effect_range)
        # Convert the figure to an HTML snippet
        fig_html_snippet = pio.to_html(fig, include_plotlyjs=False, full_html=False)

        page_sections.append(fig_html_snippet)
        page_sections.append("<hr>")

    # Final combined page, single <head> & <body>
    final_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <!-- Load plotly.js once at the top -->
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
{''.join(page_sections)}
</body>
</html>
"""
    return final_html


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import json
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir4imgs", type=str)
    args = parser.parse_args()

    dir4imgs = Path(args.dir4imgs)
    dir4imgs.mkdir(exist_ok=True, parents=True)

    original_frame = gen_experiment_res_frame()

    with open("docs/add_sys_examples.json") as f:
        sys_examples = json.load(f)

    outlier_text = "\n".join(
        [f"{read_data(a).meta.title} - {b}" for a, b, _ in sys_examples]
    )

    # Example #1: e-factor=0.25
    df_025 = overwrite_examples_with_efac(0.25, original_frame)
    mask = df_025.chemical.apply(lambda x: x not in ["Salt (NaCl)", "Copper"])
    df_025_filtered = df_025[mask]

    html_025 = generate_html_page(
        df_025_filtered,
        norm_by_effect_range=False,
        additional_site_info="Experiments Containing Salt (NaCl) or Copper as Main Stressors have been disgarded for this experiment\n\n"
        f"Using new_e=e *0.25 for outliers:\n{outlier_text}",
    )
    with open(
        dir4imgs / "lcs_with_e_fac_025_filtered.html", "w", encoding="utf-8"
    ) as f:
        f.write(html_025)

    # Example #2: e-factor="optimal"
    df_opt = overwrite_examples_with_efac("optimal", original_frame)
    mask = df_opt.chemical.apply(lambda x: x not in ["Salt (NaCl)", "Copper"])
    df_opt_filtered = df_opt[mask]

    html_opt = generate_html_page(
        df_opt_filtered,
        norm_by_effect_range=False,
        additional_site_info="Experiments Containing Salt (NaCl) or Copper as Main Stressors have been disgarded for this experiment\n"
        f"Using an optimal e-factor for each of these outliers:\n{outlier_text}",
    )
    with open(
        dir4imgs / "lcs_with_e_fac_optimal_filtered.html", "w", encoding="utf-8"
    ) as f:
        f.write(html_opt)

    # Example #3: e-factor="optimal", normalized by effect range
    df_opt_no_filter = overwrite_examples_with_efac("optimal", original_frame)

    html_opt_norm = generate_html_page(
        df_opt_no_filter,
        norm_by_effect_range=True,
        additional_site_info=(
            "Effect-range is defined as Control LC95 / LC5.\n"
            f"Using an optimal e-factor for each of these outliers:\n{outlier_text}"
        ),
    )
    with open(
        dir4imgs / "lcs_optimal_effect_range_norm.html", "w", encoding="utf-8"
    ) as f:
        f.write(html_opt_norm)
