import os

os.chdir(os.environ["SAM_REPO_PATH"])
import warnings

warnings.filterwarnings(
    "ignore",
    message="It is advised to add a data point where survival_rate becomes 0 at the highest concentration.",
)


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import argparse
import sys
import pandas as pd

sys.path.append("docs/generation_scripts/")

from img_creation.lc_increase_data import gen_experiment_res_frame
from img_creation.e_fac_mod import overwrite_examples_with_efac
import json
from sam import read_data
from sklearn.metrics import mean_absolute_percentage_error, r2_score


def gen_dot_plotly(
    input_df,
    x_axis_key: str = "stress_level",
    x_axis_name: str = "Environmental Stress",
    norm_by_effect_range=False,
    prediction_key: str = "sam_{}_frac",
    prediction_color="green",
    prediction_name="SAM Model",
):
    pred_10_frac = prediction_key.format(10)
    pred_50_frac = prediction_key.format(50)

    y_title = "Increase of Toxicant Sensitivity"

    if norm_by_effect_range:
        input_df = input_df.copy()
        input_df["true_10_frac"] /= input_df.effect_range
        input_df["true_50_frac"] /= input_df.effect_range
        input_df[pred_10_frac] /= input_df.effect_range
        input_df[pred_50_frac] /= input_df.effect_range

        y_title = "Increase of Toxicant Sensitivity / Effect Range"

    assert (
        len(input_df.days.unique()) == 1
    ), f"Found these days - must be unique!{input_df.days.unique()}"

    fig = make_subplots(rows=1, cols=2)

    color_mapping = {
        "Measurements": "blue",
        "Predictions": prediction_color,
    }

    name_to_id = []

    def add_points(y_col, pred_y_key, col):
        for _, row in input_df.iterrows():
            add_text = (
                f"<br><b>Effect Range</b>: {float(row.effect_range) :.2f}"
                if norm_by_effect_range
                else ""
            )

            # Measurements
            fig.add_trace(
                go.Scatter(
                    x=[row[x_axis_key]],
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
                        f"<br><b>E Parameter Factor</b>: {float(row.e_fac):.2f}"
                        + add_text
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
                    x=[row[x_axis_key]],
                    y=[row[pred_y_key]],
                    mode="markers",
                    name=row.Name,
                    hovertext=(
                        f"<br><b>Name</b>: {row.Name}"
                        f"<br><b>Experiment</b>: {row.experiment_name}"
                        f"<br><b>Main Stressor</b>: {row.chemical}"
                        f"<br><b>Additional Stressor</b>: {row.stress_name}"
                        f"<br><b>Duration</b>: {row.days}"
                        f"<br><b>Organism</b>: {row.organism}"
                        f"<br><b>E Parameter Factor</b>: {float(row.e_fac):.2f}"
                        + add_text
                    ),
                    showlegend=False,
                    marker=dict(color=color_mapping["Predictions"]),
                ),
                row=1,
                col=col,
            )
            name_to_id.append(f"{row.Name}_{y_col}")

            fig.add_trace(
                go.Scatter(
                    x=[row[x_axis_key], row[x_axis_key]],
                    y=[row[y_col], row[pred_y_key]],
                    mode="lines",
                    line=dict(color="grey", width=2),
                    showlegend=False,
                    opacity=0.5,
                ),
                row=1,
                col=col,
            )
            name_to_id.append(f"{row.Name}_{y_col}")

    # Add measurement and prediction points for LC10 and LC50
    add_points("true_10_frac", pred_10_frac, 1)
    add_points("true_50_frac", pred_50_frac, 2)

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

    def add_regression_line(y_key: str, name_2_id_name: str, col: int) -> float:
        if y_key.startswith("true"):
            color = "blue"
            legend_name = "Measurements (Regression)"
        else:
            color = prediction_color
            legend_name = "Predictions (Regression)"

        x_data = input_df[x_axis_key].values

        # For LC10 subplot
        y_data = input_df[y_key].values
        slope, intercept = np.polyfit(x_data, np.log10(y_data), 1)
        x_line = np.linspace(x_data.min(), x_data.max(), 100)
        y_line = 10 ** (slope * x_line + intercept)

        preds = 10 ** (slope * x_data + intercept)
        mape = mean_absolute_percentage_error(y_data, preds)
        r2 = r2_score(y_data, preds)

        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                line=dict(color=color, dash="dash"),
                name=legend_name,
                showlegend=col == 2,
            ),
            row=1,
            col=col,
        )
        name_to_id.append(name_2_id_name)
        return mape, r2

    mapes_and_r2s_10 = [
        (
            mean_absolute_percentage_error(
                input_df.true_10_frac, input_df[pred_10_frac]
            ),
            r2_score(input_df.true_10_frac, input_df[pred_10_frac]),
        ),
        add_regression_line(
            y_key="true_10_frac",
            name_2_id_name="reg_meas_10",
            col=1,
        ),
        add_regression_line(
            y_key=pred_10_frac,
            name_2_id_name="reg_pred_10",
            col=1,
        ),
    ]

    mapes_and_r2s_50 = [
        (
            mean_absolute_percentage_error(
                input_df.true_50_frac, input_df[pred_50_frac]
            ),
            r2_score(input_df.true_10_frac, input_df[pred_10_frac]),
        ),
        add_regression_line(
            y_key="true_50_frac",
            name_2_id_name="reg_meas_50",
            col=2,
        ),
        add_regression_line(
            y_key=pred_50_frac,
            name_2_id_name="reg_pred_50",
            col=2,
        ),
    ]

    fig_text = f"""
<table style="margin: auto; border-collapse: collapse; font-size: 14px; text-align: center;">
    <caption style="font-weight: bold; margin-bottom: 5px;">MAPEs (Mean Absolute Percentage Errors)</caption>
    <thead>
        <tr>
            <th style="border: 1px solid #ccc; padding: 5px;">LC</th>
            <th style="border: 1px solid #ccc; padding: 5px;">{prediction_name}</th>
            <th style="border: 1px solid #ccc; padding: 5px;">Measurement Regression</th>
            <th style="border: 1px solid #ccc; padding: 5px;">Prediction Regression</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="border: 1px solid #ccc; padding: 5px;">10</td>
            <td style="border: 1px solid #ccc; padding: 5px;">{mapes_and_r2s_10[0][0]:.3f}</td>
            <td style="border: 1px solid #ccc; padding: 5px;">{mapes_and_r2s_10[1][0]:.3f}</td>
            <td style="border: 1px solid #ccc; padding: 5px;">{mapes_and_r2s_10[2][0]:.3f}</td>
        </tr>
        <tr>
            <td style="border: 1px solid #ccc; padding: 5px;">50</td>
            <td style="border: 1px solid #ccc; padding: 5px;">{mapes_and_r2s_50[0][0]:.3f}</td>
            <td style="border: 1px solid #ccc; padding: 5px;">{mapes_and_r2s_50[1][0]:.3f}</td>
            <td style="border: 1px solid #ccc; padding: 5px;">{mapes_and_r2s_50[2][0]:.3f}</td>
        </tr>
    </tbody>
</table>
<br><br>
<table style="margin: auto; border-collapse: collapse; font-size: 14px; text-align: center;">
    <caption style="font-weight: bold; margin-bottom: 5px;">R^2 Scores</caption>
    <thead>
        <tr>
            <th style="border: 1px solid #ccc; padding: 5px;">LC</th>
            <th style="border: 1px solid #ccc; padding: 5px;">{prediction_name}</th>
            <th style="border: 1px solid #ccc; padding: 5px;">Measurement Regression</th>
            <th style="border: 1px solid #ccc; padding: 5px;">Prediction Regression</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="border: 1px solid #ccc; padding: 5px;">10</td>
            <td style="border: 1px solid #ccc; padding: 5px;">{mapes_and_r2s_10[0][1]:.3f}</td>
            <td style="border: 1px solid #ccc; padding: 5px;">{mapes_and_r2s_10[1][1]:.3f}</td>
            <td style="border: 1px solid #ccc; padding: 5px;">{mapes_and_r2s_10[2][1]:.3f}</td>
        </tr>
        <tr>
            <td style="border: 1px solid #ccc; padding: 5px;">50</td>
            <td style="border: 1px solid #ccc; padding: 5px;">{mapes_and_r2s_50[0][1]:.3f}</td>
            <td style="border: 1px solid #ccc; padding: 5px;">{mapes_and_r2s_50[1][1]:.3f}</td>
            <td style="border: 1px solid #ccc; padding: 5px;">{mapes_and_r2s_50[2][1]:.3f}</td>
        </tr>
    </tbody>
</table>
"""

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

    fig.update_xaxes(title_text=x_axis_name, row=1, col=1)
    fig.update_yaxes(title_text=y_title, type="log", row=1, col=1)
    fig.update_xaxes(title_text=x_axis_name, row=1, col=2)
    fig.update_yaxes(title_text=y_title, type="log", row=1, col=2)

    return fig, fig_text


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

        page_sections.append("<h4>Generalized Enviromental Stress</h4>")
        fig, figtext = gen_dot_plotly(
            subset,
            norm_by_effect_range=norm_by_effect_range,
            x_axis_name="Generalized Enviromental Stress",
        )
        page_sections.append(pio.to_html(fig, include_plotlyjs=False, full_html=False))
        page_sections.append(
            f"<div style='text-align: center; margin-top: 10px;'>{figtext}</div>"
        )

        page_sections.append("<br><br><h4>Environmental Stress Factor - SAM</h4>")
        fig, figtext = gen_dot_plotly(
            subset,
            norm_by_effect_range=norm_by_effect_range,
            x_axis_name="Environmental Stress Factor",
            x_axis_key="control_div",
        )
        page_sections.append(pio.to_html(fig, include_plotlyjs=False, full_html=False))
        page_sections.append(
            f"<div style='text-align: center; margin-top: 10px;'>{figtext}</div>"
        )

        page_sections.append(
            "<h4><br><br>Environmental Stress Factor - Concentration Addition</h4>"
        )
        fig, figtext = gen_dot_plotly(
            subset,
            norm_by_effect_range=norm_by_effect_range,
            x_axis_name="Environmental Stress Factor",
            x_axis_key="control_div",
            prediction_color="orange",
            prediction_key="ca_{}_frac",
            prediction_name="Concentration Addition",
        )
        page_sections.append(pio.to_html(fig, include_plotlyjs=False, full_html=False))
        page_sections.append(
            f"<div style='text-align: center; margin-top: 10px;'>{figtext}</div>"
        )

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

    different_frames: list[tuple[pd.DataFrame, str, str]] = [
        (original_frame, "Keeping all e-factors untouched.", "original"),
        (
            overwrite_examples_with_efac(0.25, original_frame),
            f"Using new_e=e *0.25 for outliers:\n{outlier_text}",
            "e_fac_0.25",
        ),
        (
            overwrite_examples_with_efac("optimal", original_frame),
            f"Using an optimal e-factor for each of these outliers:\n{outlier_text}",
            "e_fac_optimal_outliers",
        ),
        (
            overwrite_examples_with_efac("optimal", original_frame, all_rows=True),
            "Using an optimal e-factor for all points.",
            "e_fac_optimal_all",
        ),
    ]

    def save_fig(name, fig):
        with open(dir4imgs / name, "w", encoding="utf-8") as f:
            f.write(fig)

    for df, additional_site_info, e_fac_name in different_frames:
        save_fig(
            f"sensitivity_increase_{e_fac_name}.html",
            generate_html_page(
                df,
                norm_by_effect_range=False,
                additional_site_info=additional_site_info,
            ),
        )

        save_fig(
            f"sensitivity_increase_effect_range_normed_{e_fac_name}.html",
            generate_html_page(
                df,
                norm_by_effect_range=True,
                additional_site_info="Effect-range is defined as Control LC75 / LC25.\n"
                + additional_site_info,
            ),
        )
