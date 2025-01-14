import os

os.chdir(os.environ["SAM_REPO_PATH"])
import numpy as np
import seaborn as sns
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import argparse
import sys

sys.path.append("docs/generation_scripts/")

from img_creation.lc_increase_data import (
    calculate_lc_trajectories,
    gen_dose_response_frame,
    gen_experiment_res_frame,
    gen_mean_curves,
    STRESSES,
)


def gen_normed_ploty(cleaned):
    fig = make_subplots(rows=1, cols=2)

    color_mapping = {
        "Measurements": "blue",
        "Predictions": "green",
    }

    name_to_id = []

    def add_points(y_col, col):
        for _, row in cleaned.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=(row.stress_level,),
                    y=(row[y_col],),
                    mode="markers",
                    name=row.Name,
                    hovertext=f"<br><b>Name</b>: {row.Name}<br><b>Experiment</b>: {row.experiment_name} <br><b>Main Stressor</b>: {row.chemical}<br><b>Additional Stressor</b>: {row.stress_name}<br><b>Effect Range</b>: {row.effect_range :.2f}<br><b>Duration</b>: {row.days}<br><b>Organism</b>: {row.organism}",
                    showlegend=False,
                    line=dict(color=color_mapping["Measurements"]),
                ),
                row=1,
                col=col,
            )
            name_to_id.append(f"{row.Name}_{y_col}")

    add_points("true_10_frac", 1)
    add_points("true_50_frac", 2)

    cleaner = {
        "days": "Duration",
        "organism": "Organism",
        "experiment_name": "Experiment",
    }

    # color legend
    for name, color in color_mapping.items():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode=(
                    "lines"
                    if name not in ["Predictions", "Measurements"]
                    else "markers"
                ),
                line=dict(color=color),
                name=name,
            )
        )
        name_to_id.append(f"color_{name}")

    def get_visible(df, name):
        valid = {
            "color_Measurements",
            "color_Predictions",
        }
        for n in df.Name.values:
            valid.add(f"{n}_true_10_frac")
            valid.add(f"{n}_true_50_frac")

        for v in valid:
            assert v in name_to_id, f"{v} wrong!"

        return [i in valid for i in name_to_id]

    assert len(get_visible(cleaned, "all")) == len(fig.data)

    buttons = [
        dict(
            label="All",
            method="update",
            args=[{"visible": get_visible(cleaned, "all")}],
        )
    ]

    for key, label_name in cleaner.items():
        for val, frame in cleaned.groupby(key):
            name = f"{label_name} = {val}"

            buttons.append(
                dict(
                    label=name,
                    method="update",
                    args=[{"visible": get_visible(frame, name)}],
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

    fig.update_xaxes(title_text="Environmental Stress", row=1, col=1)
    fig.update_yaxes(
        title_text="Increase of Toxicant Sensitivity / Effect Range",
        type="log",
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text="Environmental Stress", row=1, col=2)
    fig.update_yaxes(
        title_text="Increase of Toxicant Sensitivity / Effect Range",
        type="log",
        row=1,
        col=2,
    )
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir4imgs", type=str)
    args = parser.parse_args()
    dir4imgs = Path(args.dir4imgs)
    dir4imgs.mkdir(exist_ok=True, parents=True)

    lc10, lc50 = calculate_lc_trajectories()

    cleaned_frame = gen_experiment_res_frame(with_effect_range=True)

    dot_fig = gen_normed_ploty(cleaned_frame)
    dot_fig.write_html(
        dir4imgs / "effect_range_normed_lcs.html",
    )
