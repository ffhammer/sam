import os

os.chdir(os.environ["SAM_REPO_PATH"])
from sam.concentration_response_fits import survival_to_stress
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.append("docs/generation_scripts/")
from img_creation.utils import create_dose_response_fits_frame
import argparse


def create_color_map(df):
    chemicals = df["chemical"].unique()
    color_map = {
        chemical: color
        for chemical, color in zip(
            chemicals, sns.color_palette("Set2", len(chemicals)).as_hex()
        )
    }
    color_map["Mean"] = "black"
    return color_map


def make_fig(surv_col, stres_col, df, color_map):
    color_map = color_map.copy()
    color_map["Selection Mean"] = "red"

    name_to_id = []

    x = np.linspace(1, 99, 1000)
    fig = make_subplots(rows=1, cols=2)

    def gen_traces(y_key, col):
        for _, row in df.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=row[y_key],
                    mode="lines",
                    name=row.Name,
                    line=dict(color=color_map[row.chemical]),
                    hovertext=f"<br><b>Name</b>: {row.Name}<br><b>Experiment</b>: {row.Experiment}<br><b>Duration</b>: {row.Duration}<br><b>Main Stressor</b>: {row.chemical}<br><b>Organism</b>: {row.Organism}",
                    showlegend=False,
                ),
                col=col,
                row=1,
            )

            name_to_id.append(f"line_{row.Name}_{y_key}")

    def add_means_old(df: pd.DataFrame, name):
        mean_curve = np.mean(np.stack(df[surv_col].values), axis=0)
        mean_stress = survival_to_stress(mean_curve / 100)
        key = "Mean" if name == "Mean" else "Selection Mean"

        fig.add_trace(
            go.Scatter(
                x=x,
                y=mean_curve,
                mode="lines",
                name=name,
                line=dict(color=color_map[key]),
                showlegend=False,
                opacity=0.7 if key != "Mean" else 1,
            ),
            col=1,
            row=1,
        )

        name_to_id.append(f"mean_{name}_surv")

        fig.add_trace(
            go.Scatter(
                x=x,
                y=mean_stress,
                mode="lines",
                name=name,
                line=dict(color=color_map[key]),
                showlegend=False,
                opacity=0.7 if key != "Mean" else 1,
            ),
            col=2,
            row=1,
        )

        name_to_id.append(f"mean_{name}_stress")

    def add_means_exp(df: pd.DataFrame, name: str):
        this_x = df.og_conc.iloc[0]

        mean_curve = np.mean(np.stack(df.og_surv.values), axis=0)
        mean_stress = survival_to_stress(mean_curve)
        key = "Selection Mean"

        fig.add_trace(
            go.Scatter(
                x=this_x,
                y=mean_curve * 100,
                mode="lines",
                name=name,
                line=dict(color=color_map[key]),
                showlegend=False,
                opacity=0.7 if key != "Mean" else 1,
            ),
            col=1,
            row=1,
        )

        name_to_id.append(f"mean_{name}_surv")

        fig.add_trace(
            go.Scatter(
                x=this_x,
                y=mean_stress,
                mode="lines",
                name=name,
                line=dict(color=color_map[key]),
                showlegend=False,
                opacity=0.7 if key != "Mean" else 1,
            ),
            col=2,
            row=1,
        )

        name_to_id.append(f"mean_{name}_stress")

    add_means_old(df, "Mean")

    cleaner = {
        "chemical": "Main Stressor",
        "Experiment": "Experiment",
        "Duration": "Duration",
        "Organism": "Organism",
    }

    for key, label_name in cleaner.items():
        for val, frame in df.groupby(key):
            name = f"{label_name} = {val}"
            if label_name != "Experiment":
                add_means_old(frame, name)
            else:
                add_means_exp(frame, name)

    gen_traces(surv_col, 1)
    gen_traces(stres_col, 2)

    def plot_experiment_same_x(frame, exp_name, y_key, col):
        for _, row in frame.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=row.og_conc,
                    y=(
                        row.og_surv * 100
                        if y_key == surv_col
                        else survival_to_stress(row.og_surv)
                    ),
                    mode="lines",
                    name=row.Name,
                    line=dict(color=color_map[row.chemical]),
                    hovertext=f"<br><b>Name</b>: {row.Name}<br><b>Experiment</b>: {row.Experiment}<br><b>Duration</b>: {row.Duration}<br><b>Main Stressor</b>: {row.chemical}<br><b>Organism</b>: {row.Organism}",
                    showlegend=False,
                ),
                col=col,
                row=1,
            )
            name_to_id.append(f"Experiment = {exp_name}_{y_key}")

    for val, frame in df.groupby("Experiment"):
        plot_experiment_same_x(frame, val, surv_col, 1)
        plot_experiment_same_x(frame, val, stres_col, 2)

    for chemical, color in color_map.items():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines" if chemical in ["Mean", "Selection Mean"] else "markers",
                line=dict(color=color),
                name=chemical,
            )
        )
        name_to_id.append(f"color_{chemical}")

    def normal_visible(df, name):
        valid = {"mean_Mean_stress", "mean_Mean_surv", "color_Mean"}

        for n in df.Name.values:
            valid.add(f"line_{n}_{surv_col}")
            valid.add(f"line_{n}_{stres_col}")

        for chem in df.chemical.unique():
            valid.add(f"color_{chem}")

        if len(df) > 1 and name != "All":
            valid.add(f"mean_{name}_surv")
            valid.add(f"mean_{name}_stress")
            valid.add("color_Selection Mean")

        for v in valid:
            assert v in name_to_id, f"{v} wrong!"

        return [i in valid for i in name_to_id]

    def exp_visible(df, name):
        valid = set()

        valid.add(f"{name}_{surv_col}")
        valid.add(f"{name}_{stres_col}")

        for chem in df.chemical.unique():
            valid.add(f"color_{chem}")

        if len(df) > 1 and name != "All":
            valid.add(f"mean_{name}_surv")
            valid.add(f"mean_{name}_stress")
            valid.add("color_Selection Mean")

        for v in valid:
            assert v in name_to_id, f"{v} wrong!"

        return [i in valid for i in name_to_id]

    def visible(df, name):
        if name.startswith("Experiment"):
            return exp_visible(df, name)
        else:
            return normal_visible(df, name)

    buttons = [
        dict(
            label="All",
            method="update",
            args=[{"visible": visible(df, "All")}],
        )
    ]
    assert len(visible(df, "All")) == len(fig.data)

    for key, label_name in cleaner.items():
        for val, frame in df.groupby(key):
            name = f"{label_name} = {val}"
            buttons.append(
                dict(
                    label=name,
                    method="update",
                    args=[{"visible": visible(frame, name)}],
                )
            )
    fig.update_yaxes(title_text="Survival Rate", row=1, col=1)
    fig.update_xaxes(
        title_text="LC (or actual concentration if Experiment is chosen)",
        type="log",
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="Stress", row=1, col=2)
    fig.update_xaxes(
        title_text="LC (or actual concentration if Experiment is chosen)",
        type="log",
        row=1,
        col=2,
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
                text="Survival",
                showarrow=False,
                font=dict(size=16),
            ),
            dict(
                x=0.8,
                y=1.1,
                xref="paper",
                yref="paper",
                text="Stress",
                showarrow=False,
                font=dict(size=16),
            ),
        ],
    )
    return fig


def graphic(surv_col, stres_col, df, color_map):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    x = np.linspace(1, 99, 1000)

    for _, row in df.iterrows():
        axs[0].plot(x, row[surv_col], label=row.chemical, color=color_map[row.chemical])

    for _, row in df.iterrows():
        axs[1].plot(
            x, row[stres_col], label=row.chemical, color=color_map[row.chemical]
        )

    mean_curve = np.mean(np.stack(df[surv_col].values), axis=0)
    mean_stress = survival_to_stress(mean_curve / 100)

    axs[0].plot(x, mean_curve, label="Mean", color=color_map["Mean"], linewidth=3)
    axs[1].plot(x, mean_stress, label="Mean", color=color_map["Mean"], linewidth=3)

    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            markersize=10,
            label=chemical,
        )
        for chemical, color in color_map.items()
    ]

    axs[1].legend(handles=legend_elements, title="Chemicals")

    axs[0].set_title("Survival Curves with Bands")
    axs[0].set_xlabel("LC")
    axs[0].set_ylabel("Survival Rate")
    axs[0].set_xscale("log")

    axs[1].set_title("Stress Curves with Bands")
    axs[1].set_xlabel("LC")
    axs[1].set_ylabel("Stress")
    axs[1].set_xscale("log")

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir4imgs", type=str)
    args = parser.parse_args()
    dir4imgs = Path(args.dir4imgs)
    dir4imgs.mkdir(exist_ok=True, parents=True)

    df = create_dose_response_fits_frame()
    color_map = create_color_map(df)

    cleaned_interactive = make_fig("cleaned_curves", "cleaned_stress", df, color_map)
    cleaned_interactive.write_html(dir4imgs / "cleaned_dosecurves.html")
    raw_interactive = make_fig("normed_curves", "stress", df, color_map)
    raw_interactive.write_html(dir4imgs / "raw_dosecurves.html")

    cleaned_static = graphic("cleaned_curves", "cleaned_stress", df, color_map)
    cleaned_static_path = dir4imgs / "cleaned_dosecurves.png"
    cleaned_static.savefig(cleaned_static_path, format="png")
    plt.close(cleaned_static)

    raw_static = graphic("normed_curves", "stress", df, color_map)
    raw_static_path = dir4imgs / "raw_dosecurves.png"
    raw_static.savefig(raw_static_path, format="png")
    plt.close(raw_static)
