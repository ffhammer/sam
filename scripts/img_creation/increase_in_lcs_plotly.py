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

sys.path.append("./")
from scripts.img_creation.lc_increase_data import (
    calculate_lc_trajectories,
    gen_dose_response_frame,
    gen_experiment_res_frame,
    gen_mean_curves,
    STRESSES,
)


def gen_different_curves_fig(meta_infos, stresses):
    unique_experiments = meta_infos.Chemical.unique()
    palette = sns.color_palette(
        "Set2", len(unique_experiments)
    )  # Use a Seaborn color palette
    color_mapping = dict(zip(unique_experiments, palette.as_hex()))
    color_mapping["Mean"] = "black"
    color_mapping["Mean of All"] = "red"

    name_to_id = []

    fig = make_subplots(rows=1, cols=2)

    def gen_traces(y_key):
        ts = list()
        for _, row in meta_infos.iterrows():
            color = color_mapping[row.Chemical]
            ts.append(
                go.Scatter(
                    x=stresses,
                    y=row[y_key],
                    mode="lines",
                    name=row.Name,
                    line=dict(color=color),
                    hovertext=f"<br><b>Name</b>: {row.Name}<br><b>Experiment</b>: {row.Experiment}<br><b>Duration</b>: {row.Duration}<br><b>Main Stressor</b>: {row.Chemical}<br><b>Organism</b>: {row.Organism}",
                    showlegend=False,
                )
            )
            name_to_id.append(f"scatter_{row.Name}_{y_key}")
        return ts

    for i in gen_traces("lc_10_frac"):
        fig.add_trace(
            i,
            row=1,
            col=1,
        )
    for i in gen_traces("lc_50_frac"):
        fig.add_trace(
            i,
            row=1,
            col=2,
        )

    for chemical, color in color_mapping.items():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color=color),
                name=chemical,  # Legend entry for each chemical
            )
        )
        name_to_id.append(f"color_{chemical}")

    cleaner = {
        "Chemical": "Main Stressor",
        "Experiment": "Experiment",
        "Duration": "Duration",
        "Organism": "Organism",
    }

    def gen_means(y_key):
        mean_curve = np.mean(np.stack(meta_infos[y_key].values), 0)

        yield go.Scatter(
            x=stresses,
            y=mean_curve,
            mode="lines",
            name="Mean",
            line=dict(color=color_mapping["Mean"]),
            showlegend=False,
        )
        name_to_id.append(f"mean_all_{y_key}")

        yield go.Scatter(
            x=stresses,
            y=mean_curve,
            mode="lines",
            name="Mean of All",
            line=dict(color=color_mapping["Mean of All"]),
            showlegend=False,
        )
        name_to_id.append(f"mean_reference_{y_key}")

        for key, label_name in cleaner.items():
            for val, df in meta_infos.groupby(key):
                mean_curve = np.mean(np.stack(df[y_key].values), 0)

                yield go.Scatter(
                    x=stresses,
                    y=mean_curve,
                    mode="lines",
                    name="Mean",
                    line=dict(color=color_mapping["Mean"]),
                    showlegend=False,
                )
                name_to_id.append(f"mean_{label_name} = {val}_{y_key}")

    for i in gen_means("lc_10_frac"):
        fig.add_trace(
            i,
            row=1,
            col=1,
        )
    for i in gen_means("lc_50_frac"):
        fig.add_trace(
            i,
            row=1,
            col=2,
        )

    fig.update_xaxes(title_text="Environmental Stress", row=1, col=1)
    fig.update_yaxes(
        title_text="Increase of Toxicant Sensitivity", type="log", row=1, col=1
    )
    fig.update_xaxes(title_text="Environmental Stress", row=1, col=2)
    fig.update_yaxes(
        title_text="Increase of Toxicant Sensitivity", type="log", row=1, col=2
    )

    def get_visible(df, name):
        valid = set()
        for n in df.Name.values:
            valid.add(f"scatter_{n}_lc_10_frac")
            valid.add(f"scatter_{n}_lc_50_frac")

        for chem in df.Chemical.unique():
            valid.add(f"color_{chem}")

        valid.add(f"color_Mean")
        valid.add(f"scatter_{n}_lc_10_frac")
        valid.add(f"scatter_{n}_lc_50_frac")
        if len(df) > 1:
            valid.add(f"mean_{name}_lc_10_frac")
            valid.add(f"mean_{name}_lc_50_frac")

        if name != "all":
            valid.add(f"color_Mean of All")
            valid.add(f"mean_reference_lc_10_frac")
            valid.add(f"mean_reference_lc_50_frac")

        for v in valid:
            assert v in name_to_id, f"{v} wrong!"

        return [i in valid for i in name_to_id]

    buttons = [
        dict(
            label="All",
            method="update",
            args=[{"visible": get_visible(meta_infos, "all")}],
        )
    ]
    assert len(get_visible(meta_infos, "all")) == len(fig.data)

    for key, label_name in cleaner.items():
        for val, df in meta_infos.groupby(key):
            name = f"{label_name} = {val}"

            buttons.append(
                dict(
                    label=name,
                    method="update",
                    args=[{"visible": get_visible(df, name)}],
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
    return fig


def gen_dot_plotly(cleaned, curves, stresses, meta_infos):
    (
        (mean_curve_10, lower_curve_10, upper_curve_10),
        (
            mean_curve_50,
            lower_curve_50,
            upper_curve_50,
        ),
    ) = curves

    fig = make_subplots(rows=1, cols=2)

    color_mapping = {
        "Mean": "orange",
        "Log Std Dev": "gray",
        "This Mean": "black",
        "Measurements": "blue",
        "Predictions": "green",
    }

    name_to_id = []

    def add(mean_curve, upper, lower, col):
        fig.add_trace(
            go.Scatter(
                x=stresses.tolist() + stresses[::-1].tolist(),
                y=upper.tolist() + lower[::-1].tolist(),
                fill="toself",
                fillcolor=color_mapping["Log Std Dev"],
                opacity=0.3,
                line=dict(color="gray"),
                name="Log Std Dev",
                showlegend=False,
            ),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=stresses,
                y=mean_curve,
                mode="lines",
                line=dict(color=color_mapping["Mean"]),
                name="SAM",
                showlegend=False,
            ),
            row=1,
            col=col,
        )

    add(mean_curve_10, lower_curve_10, upper_curve_10, 1)
    name_to_id.append("band_10")
    name_to_id.append("mean_10")
    add(mean_curve_50, lower_curve_50, upper_curve_50, 2)
    name_to_id.append("band_50")
    name_to_id.append("mean_50")

    def add_points(y_col, col):
        for _, row in cleaned.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=(row.stress_level,),
                    y=(row[y_col],),
                    mode="markers",
                    name=row.Name,
                    hovertext=f"<br><b>Name</b>: {row.Name}<br><b>Experiment</b>: {row.experiment_name} <br><b>Main Stressor</b>: {row.chemical}<br><b>Additional Stressor</b>: {row.stress_name}<br> <b>Duration</b>: {row.days}<br><b>Organism</b>: {row.organism}",
                    showlegend=False,
                    line=dict(color=color_mapping["Measurements"]),
                ),
                row=1,
                col=col,
            )
            name_to_id.append(f"{row.Name}_{y_col}")

            fig.add_trace(
                go.Scatter(
                    x=(row.stress_level,),
                    y=(row[y_col.replace("true", "sam")],),
                    mode="markers",
                    name=row.Name,
                    hovertext=f"<br><b>Name</b>: {row.Name}<br><b>Experiment</b>: {row.experiment_name} <br><b>Main Stressor</b>: {row.chemical}<br><b>Additional Stressor</b>: {row.stress_name}<br> <b>Duration</b>: {row.days}<br><b>Organism</b>: {row.organism}",
                    showlegend=False,
                    line=dict(color=color_mapping["Predictions"]),
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

    def add_specific_means(y_col, col):
        for key, label_name in cleaner.items():
            for val, frame in cleaned.groupby(key):
                name = f"{label_name} = {val}_{y_col}"

                names = set(frame.Name.unique())
                spec_col = "lc_10_frac" if "10" in y_col else "lc_50_frac"
                mean_curve = np.mean(
                    np.stack(meta_infos.query("Name in @names")[spec_col].values, 1),
                    axis=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=stresses,
                        y=mean_curve,
                        mode="lines",
                        line=dict(color=color_mapping["This Mean"]),
                        name=f"{label_name} = {val} Mean",
                        hovertext=f"{label_name} = {val} Mean",
                        showlegend=False,
                    ),
                    row=1,
                    col=col,
                )
                name_to_id.append(name)

    add_specific_means("true_10_frac", 1)
    add_specific_means("true_50_frac", 2)

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
            "mean_10",
            "band_10",
            "mean_50",
            "band_50",
            "color_Mean",
            "color_Measurements",
            "color_Log Std Dev",
            "color_Predictions",
        }

        for n in df.Name.values:
            valid.add(f"{n}_true_10_frac")
            valid.add(f"{n}_true_50_frac")

        if name != "all":
            valid.add(f"color_This Mean")
            valid.add(f"{name}_true_10_frac")
            valid.add(f"{name}_true_50_frac")

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
        title_text="Increase of Toxicant Sensitivity", type="log", row=1, col=1
    )
    fig.update_xaxes(title_text="Environmental Stress", row=1, col=2)
    fig.update_yaxes(
        title_text="Increase of Toxicant Sensitivity", type="log", row=1, col=2
    )
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir4imgs", type=str)
    args = parser.parse_args()
    dir4imgs = Path(args.dir4imgs)
    dir4imgs.mkdir(exist_ok=True, parents=True)

    lc10, lc50 = calculate_lc_trajectories()

    meta_infos = gen_dose_response_frame(lc10, lc50)
    cleaned_frame = gen_experiment_res_frame(lc10, lc50)
    mean_curvis = gen_mean_curves(lc10, lc50)

    different_curves_fig = gen_different_curves_fig(meta_infos, STRESSES)
    different_curves_fig.write_html(dir4imgs / "different_curves.html")

    dot_fig = gen_dot_plotly(cleaned_frame, mean_curvis, STRESSES, meta_infos)
    dot_fig.write_html(
        dir4imgs / "lcs.html",
    )
