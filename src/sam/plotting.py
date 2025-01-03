import matplotlib.pyplot as plt
import numpy as np
from .concentration_response_fits import ConcentrationResponsePrediction
from .data_formats import ExperimentData, ExperimentMetaData, CauseEffectData
from typing import TYPE_CHECKING
from typing import Optional
from matplotlib.colors import to_rgb, to_hex
from matplotlib.figure import Figure
from .helpers import Predicted_LCs
from seaborn import color_palette

SCATTER_SIZE = 20


def darken_color(color, amount=0.5):
    try:
        c = to_rgb(color)
        c = [max(0, min(1, 1 - (1 - channel) * (1 - amount))) for channel in c]
        return to_hex(c)
    except ValueError:
        return color


def plot_fit_prediction(model: ConcentrationResponsePrediction, title=None):
    """
    Plots the complete set of survival and stress curves on both linear and logarithmic scales.
    """
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))

    plot_survival(
        model.concentration,
        model.survival,
        ax=axs[0, 0],
        orig_series=model.inputs,
        xscale="linear",
        show_legend=False,
        xlab=None,
        ylab="Survival",
        title="Survival",
    )
    plot_stress(
        model.concentration,
        model.general_stress,
        ax=axs[0, 1],
        xscale="linear",
        show_legend=False,
        xlab=None,
        ylab="Stress",
        title="Stress",
    )
    plot_survival(
        model.concentration,
        model.survival,
        ax=axs[1, 0],
        orig_series=model.inputs,
        xscale="log",
        show_legend=False,
        xlab="Concentration",
        ylab="Survival",
        title=None,
    )
    plot_stress(
        model.concentration,
        model.general_stress,
        ax=axs[1, 1],
        xscale="log",
        show_legend=False,
        xlab="Concentration",
        ylab="Stress",
        title=None,
    )

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    return fig


def plot_stress(
    concentration_curve: np.ndarray,
    stress_curve: np.ndarray,
    ax,
    xscale="linear",
    show_legend=False,
    xlab="Concentration",
    ylab="Stress",
    title=None,
    label=None,
    color="deepskyblue",
):
    """
    Plots the stress curve.

    Args:
        ax (matplotlib.axes.Axes): The axes object on which to plot the stress curve.
        xscale (str, optional): Scale for the x-axis. Defaults to "linear".
        show_legend (bool, optional): Whether to show the legend. Defaults to False.
        xlab (str, optional): Label for the x-axis. Defaults to "Concentration".
        ylab (str, optional): Label for the y-axis. Defaults to "Stress".
        title (str, optional): Title for the plot. Defaults to None.
    """
    ax.set_xscale(xscale)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    if title:
        ax.set_title(title)

    ax.plot(concentration_curve, stress_curve, color=color, linestyle="--", label=label)

    if show_legend:
        ax.legend()


def plot_survival(
    concentration_curve: np.ndarray,
    survival_curve: np.ndarray,
    ax,
    orig_series: Optional[CauseEffectData] = None,
    xscale="linear",
    show_legend=False,
    xlab="Concentration",
    ylab="Survival",
    label=None,
    title=None,
    color="deepskyblue",
    hormesis_index: Optional[int] = None,
):
    """
    Plots the survival curve and observed survival data points.

    Args:
        model (ModelPredictions): The model predictions containing the survival data to be plotted.
        ax (matplotlib.axes.Axes): The axes object on which to plot the survival curve.
        xscale (str, optional): Scale for the x-axis. Defaults to "linear".
        show_legend (bool, optional): Whether to show the legend. Defaults to False.
        xlab (str, optional): Label for the x-axis. Defaults to "Concentration".
        ylab (str, optional): Label for the y-axis. Defaults to "Survival".
        title (str, optional): Title for the plot. Defaults to None.
    """
    ax.set_xscale(xscale)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    if title:
        ax.set_title(title)

    ax.plot(
        concentration_curve,
        survival_curve,
        color=color,
        linestyle="--",
        label=label,
    )

    if orig_series is not None:
        ax.scatter(
            orig_series.concentration,
            orig_series.survival_rate,
            label=label,
            zorder=5,
            s=SCATTER_SIZE,
            color=color,
        )

        if hormesis_index is not None:
            mask = np.arange(len(orig_series.concentration)) != hormesis_index

            ax.scatter(
                orig_series.concentration[mask],
                orig_series.survival_rate[mask],
                label=label,
                zorder=5,
                s=SCATTER_SIZE,
                color=color,
            )

            ax.scatter(
                [orig_series.concentration[hormesis_index]],
                [orig_series.survival_rate[hormesis_index]],
                zorder=5,
                color="red",
                s=SCATTER_SIZE,
            )

        else:
            ax.scatter(
                orig_series.concentration,
                orig_series.survival_rate,
                label=label,
                zorder=5,
                color=color,
                s=SCATTER_SIZE,
            )

    if show_legend:
        ax.legend()


def plot_sam_prediction(
    prediction,
    lcs: Optional[Predicted_LCs] = None,
    survival_max: float = 100,
    title=None,
) -> Figure:
    main_fit: ConcentrationResponsePrediction = prediction.control
    stressor_fit: ConcentrationResponsePrediction = prediction.co_stressor
    predicted_survival_curve = prediction.predicted_survival
    predicted_stress_curve = prediction.predicted_general_stress

    colors = color_palette("tab10", 3)

    stress_label = "Tox + Env"
    tox_label = "Tox"
    sam_label = "SAM"

    to_color = {tox_label: colors[0], stress_label: colors[1], sam_label: colors[2]}

    fig, axs = plt.subplots(1, 2, figsize=(10, 3))

    def third_plot(conc, surv, orig_series, label):
        plot_survival(
            conc,
            surv,
            ax=axs[0],
            orig_series=orig_series,
            xscale="log",
            show_legend=False,
            xlab="Concentration",
            ylab="Survival",
            title="Survival",
            label=label,
            color=to_color[label],
        )

    def fourth_plot(x, y, label):
        plot_stress(
            x,
            y,
            ax=axs[1],
            xscale="log",
            show_legend=True,
            xlab="Concentration",
            ylab="Stress",
            title="Stress",
            color=None,
            label=label,
        )

    def plt_lcs(ax, level: float, surv, label: str):
        level = max(level, main_fit.concentration[0])
        ax.plot(
            [level, level],
            [0, surv * survival_max],
            linestyle="-",
            label=label,
            c=to_color[label],
            alpha=0.7,
        )

    third_plot(
        main_fit.concentration,
        main_fit.survival,
        main_fit.inputs,
        label=tox_label,
    )
    third_plot(
        stressor_fit.concentration,
        stressor_fit.survival,
        stressor_fit.inputs,
        label=stress_label,
    )
    third_plot(
        stressor_fit.concentration, predicted_survival_curve, None, label=sam_label
    )

    fourth_plot(main_fit.concentration, main_fit.general_stress, label=tox_label)
    fourth_plot(
        stressor_fit.concentration, stressor_fit.general_stress, label=stress_label
    )
    fourth_plot(stressor_fit.concentration, predicted_stress_curve, label=sam_label)

    if lcs is not None:
        plt_lcs(
            axs[0], lcs.stress_lc10, 0.9 * stressor_fit.optim_param["d"], stress_label
        )
        plt_lcs(
            axs[0], lcs.stress_lc50, 0.5 * stressor_fit.optim_param["d"], stress_label
        )
        plt_lcs(axs[0], lcs.sam_lc10, 0.9 * stressor_fit.optim_param["d"], sam_label)
        plt_lcs(axs[0], lcs.sam_lc50, 0.5 * stressor_fit.optim_param["d"], sam_label)

    if title is not None:
        plt.suptitle(title)

    plt.tight_layout()

    return fig
