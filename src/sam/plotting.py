import matplotlib.pyplot as plt
import numpy as np
from .dose_reponse_fit import ModelPredictions
from .data_formats import ExperimentData, ExperimentMetaData, DoseResponseSeries
from typing import Optional
from matplotlib.colors import to_rgb, to_hex
from .helpers import Predicted_LCs
from seaborn import color_palette


def darken_color(color, amount=0.5):
    """
    Darkens the given color by multiplying (1-amount) to its RGB values.
    
    Args:
        color (str or tuple): The color to darken.
        amount (float): The amount to darken the color by (0 to 1).
        
    Returns:
        str: The darkened color as a hex string.
    """
    try:
        c = to_rgb(color)
        c = [max(0, min(1, 1 - (1 - channel) * (1 - amount))) for channel in c]
        return to_hex(c)
    except ValueError:
        return color

def plot_fit_prediction(model: ModelPredictions, title=None):
    """
    Plots the complete set of survival and stress curves on both linear and logarithmic scales.

    Args:
        model (ModelPredictions): The model predictions containing the data to be plotted.
        title (str, optional): Title for the entire figure. Defaults to None.

    Returns:
        matplotlib.figure.Figure: The figure object containing the plots.
    """
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))

    plot_survival(
        model.concentration_curve,
        model.survival_curve,
        ax=axs[0, 0],
        orig_series=model.inputs,
        xscale="linear",
        show_legend=False,
        xlab=None,
        ylab="Survival",
        title="Survival",
    )
    plot_stress(
        model.concentration_curve,
        model.stress_curve,
        ax=axs[0, 1],
        xscale="linear",
        show_legend=False,
        xlab=None,
        ylab="Stress",
        title="Stress",
    )
    plot_survival(
        model.concentration_curve,
        model.survival_curve,
        ax=axs[1, 0],
        orig_series=model.inputs,
        xscale="log",
        show_legend=False,
        xlab="Concentration",
        ylab="Survival",
        title=None,
    )
    plot_stress(
        model.concentration_curve,
        model.stress_curve,
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
    label = None,
    color = "deepskyblue",
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

    ax.plot(concentration_curve, stress_curve, color=color, linestyle="--", label = label)

    if show_legend:
        ax.legend()





def plot_survival(
    concentration_curve: np.ndarray,
    survival_curve: np.ndarray,
    ax,
    orig_series: Optional[DoseResponseSeries] = None,
    xscale="linear",
    show_legend=False,
    xlab="Concentration",
    ylab="Survival",
    label = None,
    title=None,
    color = "deepskyblue",
    hormesis_index : Optional[int] = None,
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
            color=color,
        )
        
        if hormesis_index is not None:
            
            mask = np.arange(len(orig_series.concentration)) != hormesis_index
            
            ax.scatter(
                orig_series.concentration[mask],
                orig_series.survival_rate[mask],
                label=label,
                zorder=5,
                color=color,
            )

            ax.scatter(
                [orig_series.concentration[hormesis_index]],
                [orig_series.survival_rate[hormesis_index]],
                zorder=5,
                color="red",
            )

        else:
            ax.scatter(
                orig_series.concentration,
                orig_series.survival_rate,
                label=label,
                zorder=5,
                color=color,
            )


    if show_legend:
        ax.legend()




def plot_sam_prediction(
    main_fit: ModelPredictions,
    stressor_fit: ModelPredictions,
    predicted_survival_curve,
    predicted_stress_curve,
    lcs : Optional[Predicted_LCs] = None,
    survival_max : float = 100,
    title = None
):
    colors = color_palette("tab10", 3)

    stress_label = "Toxicant + " + stressor_fit.inputs.name
    tox_label = "Toxicant"
    sam_label = "SAM"
    
    to_color = {tox_label : colors[0], stress_label : colors[1], sam_label : colors[2]}

    fig, axs = plt.subplots(2, 2, figsize=(10, 6))

    def first_plot(conc, surv, orig_series, label):
        plot_survival(
            conc,
            surv,
            ax=axs[0, 0],
            orig_series=orig_series,
            xscale="linear",
            show_legend=False,
            xlab=None,
            ylab="Survival",
            title="Survival",
            label=label,
            color=to_color[label],
        )

    def second_plot(x, y, label):
        plot_stress(
            x,
            y,
            ax=axs[0, 1],
            xscale="linear",
            show_legend=False,
            xlab=to_color[label],
            ylab="Stress",
            title="Stress",
            color=to_color[label],
            label=label,
        )

    def third_plot(conc, surv, orig_series, label):
        plot_survival(
            conc,
            surv,
            ax=axs[1, 0],
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
            ax=axs[1, 1],
            xscale="log",
            show_legend=True,
            xlab="Concentration",
            ylab="Stress",
            title="Stress",
            color=None,
            label=label,
        )

    def plt_lcs(ax, level : float, surv, label : str):
        level = max(level, main_fit.concentration_curve[0])
        ax.plot([level,level],[0, surv * survival_max],linestyle="-", label =label, c = to_color[label], alpha = 0.7)
        
        
    # Plotting in the correct order
    first_plot(main_fit.concentration_curve, main_fit.survival_curve, main_fit.inputs, label=tox_label)
    first_plot(stressor_fit.concentration_curve, stressor_fit.survival_curve, stressor_fit.inputs, label=stress_label)
    first_plot(stressor_fit.concentration_curve, predicted_survival_curve, None, label=sam_label)
    
    
        

    second_plot(main_fit.concentration_curve, main_fit.stress_curve, label=tox_label)
    second_plot(stressor_fit.concentration_curve, stressor_fit.stress_curve, label=stress_label)
    second_plot(stressor_fit.concentration_curve, predicted_stress_curve, label=sam_label)

    third_plot(main_fit.concentration_curve, main_fit.survival_curve, main_fit.inputs, label=tox_label)
    third_plot(stressor_fit.concentration_curve, stressor_fit.survival_curve, stressor_fit.inputs, label=stress_label)
    third_plot(stressor_fit.concentration_curve, predicted_survival_curve, None, label=sam_label)

    fourth_plot(main_fit.concentration_curve, main_fit.stress_curve, label=tox_label)
    fourth_plot(stressor_fit.concentration_curve, stressor_fit.stress_curve, label=stress_label)
    fourth_plot(stressor_fit.concentration_curve, predicted_stress_curve, label=sam_label)
    
    if lcs is not None:
        plt_lcs(axs[1,0], lcs.stress_lc10, 0.9 * stressor_fit.optim_param["d"], stress_label)
        plt_lcs(axs[1,0], lcs.stress_lc50, 0.5 * stressor_fit.optim_param["d"], stress_label)
        plt_lcs(axs[1,0], lcs.sam_lc10, 0.9 * stressor_fit.optim_param["d"], sam_label)
        plt_lcs(axs[1,0], lcs.sam_lc50, 0.5 * stressor_fit.optim_param["d"], sam_label)

    if title is not None:
        plt.suptitle(title)

    plt.tight_layout()
    
    
    return fig