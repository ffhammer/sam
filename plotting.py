import matplotlib.pyplot as plt
import numpy as np
from model import ModelPredictions, ModelInputs

def plot_complete(model: ModelPredictions, title=None):
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))

    plot_survival(model, ax=axs[0, 0], xscale="linear", show_legend=False, xlab=None, ylab="Survival", title="Survival")
    plot_stress(model, ax=axs[0, 1], xscale="linear", show_legend=False, xlab=None, ylab="Stress", title="Stress")
    plot_survival(model, ax=axs[1, 0], xscale="log", show_legend=False, xlab="Concentration", ylab="Survival", title=None)
    plot_stress(model, ax=axs[1, 1], xscale="log", show_legend=False, xlab="Concentration", ylab="Stress", title=None)
    
    if title:
        fig.suptitle(title)
    
    plt.tight_layout()
    return fig

def plot_stress(model: ModelPredictions, ax, xscale="linear", show_legend=False, xlab="Concentration", ylab="Stress", title=None):
    ax.set_xscale(xscale)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    if title:
        ax.set_title(title)

    ax.plot(model.concentration_curve, model.stress_curve, color="deepskyblue", linestyle='--')

    if show_legend:
        ax.legend()

def plot_survival(model: ModelPredictions, ax, xscale="linear", show_legend=False, xlab="Concentration", ylab="Survival", title=None):
    ax.set_xscale(xscale)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    if title:
        ax.set_title(title)
    
    inputs = model.inputs
    
    if inputs.hormesis_concentration is not None:
        colors = np.where(inputs.concentration == inputs.hormesis_concentration, "red", "blue")
    else:
        colors = np.array(["blue" for _ in range(len(inputs.concentration))])
    
    ax.scatter(inputs.concentration, inputs.survival_observerd, label="Survival_observed", zorder=5, c=colors)

    ax.plot(model.concentration_curve, model.survival_curve, color="deepskyblue", linestyle='--', label="Survival")

    if show_legend:
        ax.legend()
