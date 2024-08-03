
import matplotlib.pyplot as plt
import numpy as np



def plot_stress(model, xscalse = "linear", show_legend=False, xlab="concentration", ylab="stress", main=None):
    if not isinstance(model, dict):
        raise ValueError("Model must be a dictionary containing necessary data")

    curves = model["curves"]
    point_concentration = np.concatenate(([curves["concentration_for_plots"][0]], model["args"]["concentration"][1:]))

    # Determine ymax for the plot
    ymax = max(curves["stress"].max(), 1)

    plt.figure()
    plt.xscale(xscalse)
    # plt.xlim([curves["concentration_for_plots"][0], curves["concentration_for_plots"][-1]])
    # plt.ylim([0, ymax])
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    if main:
        plt.title(main)

    plt.plot(curves["concentration_for_plots"], curves["stress"], color="deepskyblue", linestyle='--')

    # Add legend if requested
    if show_legend:
        plt.legend()

    plt.show()
    

def plot_survival(model, xscalse = "linear", show_legend=False, xlab="concentration", ylab="survival", main=None):
    if not isinstance(model, dict):
        raise ValueError("Model must be a dictionary containing necessary data")

    curves = model["curves"]
    point_concentration = np.concatenate(([curves["concentration_for_plots"][0]], model["args"]["concentration"][1:]))

    plt.figure()
    plt.xscale(xscalse)
    # plt.xlim([curves["concentration_for_plots"][0], curves["concentration_for_plots"][-1]])
    # plt.ylim([0, model["args"]["survival_max"]])
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    if main:
        plt.title(main)

    # Plot "survival_tox_observed"
    
    
    colors = np.where(point_concentration == model["args"]["hormesis_concentration"], "red", "blue")
    
    
    plt.scatter(point_concentration, model["args"]["survival_tox_observed"], label="survival_tox_observed", zorder=5, c= colors)

    # Plot "survival_tox"
    plt.plot(curves["concentration_for_plots"], curves["survival"], color="deepskyblue", linestyle='--', label="survival")


    # x = np.linspace(0, 1, 1000)

    # plt.plot(x * curves["concentration_for_plots"][-1], (1 - beta.cdf(x, 3.2, 3.2)) * 100)

    # Plot "survival_tox_LL5"
    # plt.plot(curves["concentration_for_plots"], curves["survival_LL5"], color="darkblue", linestyle='-.', label="survival_LL5")

    # Add legend if requested
    if show_legend:
        plt.legend()

    plt.show()
