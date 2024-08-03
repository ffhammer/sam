import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from model import *
import glob


def plot_diffs(pred, r_pred, save_to):

    plt.figure(figsize=(10, 3))

    def create_comparison_plot(x, y, r_label, python_label, x_label, y_label):

        plt.plot(r_pred[x], r_pred[y], label=f"R", color="blue")

        plt.plot(pred[x], pred[y], label=f"Python", color="green")

        # Add labels and title
        plt.xlabel(x_label)
        plt.title(y_label)
        plt.grid(True)

    ax1 = plt.subplot(1, 3, 1)
    create_comparison_plot(
        x="concentration_for_plots",
        y="survival_tox",
        r_label="Survival Tox",
        python_label="Survival Tox",
        x_label="Concentration",
        y_label="Survival Tox",
    )

    ax2 = plt.subplot(1, 3, 2)
    create_comparison_plot(
        x="concentration_for_plots",
        y="survival_tox_LL5",
        r_label="Survival Tox LL5",
        python_label="Survival Tox LL5",
        x_label="Concentration",
        y_label="Survival Tox LL5",
    )

    ax3 = plt.subplot(1, 3, 3)
    create_comparison_plot(
        x="concentration_for_plots",
        y="stress_tox",
        r_label="Stress Tox",
        python_label="Stress Tox",
        x_label="Concentration",
        y_label="Stress Tox",
    )
    plt.legend()

    plt.tight_layout()

    plt.savefig(save_to)


paths = os.listdir("r_preds")

for path in paths:

    try:
        
        df = pd.read_csv(f"formatted_data/{path}")

        model = ecxsys(
            df.conc.values, df.hormesis_concentration.iloc[0], df["no stress"].values
        )

        curves = model["curves"]

        pred = (
            pd.DataFrame(curves)
            .rename(
                columns={
                    "survival_LL5": "survival_tox_LL5",
                    "stress": "stress_tox",
                    "survival": "survival_tox",
                }
            )
            .drop(columns="concentration")
        )

        r_pred = pd.read_csv(os.path.join("r_preds", path))

        save_to = f"migration_validation/{path.replace('csv','png')}"

        plot_diffs(pred, r_pred, save_to)

    except Exception as e:
        print(path, e)