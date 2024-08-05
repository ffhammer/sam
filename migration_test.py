import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from model import dose_response_fit, ModelPredictions
import glob


def plot_diffs(orig_conc, orig_survival,hormesis_conc, new_hormesis_index, r_pred, res_new, save_to):

    plt.figure(figsize=(10, 6))

    def create_comparison_plot(x, y, x_label, y_label, xscale = "linear"):


        plt.plot(res_new[x], res_new[y], label=f"Python", color="red")
        plt.plot(r_pred[x], r_pred[y], label=f"R", color="blue")
        

        # Add labels and title
        plt.xlabel(x_label)
        plt.title(y_label)
        plt.grid(True)
        plt.xscale(xscale)

    ax1 = plt.subplot(2, 2, 1)
    create_comparison_plot(
        x="concentration_for_plots",
        y="survival_tox",
        x_label="Concentration",
        y_label="Survival Tox",
    )
    
    hormesis_index = np.where(orig_conc == hormesis_conc)[0][0]
    colors = np.where(orig_conc == hormesis_conc, "red", "blue")
    
    plt.scatter(orig_conc, orig_survival, c=colors)

    if hormesis_index != new_hormesis_index:
        print("orig hormesis index", hormesis_index, "new", new_hormesis_index)


    ax2 = plt.subplot(2, 2, 2)
    create_comparison_plot(
        x="concentration_for_plots",
        y="stress_tox",
        x_label="Concentration",
        y_label="Stress Tox",
    )
    
    ax3 = plt.subplot(2, 2, 3)
    create_comparison_plot(
        x="concentration_for_plots",
        y="survival_tox",
        x_label=None,
        y_label=None,
        xscale="log"
    )
    
    hormesis_index = np.where(orig_conc == hormesis_conc)[0][0]
    colors = np.where(orig_conc == hormesis_conc, "red", "blue")
    
    plt.scatter(orig_conc, orig_survival, c=colors)


    ax4 = plt.subplot(2, 2, 4)
    create_comparison_plot(
        x="concentration_for_plots",
        y="stress_tox",
        x_label=None,
        y_label=None,
        xscale="log"
    )
    
    
    
    
    plt.legend()

    plt.tight_layout()

    plt.savefig(save_to)


paths = os.listdir("r_preds")

for path in paths:

    try:
        
        
        df = pd.read_csv(f"formatted_data/{path}")
        
        orig_conc = df.conc.values
        orig_survival = df["no stress"].values
        hormesis = df.hormesis_concentration.iloc[0]
      
        res_new : ModelPredictions = dose_response_fit(orig_conc.astype(np.float64), orig_survival.astype(np.float64), hormesis)

        new_df = pd.DataFrame({
            "concentration_for_plots" : res_new.concentration_curve,
            "stress_tox": res_new.stress_curve,
            "survival_tox" : res_new.survival_curve,
        })



        r_pred = pd.read_csv(os.path.join("r_preds", path))

        save_to = f"migration_validation/{path.replace('csv','png')}"

        plot_diffs(orig_conc, orig_survival, hormesis, res_new.hormesis_index, r_pred, new_df, save_to)

    except Exception as e:
        print(path, e)