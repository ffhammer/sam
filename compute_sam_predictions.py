import glob
from stress_addition_model import sam_prediction, Predicted_LCs, get_sam_lcs
from helpers import compute_lc, find_lc_99_max, compute_lc_from_curve
from plotting import plot_sam_prediction
from data_formats import ExperimentData, read_data
import os
import matplotlib.pyplot as plt
import pandas as pd

PLOT = False

rows = []

for path in glob.glob("data/*.xlsx"):

    data: ExperimentData = read_data(path)

    for name, val in data.additional_stress.items():

        main_fit, stress_fit, sam_sur, sam_stress = sam_prediction(
            data.main_series, val, data.meta
        )

        lcs = get_sam_lcs(stress_fit=stress_fit, sam_sur=sam_sur, meta=data.meta)

        if PLOT:

            title = f"Fitted LC10: {lcs.stress_lc10 :.2f} LC50: {lcs.stress_lc50 :.2f} - SAM Predictions LC10: {lcs.sam_lc10 :.2f} LC50: {lcs.sam_lc50 :.2f}"
            fig = plot_sam_prediction(
                main_fit, stress_fit, sam_sur, sam_stress, title=title
            )
            name = os.path.split(path)[1].replace(".xlsx", f"_{name}.png")
            save_path = f"sam_plots/{name}"

            fig.savefig(save_path)
            plt.close()

        row = {
            "path": path,
            "stressor": name,
            "stress_lc10": lcs.stress_lc10,
            "stress_lc50": lcs.stress_lc50,
            "sam_lc10": lcs.sam_lc10,
            "sam_lc50": lcs.sam_lc50,
            "survival_max" : data.meta.max_survival
        }
        
        rows.append(row)
        
df = pd.DataFrame(rows)
df.to_csv("sam_predictions.csv")