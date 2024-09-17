from sam.stress_addition_model import (
    sam_prediction,
    get_sam_lcs,
    OLD_STANDARD
)
from sam.plotting import plot_sam_prediction
from sam.data_formats import load_datapoints
import os
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from tqdm import tqdm

PLOT_PATH = "control_imgs/sam_prediction"
SETTINGS = OLD_STANDARD

def compute_all(plot : bool):
    
    rows = []
    
    for path, data, name, val in tqdm(load_datapoints()):
    
        main_fit, stress_fit, sam_sur, sam_stress, additional_stress = sam_prediction(
            data.main_series,
            val,
            data.meta,
            settings=SETTINGS,
        )

        lcs = get_sam_lcs(stress_fit=stress_fit, sam_sur=sam_sur, meta=data.meta)

        if plot:

            # title = f"Fitted LC10: {lcs.stress_lc10 :.2f} LC50: {lcs.stress_lc50 :.2f} - SAM Predictions LC10: {lcs.sam_lc10 :.2f} LC50: {lcs.sam_lc50 :.2f}"
            title = None
            fig = plot_sam_prediction(
                main_fit,
                stress_fit,
                sam_sur,
                sam_stress,
                survival_max=data.meta.max_survival,
                lcs=lcs,
                title=title,
            )
            name = os.path.split(path)[1].replace(".xlsx", f"_{name}.png")
            save_path = os.path.join(PLOT_PATH, name)

            fig.savefig(save_path)
            plt.close()

        row = {
            "path": path,
            "stressor": name,
            "stress_lc10": lcs.stress_lc10,
            "stress_lc50": lcs.stress_lc50,
            "sam_lc10": lcs.sam_lc10,
            "sam_lc50": lcs.sam_lc50,
            "survival_max": data.meta.max_survival,
        }

        rows.append(row)

    df = pd.DataFrame(rows)
    # df.to_csv("sam_predictions.csv")


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", type=bool, default=False)
    
    args = parser.parse_args()
    
    compute_all(args.plot)

if __name__ == "__main__":
    main()
