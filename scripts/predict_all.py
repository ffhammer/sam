from sam import chdir_to_repopath
chdir_to_repopath()
from sam.stress_addition_model import (
    sam_prediction,
    get_sam_lcs,
    STANDARD_SAM_SETTING
)
from sam.plotting import plot_sam_prediction
from sam.data_formats import load_datapoints
import os
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import argparse

SETTINGS = STANDARD_SAM_SETTING

def compute_all(plot : bool, dir4imgs : str):
    
    
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
            name = f"{data.meta.title}_{name}.png"
            save_path = os.path.join(dir4imgs, name)

            fig.savefig(save_path)
            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", type=bool, default=True)
    parser.add_argument("--dir4imgs", type=str, default="control_imgs/sam_prediction")
    args = parser.parse_args()
    os.makedirs(args.dir4imgs, exist_ok=True)
    
    compute_all(args.plot, args.dir4imgs)
