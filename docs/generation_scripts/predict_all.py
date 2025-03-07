import warnings

warnings.filterwarnings(
    "ignore",
    message="It is advised to add a data point where survival_rate becomes 0 at the highest concentration.",
)

import os

os.chdir(os.environ["SAM_REPO_PATH"])
from sam.stress_addition_model import (
    SAMPrediction,
    STANDARD_SAM_SETTING,
)
from sam import CauseEffectData
from sam.data_formats import load_datapoints
import os
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import argparse

SETTINGS = STANDARD_SAM_SETTING


def compute_all(plot: bool, dir4imgs: str):
    for path, data, name, val in tqdm(load_datapoints()):
        original_max_surv = data.meta.max_survival

        res = SAMPrediction.generate(
            CauseEffectData(
                data.main_series.concentration,
                data.main_series.survival_rate / original_max_surv,
                "main_series",
            ),
            CauseEffectData(
                val.concentration, val.survival_rate / original_max_surv, name
            ),
            data.meta,
            settings=SETTINGS,
            max_survival=1.0,
        )

        if plot:
            # title = f"Fitted LC10: {lcs.stress_lc10 :.2f} LC50: {lcs.stress_lc50 :.2f} - SAM Predictions LC10: {lcs.sam_lc10 :.2f} LC50: {lcs.sam_lc50 :.2f}"
            title = None
            fig = res.plot(with_lcs=True, title=title, inlcude_control_addition=True)
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
