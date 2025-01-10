import os

os.chdir(os.environ["SAM_REPO_PATH"])
from sam.concentration_response_fits import (
    concentration_response_fit,
    ConcentrationResponsePrediction,
    CRF_Settings,
)
import matplotlib.pyplot as plt
from sam.data_formats import load_files
from sam.helpers import (
    pad_c0,
    weibull_2param_inverse,
)
from pathlib import Path
from io import BytesIO
from tqdm import tqdm
import argparse
import sys

sys.path.append("docs/generation_scripts/")

from img_creation.utils import predict_cleaned_curv


def cleaned_difference_plots():
    for path, data in tqdm(load_files()):
        meta = data.meta
        res: ConcentrationResponsePrediction = concentration_response_fit(
            data.main_series,
            CRF_Settings(param_d_norm=True, max_survival=meta.max_survival),
        )

        hormesis_index = data.hormesis_index or 1
        cleaned_func, inverse = predict_cleaned_curv(data)

        def find_lc(lc):
            lc = 1 - lc / 100
            return inverse(lc)

        lc1 = find_lc(1)
        lc99 = find_lc(99)

        title = data.meta.title

        color = [
            "blue" if i != hormesis_index else "red"
            for i in range(len(data.main_series.concentration))
        ]

        plt.scatter(
            pad_c0(data.main_series.concentration),
            data.main_series.survival_rate,
            label="orig",
            color=color,
        )
        plt.plot(res.concentration, res.survival, label="Raw")
        plt.plot(
            res.concentration,
            cleaned_func(res.concentration) * meta.max_survival,
            label="Cleaned",
        )
        plt.axvline(lc1, 0, 1, color="red", ls="--")
        plt.axvline(lc99, 0, 1, color="red", ls="--")
        plt.legend()
        plt.title(title)
        plt.xscale("log")

        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close(plt.gcf())
        yield title, buf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir4imgs", type=str)
    args = parser.parse_args()

    dir4imgs = Path(args.dir4imgs)
    dir4imgs.mkdir(exist_ok=True, parents=True)

    for title, buf in cleaned_difference_plots():
        file_path = dir4imgs / f"{title}.png"
        with open(file_path, "wb") as f:
            f.write(buf.getvalue())
