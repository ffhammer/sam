import os

os.chdir(os.environ["SAM_REPO_PATH"])
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import brentq
from tqdm import tqdm

from sam.data_formats import load_datapoints, load_files, read_data
from sam.dose_reponse_fit import DRF_Settings, dose_response_fit, survival_to_stress
from sam.helpers import compute_lc
from sam.stress_addition_model import (
    STANDARD_SAM_SETTING,
    get_sam_lcs,
    sam_prediction,
    stress_to_survival,
    survival_to_stress,
)
import sys

sys.path.append("docs/generation_scripts/")

from img_creation.utils import predict_cleaned_curv

STRESSES = np.linspace(0, 0.6, 100)


def gen_func(stress, cleaned_func):
    def func(x, stress):
        y = cleaned_func(x)

        stress = survival_to_stress(y) + stress

        return stress_to_survival(stress)

    return np.vectorize(lambda x: func(x, stress=stress))


def find_lc_brentq(func, lc, min_v=1e-8, max_v=100000):
    left_val = func(min_v)
    lc = (100 - lc) / 100 * left_val

    brent_func = lambda x: func(x) - lc

    return brentq(brent_func, min_v, max_v)


def compute_lc_trajectory(path: str):
    data = read_data(path)

    cfg = DRF_Settings(
        max_survival=data.meta.max_survival,
        param_d_norm=True,
    )

    fit = dose_response_fit(data.main_series, cfg)
    cleaned_func, hormesis_index, popt = predict_cleaned_curv(data)

    x = fit.concentrations

    lcs = []

    for stress in STRESSES:
        func = gen_func(stress, cleaned_func=fit.model)

        lcs.append(
            (
                find_lc_brentq(func, 10, max_v=x.max()),
                find_lc_brentq(func, 50, max_v=x.max()),
            )
        )

    return np.array(lcs)


def calculate_lc_trajectories() -> tuple[np.ndarray, np.ndarray]:
    results = {}

    for path, _ in tqdm(
        load_files(), desc="Computing increase in LCs. Can take a while"
    ):
        results[path] = compute_lc_trajectory(path)

    lc10 = np.array([i[:, 0] for i in results.values()])
    lc50 = np.array([i[:, 1] for i in results.values()])
    return lc10, lc50


def gen_dose_response_frame(lc10, lc50) -> pd.DataFrame:
    lc_10_frac = lc10[:, 0][:, None] / lc10
    lc_50_frac = lc50[:, 0][:, None] / lc50

    df = []
    for _, data in load_files():
        meta = data.meta
        df.append(
            {
                "Name": meta.title,
                "Chemical": meta.main_stressor,
                "Organism": meta.organism,
                "Experiment": Path(data.meta.path).parent.name,
                "Duration": int(meta.days),
            }
        )
    df = pd.DataFrame(df)
    df["lc_10_frac"] = [np.array(a) for a in lc_10_frac]
    df["lc_50_frac"] = [np.array(a) for a in lc_50_frac]
    return df


def gen_experiment_res_frame(lc10, lc50):
    dfs = []

    for path, data, stress_name, stress_series in load_datapoints():
        meta = data.meta

        res = sam_prediction(
            data.main_series,
            stress_series,
            data.meta,
            settings=STANDARD_SAM_SETTING,
        )

        lcs = get_sam_lcs(
            stress_fit=res.stressor_fit,
            sam_sur=res.predicted_survival_curve,
            max_survival=data.meta.max_survival,
        )

        main_lc10 = compute_lc(optim_param=res.main_fit.optim_param, lc=10)
        main_lc50 = compute_lc(optim_param=res.main_fit.optim_param, lc=50)

        dfs.append(
            {
                "title": path[:-4],
                "days": meta.days,
                "chemical": meta.main_stressor,
                "organism": meta.organism,
                "main_fit": res.main_fit,
                "stress_fit": res.stressor_fit,
                "stress_name": stress_name,
                "main_lc10": main_lc10,
                "main_lc50": main_lc50,
                "stress_lc10": lcs.stress_lc10,
                "stress_lc50": lcs.stress_lc50,
                "sam_lc10": lcs.sam_lc10,
                "sam_lc50": lcs.sam_lc50,
                "experiment_name": Path(data.meta.path).parent.name,
                "Name": data.meta.title,
            }
        )

    df = pd.DataFrame(dfs)
    df["true_10_frac"] = df.main_lc10 / df.stress_lc10
    df["true_50_frac"] = df.main_lc50 / df.stress_lc50
    df["sam_10_frac"] = df.main_lc10 / df.sam_lc10
    df["sam_50_frac"] = df.main_lc50 / df.sam_lc50
    df["stress_level"] = df.stress_fit.apply(
        lambda x: survival_to_stress(x.optim_param["d"])
    )
    return df


def gen_mean_curves(lc10, lc50):
    log_lc10 = [np.log(ar[0] / ar) for ar in lc10]
    log_lc50 = [np.log(ar[0] / ar) for ar in lc50]

    # Calculate the mean and std in the log-space
    log_mean_curve_10 = np.mean(log_lc10, axis=0)
    log_std_curve_10 = np.std(log_lc10, axis=0)

    log_mean_curve_50 = np.mean(log_lc50, axis=0)
    log_std_curve_50 = np.std(log_lc50, axis=0)

    # Exponentiate back to the original scale
    mean_curve_10 = np.exp(log_mean_curve_10)
    upper_curve_10 = np.exp(log_mean_curve_10 + log_std_curve_10)
    lower_curve_10 = np.exp(log_mean_curve_10 - log_std_curve_10)

    mean_curve_50 = np.exp(log_mean_curve_50)
    upper_curve_50 = np.exp(log_mean_curve_50 + log_std_curve_50)
    lower_curve_50 = np.exp(log_mean_curve_50 - log_std_curve_50)

    return (mean_curve_10, lower_curve_10, upper_curve_10), (
        mean_curve_50,
        lower_curve_50,
        upper_curve_50,
    )
