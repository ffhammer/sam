from sam.stress_addition_model import (
    sam_prediction,
    get_sam_lcs,
    SAM_Settings,
    STANDARD_SAM_SETTING,
    NEW_STANDARD,
)
from sam.data_formats import load_datapoints
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import numpy as np
from scipy.optimize import brentq


def surv_to_stress_new(x0):
    x0 = np.clip(x0, 0, 1 - 1e-9)

    x0 = 1 - x0  # Adjusting the input
    term1 = 0.000995
    numerator = np.log(0.907 * x0)
    denominator = x0 - (1.09 / x0)
    x = term1 + numerator / denominator
    
    x = np.clip(x, 0, 1)
    
    
    return x

MIN_VAL = surv_to_stress_new(1 - 1e-9)

@np.vectorize
def stress_to_surv_new(y):
    y = np.maximum(y, MIN_VAL)


    def equation(x0):
        return surv_to_stress_new(x0) - y

    # Use brentq to find the root of the equation in the interval [0, 1]
    try:
        x0_inverse = brentq(
            equation, 0, 1 - 1e-9
        )  # Start from a small positive value to avoid division by zero
        return x0_inverse
    except ValueError as e:
        raise ValueError(
            f"Cannot find a root in the interval [0, 1] for y = {y}. Error: {e}"
        )



def surv_to_stress_new2(x0):
    x0 = np.clip(x0, 0,1-1e-7)   
    x0 = 1-x0
    pred = 0.000995 + np.log(0.907 * x0) / (x0 - (1.09 / x0))
    
    return np.clip(pred, 0, 1)
    

MIN_VAL = surv_to_stress_new2(1 - 1e-9)

@np.vectorize
def stress_to_surv_new2(y):
    
    y = np.clip(y, MIN_VAL, 1)


    def equation(x0):
        return surv_to_stress_new2(x0) - y

    # Use brentq to find the root of the equation in the interval [0, 1]
    try:
        x0_inverse = brentq(
            equation, 0, 1 - 1e-9
        )  # Start from a small positive value to avoid division by zero
        return x0_inverse
    except ValueError as e:
        raise ValueError(
            f"Cannot find a root in the interval [0, 1] for y = {y}. Error: {e}"
        )


PLOT = True

SETTINGS = {
    "new": NEW_STANDARD,
    "marco": STANDARD_SAM_SETTING,
    # "marco_push": SAM_Setting(
    #     param_d_norm=True,
    #     stress_form="div",
    #     max_control_survival=0.995,
    # ),
    # "learned": SAM_Setting(
    #     param_d_norm=True,
    #     stress_form="div",
    #     stress_intercept_in_survival=1,
    #     max_control_survival=1,
    #     stress_to_survival=stress_to_surv_new,
    #     survival_to_stress=surv_to_stress_new,
    # ),
    # "learned_sub": SAM_Setting(
    #     param_d_norm=False,
    #     stress_form="stress_sub",
    #     stress_intercept_in_survival=1,
    #     max_control_survival=1,
    #     stress_to_survival=stress_to_surv_new2,
    #     survival_to_stress=surv_to_stress_new2,
    # ),
}


def compute_variations(main_series, stress_series, meta):
    results = {}

    for name, setting in SETTINGS.items():

        main_fit, stress_fit, sam_sur, sam_stress, additional_stress = sam_prediction(
            main_series, stress_series, meta, settings=setting
        )

        lcs = get_sam_lcs(stress_fit=stress_fit, sam_sur=sam_sur, meta=data.meta)

        results[name] = sam_sur, lcs

    return stress_fit, results


rows = []

for path, data, name, val in tqdm(load_datapoints()):


    if data.main_series.survival_rate[0] < val.survival_rate[0]:
        continue

    target, results = compute_variations(data.main_series, val, data.meta)

    if PLOT:

        fig = plt.figure(figsize=(10, 4))

        x = target.concentrations

        plt.plot(x, target.survival_curve, label="Stressor")

        for res_name, (sam_sur, lcs) in results.items():
            plt.plot(x, sam_sur, label=res_name)

        plt.legend()

        name = os.path.split(path)[1].replace(".xlsx", f"_{name}.png")
        save_path = f"migration/variations/{name}"
        plt.xscale("log")
        plt.tight_layout()
        fig.savefig(save_path)
        plt.close()

    row = {
        "path": path,
        "stressor": name,
        "metric": "mse",
    }

    for name, (sam_sur, lcs) in results.items():
        row[name] = mean_squared_error(
            target.survival_curve / data.meta.max_survival,
            sam_sur / data.meta.max_survival,
        )

    rows.append(row)

    row = {
        "path": path,
        "stressor": name,
        "metric": "r2",
    }

    for name, (sam_sur, lcs) in results.items():
        row[name] = r2_score(target.survival_curve, sam_sur)

    rows.append(row)


df = pd.DataFrame(rows)

mse = df.query("metric == 'mse'").iloc[:, 3:]
r2 = df.query("metric == 'r2'").iloc[:, 3:]

print(pd.concat([mse.mean(), r2.mean()], axis=1, keys=["mse", "r2"]))

df.to_csv("sam_variations.csv")
