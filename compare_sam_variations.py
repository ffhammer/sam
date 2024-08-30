import glob
from stress_addition_model import (
    sam_prediction,
    Predicted_LCs,
    get_sam_lcs,
    SAM_Setting,
    StandardSettings,
    OLD_STANDARD,
    NEW_STANDARD,
)
from helpers import compute_lc, find_lc_99_max, compute_lc_from_curve
from plotting import plot_sam_prediction
from data_formats import ExperimentData, read_data
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

PLOT = True

SETTINGS = {
    "new": NEW_STANDARD,
    "old": OLD_STANDARD,
    "new_wo_max": SAM_Setting(
        beta_p=3.2,
        beta_q=3.2,
        param_d_norm=False,
        stress_form="stress_sub",
        stress_intercept_in_survival=0.9995,
        max_control_survival=1,
    ),
    # "new_wo_add": SAM_Setting(
    #     beta_p=3.2,
    #     beta_q=3.2,
    #     param_d_norm=False,
    #     stress_form="stress_sub",
    #     stress_intercept_in_survival=1,
    #     max_control_survival=0.995,
    # ),
    "new_norm": SAM_Setting(
        beta_p=3.2,
        beta_q=3.2,
        param_d_norm=True,
        stress_form="stress_sub",
        stress_intercept_in_survival=0.9995,
        max_control_survival=1,
    ),
    
}


def compute_variations(main_series, stress_series, meta):
    main_fit, stress_fit, sam_sur, sam_stress, additional_stress = sam_prediction(
        main_series, stress_series, meta
    )

    target = stress_fit

    results = {}

    for name, setting in SETTINGS.items():

        main_fit, stress_fit, sam_sur, sam_stress, additional_stress = sam_prediction(
            main_series, stress_series, meta, settings=setting
        )
        
        lcs = get_sam_lcs(stress_fit=stress_fit, sam_sur=sam_sur, meta=data.meta)

        results[name] = sam_sur, lcs

    return target, results


rows = []

for path in glob.glob("data/*.xlsx"):

    data: ExperimentData = read_data(path)

    for name, val in data.additional_stress.items():

        target, results = compute_variations(data.main_series, val, data.meta)

        if PLOT:

            fig = plt.figure(figsize=(10, 4))

            x = target.concentration_curve

            plt.plot(x, target.survival_curve, label="Stressor")

            for name, (sam_sur, lcs) in results.items():
                plt.plot(x, sam_sur, label=name)

            plt.legend()

            name = os.path.split(path)[1].replace(".xlsx", f"_{name}.png")
            save_path = f"migration/variations/{name}"

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
