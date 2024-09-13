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
from tqdm import tqdm


PLOT = False

SETTINGS = {
    "new": NEW_STANDARD,
    "old": OLD_STANDARD,
    "marco_ohne_div": SAM_Setting(
        beta_p=3.2,
        beta_q=3.2,
        param_d_norm=True,
        stress_form="stress_sub",
        stress_intercept_in_survival=1,
        max_control_survival=1,
    ),
    "marco_ohne_temp": SAM_Setting(
        beta_p=3.2,
        beta_q=3.2,
        param_d_norm=False,
        stress_form="div",
        stress_intercept_in_survival=1,
        max_control_survival=1,
    ),
    # "just_sub": SAM_Setting(
    #     beta_p=3.2,
    #     beta_q=3.2,
    #     param_d_norm=False,
    #     stress_form="stress_sub",
    #     stress_intercept_in_survival=1,
    #     max_control_survival=1,
    # ),
    # "no_intercepts": SAM_Setting(
    #     beta_p=3.2,
    #     beta_q=3.2,
    #     param_d_norm=False,
    #     stress_form="stress_sub",
    #     stress_intercept_in_survival=1,
    #     max_control_survival=0.995,
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

for path in tqdm(glob.glob("data/*.xlsx")):

    data: ExperimentData = read_data(path)

    for name, val in data.additional_stress.items():

        if data.main_series.survival_rate[0] < val.survival_rate[0]:
            continue

        target, results = compute_variations(data.main_series, val, data.meta)

        if PLOT:

            fig = plt.figure(figsize=(10, 4))

            x = target.concentration_curve

            plt.plot(x, target.survival_curve, label="Stressor")

            for res_name, (sam_sur, lcs) in results.items():
                plt.plot(x, sam_sur, label=res_name)

            plt.legend()

            name = os.path.split(path)[1].replace(".xlsx", f"_{name}.png")
            save_path = f"migration/variations_diff/{name}"

            fig.savefig(save_path)
            plt.close()

        row = {
            "path": path,
            "stressor": name,
            "metric" : "r2"
        }

        for name, (sam_sur, lcs) in results.items():
            row[name] = r2_score(target.survival_curve, sam_sur)

        rows.append(row)
        
        
        row = {
            "path": path,
            "stressor": name,
            "metric" : "r2_50"
        }

        for name, (sam_sur, lcs) in results.items():
            l = int(len(target.survival_curve) * 0.50)
            row[name] = r2_score(target.survival_curve[:l], sam_sur[:l])


        rows.append(row)


        row = {
            "path": path,
            "stressor": name,
            "metric" : "r2_50r"
        }

        for name, (sam_sur, lcs) in results.items():
            l = int(len(target.survival_curve) * 0.5)
            row[name] = r2_score(target.survival_curve[l:], sam_sur[l:])


        rows.append(row)

        
        

        row = {
            "path": path,
            "stressor": name,
            "metric" : "diff"
        }

        for name, (sam_sur, lcs) in results.items():
            row[name] = (target.survival_curve -sam_sur).astype(str).tolist()

        rows.append(row)



df = pd.DataFrame(rows)

# print(df.iloc[:,4:].mean())

df.to_csv("was_ist_das_problem.csv")


