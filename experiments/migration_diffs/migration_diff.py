import glob
from stress_addition_model import sam_prediction, Predicted_LCs, get_sam_lcs, SAM_Setting, NEW_STANDARD, OLD_STANDARD
from helpers import compute_lc, find_lc_99_max, compute_lc_from_curve
from plotting import plot_sam_prediction
from data_formats import ExperimentData, read_data
import os
import matplotlib.pyplot as plt
import pandas as pd



def compute_preds(settings : SAM_Setting, dir : str) -> pd.DataFrame:

    dir_path = os.path.join("migration", dir)
    os.makedirs(dir_path, exist_ok=True)

    clean_path = lambda x: os.path.basename(x).split(".")[0]
    for path in glob.glob("data/*.xlsx"):
        
        

        data: ExperimentData = read_data(path)

        for name, val in data.additional_stress.items():

            main_fit, stress_fit, sam_sur, sam_stress, additional_stress = sam_prediction(
                data.main_series, val, data.meta, settings=settings,
            )

            row = {
                "Concentration": main_fit.concentration_curve,
                "Survival_A": main_fit.survival_curve,
                "Survival_B" : stress_fit.survival_curve,
                "SAM": sam_sur,
                "Stress_A": main_fit.stress_curve,
                "Stress_B": stress_fit.stress_curve,
            }
            df = pd.DataFrame(row)
            df["exp_name"] = name
            df["path"] = path
            save_to = f"{dir_path}/{clean_path(path)}_{name}.csv"
            df.to_csv(save_to)
        
    return df


new = compute_preds(NEW_STANDARD, "new_standard")
old = compute_preds(OLD_STANDARD, "old_standard")