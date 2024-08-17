import glob
from stress_addition_model import sam_prediction, Predicted_LCs, get_sam_lcs, SAM_Setting
from helpers import compute_lc, find_lc_99_max, compute_lc_from_curve
from plotting import plot_sam_prediction
from data_formats import ExperimentData, read_data
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

PLOT = True

additional_stress_calc =  ["div","substract", "stress_add"] # "substract"
param_d_norm = [True, False]


def compute_variations(main_series, stress_series, meta):
    main_fit, stress_fit, sam_sur, sam_stress = sam_prediction(
            main_series, stress_series, meta
        )
    
    target = stress_fit
    
    results = {} 
    
    for stress_form in additional_stress_calc:
        for d_norm in param_d_norm:
            settings = SAM_Setting(stress_form=stress_form, param_d_norm=d_norm)
            
            main_fit, stress_fit, sam_sur, sam_stress = sam_prediction(
                main_series, stress_series, meta, settings=settings
            )
            
            results[(stress_form, d_norm)] = sam_sur
            
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
            
            for (stress_form, d_norm), sam_sur in results.items():
                plt.plot(x, sam_sur, label=f"{stress_form}-{d_norm}")
                
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
        
        for (stress_form, d_norm), sam_sur in results.items():
            row[f"{stress_form}-{d_norm}"] = mean_squared_error(target.survival_curve / data.meta.max_survival, sam_sur / data.meta.max_survival)
            
        rows.append(row)
        
        row = {
            "path": path,
            "stressor": name,
            "metric": "r2",
        }
        
        for (stress_form, d_norm), sam_sur in results.items():
            row[f"{stress_form}-{d_norm}"] = r2_score(target.survival_curve, sam_sur)
            
        rows.append(row)
        
        
        
df = pd.DataFrame(rows)

mse = df.query("metric == 'mse'").iloc[:, 3:]
r2 = df.query("metric == 'r2'").iloc[:, 3:]

print(pd.concat([mse.mean(), r2.mean()], axis=1, keys=["mse", "r2"]))

df.to_csv("sam_variations.csv")