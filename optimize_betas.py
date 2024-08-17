import glob
from stress_addition_model import sam_prediction, Predicted_LCs, get_sam_lcs, SAM_Setting
from helpers import compute_lc, find_lc_99_max, compute_lc_from_curve
from plotting import plot_sam_prediction
from data_formats import ExperimentData, read_data
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

PLOT = False

def compute_variations(main_series, stress_series, meta):
    main_fit, stress_fit, sam_sur, sam_stress = sam_prediction(
        main_series, stress_series, meta
    )
    
    target = stress_fit
    
    results = {} 
    
    for beta_q in np.linspace(0.1, 10, 30):
        for beta_p in np.linspace(0.1, 10, 30):
            settings = SAM_Setting(stress_form="stress_add", param_d_norm=False, beta_p=beta_p, beta_q=beta_q)
            
            main_fit, stress_fit, sam_sur, sam_stress = sam_prediction(
                main_series, stress_series, meta, settings=settings
            )
            
            results[(beta_p, beta_q)] = sam_sur
            
    return target, results

def process_file(path):
    rows = []
    data: ExperimentData = read_data(path)
    
    for name, val in data.additional_stress.items():
        target, results = compute_variations(data.main_series, val, data.meta)
       
        if PLOT:
            fig = plt.figure(figsize=(10, 4))
            x = target.concentration_curve
            plt.plot(x, target.survival_curve, label="Stressor")
            
            for (beta_p, beta_q), sam_sur in results.items():
                plt.plot(x, sam_sur, label=f"{beta_p}-{beta_q}")
                
            plt.legend()
            name_fig = os.path.split(path)[1].replace(".xlsx", f"_{name}.png")
            save_path = f"sam_variations/{name_fig}"
            fig.savefig(save_path)
            plt.close()
        
        mse_row = {
            "path": path,
            "stressor": name,
            "metric": "mse",
        }
        
        for (beta_p, beta_q), sam_sur in results.items():
            mse_row[f"{beta_p}-{beta_q}"] = mean_squared_error(target.survival_curve / data.meta.max_survival, sam_sur / data.meta.max_survival)
            
        rows.append(mse_row)
        
        r2_row = {
            "path": path,
            "stressor": name,
            "metric": "r2",
        }
        
        for (beta_p, beta_q), sam_sur in results.items():
            r2_row[f"{beta_p}-{beta_q}"] = r2_score(target.survival_curve, sam_sur)
            
        rows.append(r2_row)
        
    return rows

if __name__ == "__main__":
    paths = glob.glob("data/*.xlsx")
    
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_file, paths), total=len(paths)))
    
    # Flatten the list of lists
    all_rows = [row for result in results for row in result]
    
    df = pd.DataFrame(all_rows)
    
    mse = df.query("metric == 'mse'").iloc[:, 3:]
    r2 = df.query("metric == 'r2'").iloc[:, 3:]
    
    print(pd.concat([mse.mean(), r2.mean()], axis=1, keys=["mse", "r2"]))
    
    df.to_csv("beta_variations_stress_add.csv")
