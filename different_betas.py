import pickle
from dose_reponse_fit import dose_response_fit, ModelPredictions, StandardSettings
# from plotting import plot_complete
import numpy as np
from data_formats import read_data
from stress_survival_conversion import stress_to_survival
from sklearn.metrics import r2_score
import glob
from stress_addition_model import SAM_Setting
from data_formats import ExperimentData, read_data
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
from dataclasses import dataclass
from tqdm import tqdm
from joblib import Parallel, delayed

@dataclass
class Prediction:
    data: ExperimentData
    main_fit: ModelPredictions
    stressor_fit: ModelPredictions
    best_stress: float
    best_pred: np.ndarray[np.float32]
    stressor_name: str
    best_r2 : float
    best_pred_r2 : np.ndarray
    lowest_mse : float

paths = glob.glob("data/*.xlsx")


def compute_preds(settings : SAM_Setting):

    results: list[Prediction] = []

    for p in tqdm(paths, disable=True):
        
        data = read_data(p)
        
        for name, val in data.additional_stress.items():

            main_series, stressor_series, meta = data.main_series, val, data.meta

            dose_cfg = StandardSettings(
                survival_max=meta.max_survival,
                beta_q=settings.beta_q,
                beta_p=settings.beta_p,
                param_d_norm=settings.param_d_norm,
            )

            main_fit = dose_response_fit(main_series, cfg=dose_cfg)


            stressor_fit = dose_response_fit(stressor_series, cfg=dose_cfg)


            goal = stressor_fit.survival_curve

            def mse(y, y_pred):
                return np.square(y - y_pred).sum(axis=-1)

            stresses = np.linspace(0, 0.7, 150)


            preds = stress_to_survival(stresses[:, None]  + main_fit.stress_curve[None,:], p = settings.beta_p, q = settings.beta_q) * meta.max_survival

            losses = mse(goal[None,] / meta.max_survival, preds / meta.max_survival)

            best_stress_arg = np.argmin(losses)
            best_pred = preds[best_stress_arg]

            r2_losses = np.array([r2_score(goal, p) for p in preds])

            best_r2_stress_arg = np.argmax(r2_losses)

            results.append(
                Prediction(
                    data=data,
                    main_fit=main_fit,
                    stressor_fit=stressor_fit,
                    best_stress=stresses[best_stress_arg],
                    best_pred=best_pred,
                    stressor_name=name,
                    best_r2=stresses[best_r2_stress_arg],
                    best_pred_r2=preds[best_r2_stress_arg],
                    lowest_mse=losses.min()
                )
            )
    return results


workers = Parallel(n_jobs=7)

@delayed
def my_function(p):
    res = compute_preds(SAM_Setting(param_d_norm=False, beta_p=p, beta_q=p))
    
    df = pd.DataFrame({
    "d_main" : [r.main_fit.optim_param["d"] for r in res],
    "d_stress" : [r.stressor_fit.optim_param["d"] for r in res],
    "best" : [r.best_stress for r in res],
    "best_r2" : [r.best_r2 for r in res],
    "path" : [r.data.meta.path for r in res],
    "stress" : [r.stressor_name for r in res],
    "chemical" : [r.data.meta.chemical for r in res],
    "lowest_mse" : [r.lowest_mse for r in res],
})

    
    print("done", p)
    return df



params = list(np.arange(2.4,3.7,0.1)) + [4,4.5, 5, 6, 7, 8, 10, 15, 20]

res = workers(my_function(x) for x in params)

for q, df in zip(params, res):
    
    df["params"] = q
    df.to_csv(f"betas/{float(q) :.1f}.csv")
    