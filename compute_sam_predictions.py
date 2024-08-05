import glob
from stress_addition_model import sam_prediction
from helpers import compute_lc, find_lc_99_max, compute_lc_from_curve
from plotting import plot_sam_prediction
from data_formats import ExperimentData, read_data
import os
import matplotlib.pyplot as plt


for path in glob.glob("data/*.xlsx"):

    data : ExperimentData = read_data(path)
    
    for name, val in data.additional_stress.items():
    
        main_fit, stress_fit, sam_sur, sam_stress = sam_prediction(data.main_series, val, data.meta)


        
        max_val = find_lc_99_max(stress_fit.model)
        stress_lc10 = compute_lc(stress_fit.model, 10, 1e-7, max_val)
        stress_lc50 = compute_lc(stress_fit.model, 50, 1e-7, max_val)
        
        sam_lc10 = compute_lc_from_curve(main_fit.concentration_curve, sam_sur, llc=10, survival_max=data.meta.max_survival)
        sam_lc50 = compute_lc_from_curve(main_fit.concentration_curve, sam_sur, llc=50, survival_max=data.meta.max_survival)

        fig = plot_sam_prediction(main_fit, stress_fit, sam_sur, sam_stress, title = f"Fitted LC10: {stress_lc10 :.2f} LC50: {stress_lc50 :.2f} - SAM Predictions LC10: {sam_lc10 :.2f} LC50: {sam_lc50 :.2f}")
        name = os.path.split(path)[1].replace(".xlsx", f"_{name}.png")
        save_path = f"sam_plots/{name}"
        
        fig.savefig(save_path)
        plt.close()

