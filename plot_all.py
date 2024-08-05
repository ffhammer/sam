import os
from dose_reponse_fit import dose_response_fit, ModelPredictions, StandardSettings
from plotting import plot_fit_prediction
import pandas as pd
import numpy as np
import glob
from data_formats import ExperimentData, ExperimentMetaData, DoseResponseSeries, read_data

for path in glob.glob("data/*.xlsx"):

    data : ExperimentData = read_data(path)
    
    
    res: ModelPredictions = dose_response_fit(
        data.main_series, StandardSettings(survival_max=data.meta.max_survival)
    )

    title = f"{data.meta.chemical} - {data.meta.organism}"
    fig = plot_fit_prediction(model=res, title=title)

    save_path = f"python_plots/{os.path.split(path.replace('xlsx', 'png'))[1]}"

    fig.savefig(save_path)
