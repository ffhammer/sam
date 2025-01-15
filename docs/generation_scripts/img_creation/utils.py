import os
from pathlib import Path

import numpy as np
import pandas as pd

from sam.data_formats import ExperimentData, load_files
from sam.concentration_response_fits import (
    CRF_Settings,
    ConcentrationResponsePrediction,
    concentration_response_fit,
    survival_to_stress,
)
from sam.helpers import (
    detect_hormesis_index,
    pad_c0,
    weibull_2param_inverse,
)
from sam.hormesis_free_response_fitting import fit_hormesis_free_response

os.chdir(os.environ["SAM_REPO_PATH"])


def predict_cleaned_curv(data: ExperimentData):
    """ "Predicts the cleaned Curv"""

    (
        concentrations,
        survival_rate,
        tox_survival,
        cleaned_tox_func,
        hormesis_index,
        tox_fit_params,
    ) = fit_hormesis_free_response(
        data=data.main_series,
        max_survival=data.meta.max_survival,
        hormesis_index=data.hormesis_index or 1,
        interpolate=True,
    )

    def inverse(x):
        return weibull_2param_inverse(x, b=tox_fit_params.b, e=tox_fit_params.e)

    return cleaned_tox_func, inverse


def create_dose_response_fits_frame() -> pd.DataFrame:
    dfs = []

    for path, data in load_files():
        if data.main_series.survival_rate[-1] != 0:
            continue

        meta = data.meta
        res: ConcentrationResponsePrediction = concentration_response_fit(
            data.main_series,
            CRF_Settings(param_d_norm=True, max_survival=meta.max_survival),
        )

        cleaned_func, inverse = predict_cleaned_curv(data)

        def find_lc(lc):
            lc = 1 - lc / 100
            return inverse(lc)

        lc1 = find_lc(1)
        lc99 = find_lc(99)
        dfs.append(
            {
                "og_conc": res.concentration,
                "og_surv": cleaned_func(res.concentration),
                "title": os.path.split(path[:-5])[1],
                "chemical": meta.main_stressor,
                "Organism": meta.organism,
                "model": res,
                "cleaned_func": cleaned_func,
                "lc1": lc1,
                "lc99": lc99,
                "Name": meta.title,
                "Duration": int(meta.days),
                "Experiment": Path(meta.path).parent.name,
            }
        )

    df = pd.DataFrame(dfs)

    def compute_normalised_curve(model: ConcentrationResponsePrediction):
        if np.isnan(model.lc1):
            print("nan")
            model.lc1 = 0.0
        x = np.linspace(model.lc1, model.lc99, 1000)

        return model.model(x) * 100

    def compute_cleaned_curve(row):
        x = np.linspace(row.lc1, row.lc99, 1000)

        return row.cleaned_func(x) * 100

    df["normed_curves"] = df.model.apply(compute_normalised_curve)
    df["stress"] = df.normed_curves.apply(lambda x: survival_to_stress(x / 100))

    df["cleaned_curves"] = df.apply(compute_cleaned_curve, axis=1)
    df["cleaned_stress"] = df.cleaned_curves.apply(
        lambda x: survival_to_stress(x / 100)
    )

    return df
