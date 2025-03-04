import os

from sam.concentration_addition import compute_control_addition_lc

os.chdir(os.environ["SAM_REPO_PATH"])
from pathlib import Path

import pandas as pd

from sam.data_formats import load_datapoints
from sam.helpers import compute_lc
from sam.stress_addition_model import (
    STANDARD_SAM_SETTING,
    SAMPrediction,
)
import sys

sys.path.append("docs/generation_scripts/")


def gen_experiment_res_frame():
    dfs = []

    for path, data, stress_name, stress_series in load_datapoints():
        meta = data.meta

        res = SAMPrediction.generate(
            data.main_series,
            stress_series,
            data.meta,
            settings=STANDARD_SAM_SETTING,
        )

        lcs = res.get_lethal_concentrations()

        main_lc10 = compute_lc(optim_param=res.control.optim_param, lc=10)
        main_lc50 = compute_lc(optim_param=res.control.optim_param, lc=50)

        ca_lc_10 = compute_control_addition_lc(
            control_params=res.control.optim_param,
            co_stressor_params=res.control.optim_param,
            lc=10,
        )
        ca_lc_50 = compute_control_addition_lc(
            control_params=res.control.optim_param,
            co_stressor_params=res.control.optim_param,
            lc=50,
        )

        row = {
            "title": path[:-4],
            "days": meta.days,
            "chemical": meta.main_stressor,
            "organism": meta.organism,
            "main_fit": res.control,
            "stress_fit": res.co_stressor,
            "stress_name": stress_name,
            "main_lc10": main_lc10,
            "main_lc50": main_lc50,
            "stress_lc10": lcs.stress_lc10,
            "stress_lc50": lcs.stress_lc50,
            "sam_lc10": lcs.sam_lc10,
            "sam_lc50": lcs.sam_lc50,
            "ca_lc10": ca_lc_10,
            "ca_lc50": ca_lc_50,
            "experiment_name": Path(data.meta.path).parent.name,
            "Name": data.meta.title,
            "add_stress": res.assumed_additional_stress,
            "control_div": 1
            - (res.co_stressor.optim_param["d"] / res.control.optim_param["d"]),
        }
        row["effect_range"] = compute_lc(
            optim_param=res.control.optim_param, lc=75
        ) / compute_lc(optim_param=res.control.optim_param, lc=25)
        dfs.append(row)

    df = pd.DataFrame(dfs)

    df["true_10_frac"] = df.main_lc10 / df.stress_lc10
    df["true_50_frac"] = df.main_lc50 / df.stress_lc50
    df["sam_10_frac"] = df.main_lc10 / df.sam_lc10
    df["sam_50_frac"] = df.main_lc50 / df.sam_lc50
    df["ca_10_frac"] = df.main_lc10 / df.ca_lc10
    df["ca_50_frac"] = df.main_lc50 / df.ca_lc50

    df["stress_level"] = df.add_stress
    df["e_fac"] = 1.0
    return df
