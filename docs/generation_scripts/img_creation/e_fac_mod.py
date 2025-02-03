import os

os.chdir(os.environ["SAM_REPO_PATH"])
import numpy as np
import sys

sys.path.append("docs/generation_scripts/")


from sam import generate_sam_prediction, read_data, STANDARD_SAM_SETTING, get_sam_lcs
import json
from copy import deepcopy
from sklearn.metrics import r2_score
from scipy.optimize import minimize_scalar
import pandas as pd


def find_optimal_efac(
    control_data,
    co_stressor_data,
    meta,
) -> float:
    def func(e_fac):
        settings = deepcopy(STANDARD_SAM_SETTING)
        settings.e_param_modifier_pre_sam = lambda x: x * e_fac

        res = generate_sam_prediction(
            meta=meta,
            co_stressor_data=co_stressor_data,
            control_data=control_data,
            settings=settings,
        )
        return -r2_score(res.co_stressor.survival, res.predicted_survival)

    result = minimize_scalar(func, bounds=(0.05, 1.0), method="bounded")
    return result.x


def overwrite_examples_with_efac(e_fac: float | str, df: pd.DataFrame) -> pd.DataFrame:
    with open("docs/add_sys_examples.json") as f:
        sys_examples = json.load(f)

    new_df = df.copy()
    for example in sys_examples:
        data = read_data(example[0])
        stress_name = example[1]

        row = df.query("Name == @data.meta.title and stress_name == @stress_name")
        if len(row) == 0:
            print("skipping", example)
            continue
        assert len(row) == 1, f"{row}"
        index = row.index[0]
        row = row.iloc[0]

        settings = deepcopy(STANDARD_SAM_SETTING)

        if e_fac == "optimal":
            optimal_e_fac = find_optimal_efac(
                meta=data.meta,
                co_stressor_data=data.additional_stress[stress_name],
                control_data=data.main_series,
            )
            settings.e_param_modifier_pre_sam = lambda x: x * optimal_e_fac

        elif isinstance(e_fac, float):
            settings.e_param_modifier_pre_sam = lambda x: x * e_fac
        else:
            raise ValueError("wrong e_fac")

        res = generate_sam_prediction(
            meta=data.meta,
            co_stressor_data=data.additional_stress[stress_name],
            control_data=data.main_series,
            settings=settings,
        )

        lcs = get_sam_lcs(
            stress_fit=res.co_stressor,
            sam_sur=res.predicted_survival,
            max_survival=data.meta.max_survival,
        )

        row.sam_lc10 = lcs.sam_lc10
        row.sam_lc50 = lcs.sam_lc50
        row["sam_10_frac"] = row.main_lc10 / row.sam_lc10
        row["sam_50_frac"] = row.main_lc50 / row.sam_lc50
        new_df.loc[index] = row
    # assert np.sum(~(new_df == df).all(1)) == len(sys_examples)
    return new_df
