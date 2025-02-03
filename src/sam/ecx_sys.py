from dataclasses import dataclass
from typing import Optional

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from py_lbfgs import lbfgs_fit
from seaborn import color_palette

from .data_formats import CauseEffectData
from .helpers import weibull_2param_inverse, pad_c0, fix_wlb1
from .stress_survival_conversion import stress_to_survival, survival_to_stress
from .hormesis_free_response_fitting import fit_hormesis_free_response
from .plotting import SCATTER_SIZE


def generate_ecx_sys_prediction(
    data: CauseEffectData,
    max_survival: float,
    hormesis_index: Optional[int] = None,
    len_curves: int = 1000,
    beta_q=3.2,
    beta_p=3.2,
    interpolate=True,
):
    (
        concentrations,
        survival_rate,
        tox_survival,
        cleaned_tox_func,
        _,
        tox_fit_params,
    ) = fit_hormesis_free_response(data, max_survival, hormesis_index, interpolate)

    observed_stress = survival_to_stress(survival_rate, p=beta_p, q=beta_q)
    tox_stress = survival_to_stress(tox_survival, p=beta_p, q=beta_q)

    sys_stress_points = np.clip(observed_stress - tox_stress, 0, 1)

    if np.all(sys_stress_points == 0):
        raise ValueError("All sys stress points are 0")

    sys_stress_params = lbfgs_fit(
        tox_stress.tolist(), sys_stress_points.tolist(), b=None, c=0, d=None, e=None
    )
    sys_stress_func = np.vectorize(fix_wlb1(sys_stress_params))

    sys_stress = sys_stress_func(np.maximum(tox_stress, 1e-8))

    if any(sys_stress < 0):
        raise ValueError(
            "Something went wrong. The predicted system stress should not be < 0"
        )

    tox_sys_stress = tox_stress + sys_stress
    tox_sys_survival = stress_to_survival(tox_sys_stress, p=beta_p, q=beta_q)

    def get_lc_only_tox(lc):
        val = 1 - (lc / 100)
        return weibull_2param_inverse(val, b=tox_fit_params.b, e=tox_fit_params.e)

    lc99_9 = get_lc_only_tox(99.9)
    concentrations_smooth = np.logspace(
        np.log10(pad_c0(data.concentration)[0]),
        np.log10(max(concentrations.max(), lc99_9)),
        len_curves,
    )
    tox_survival_smooth_raw = cleaned_tox_func(concentrations_smooth)
    tox_stress_smooth = survival_to_stress(tox_survival_smooth_raw, p=beta_p, q=beta_q)
    tox_survival_smooth = tox_survival_smooth_raw * max_survival
    sys_stress_smooth = sys_stress_func(tox_stress_smooth)
    tox_sys_stress_smooth = tox_stress_smooth + sys_stress_smooth
    tox_sys_survival_smooth = (
        stress_to_survival(tox_sys_stress_smooth, p=beta_p, q=beta_q) * max_survival
    )

    return ECxSySOutput(
        input_data=data,
        hormesis_index=hormesis_index,
        concentration=concentrations_smooth,
        tox_survival=tox_survival_smooth,
        tox_stress=tox_stress_smooth,
        sys_stress=sys_stress_smooth,
        tox_sys_stress=tox_sys_stress_smooth,
        tox_sys_survival=tox_sys_survival_smooth,
        beta_p=beta_p,
        beta_q=beta_q,
        tox_surv_params=tox_fit_params._asdict(),
        sys_stress_params=sys_stress_params._asdict(),
        max_survival=max_survival,
    )


COLORS = color_palette("tab10", 4)
TOX_COLOR = COLORS[0]
TOX_SYS_COLOR = COLORS[1]
SYS_COLOR = COLORS[2]
HORMESIS_COLOR = COLORS[3]
DATA_COLOR = "black"


@dataclass
class ECxSySOutput:
    input_data: CauseEffectData
    hormesis_index: int
    concentration: np.ndarray
    tox_survival: np.ndarray
    tox_stress: np.ndarray
    sys_stress: np.ndarray
    tox_sys_stress: np.ndarray
    tox_sys_survival: np.ndarray

    beta_q: float
    beta_p: float
    tox_surv_params: dict[str, float]
    sys_stress_params: dict[str, float]
    max_survival: float

    def plot(self, figsize=(10, 4), title=None) -> Figure:
        fig = plt.figure(figsize=figsize)
        plt.subplot(1, 2, 1)
        plt.plot(
            self.concentration,
            self.tox_sys_survival,
            label="Tox+Sys",
            color=TOX_SYS_COLOR,
        )
        plt.plot(self.concentration, self.tox_survival, label="Tox", color=TOX_COLOR)
        plt.scatter(
            pad_c0(self.input_data.concentration),
            self.input_data.survival_rate,
            c=[
                DATA_COLOR if i != self.hormesis_index else HORMESIS_COLOR
                for i in range(len(self.input_data.concentration))
            ],
            s=SCATTER_SIZE,
        )

        plt.xscale("log")
        plt.title("Survival")

        plt.subplot(1, 2, 2)
        plt.scatter([], [], color=DATA_COLOR, label="Measurements")
        plt.scatter([], [], color=HORMESIS_COLOR, label="Hormesis Point")
        plt.plot(self.concentration, self.tox_stress, label="Tox", color=TOX_COLOR)
        plt.plot(
            self.concentration,
            self.tox_sys_stress,
            label="Tox+Sys",
            color=TOX_SYS_COLOR,
        )
        plt.plot(self.concentration, self.sys_stress, label="Sys", color=SYS_COLOR)
        plt.xscale("log")
        plt.title("Stress")
        plt.legend()
        if title is not None:
            plt.suptitle(title)
        plt.tight_layout()
        return fig
