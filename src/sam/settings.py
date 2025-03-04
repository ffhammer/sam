from dataclasses import dataclass, field
from typing import Callable, Optional

from dataclasses_json import config, dataclass_json


from .concentration_response_fits import (
    Transforms,
)
from .stress_survival_conversion import stress_to_survival, survival_to_stress


@dataclass_json
@dataclass
class SAM_Settings:
    """Settings for configuring the SAM (Stress Addition Model) used in concentration-response predictions."""

    #: If `True`, normalizes survival values based on environmental stress.
    normalize_survival_for_stress_conversion: bool = False

    #: Defines the formula used to calculate environmental stress. Supported options:
    #: - `"div"`: `additional_stress = stressor / control`
    #: - `"subtract"`: `additional_stress = 1 - (control - stressor)`
    #: - `"only_stress"`: `additional_stress = 1 - stressor`
    #: - `"stress_sub"`: Subtracts transformed control stress from transformed stressor stress.
    stress_form: str = "div"

    #: Transformation function applied to regression data prior to model fitting.
    transform: "Transforms" = field(
        default=Transforms.williams_and_linear_interpolation,
        metadata=config(
            encoder=lambda t: t.__name__.replace("transform_", ""),
            decoder=lambda t: getattr(Transforms, t),
        ),
    )

    #: q Parameter of beta distribution for survival to stress and vice versa conversions
    beta_q: float = 3.2

    #: p Parameter of beta distribution for survival to stress and vice versa conversions
    beta_p: float = 3.2

    #: Controls which library is used for DoseResponse Curve fitting. Either scipy for scipy.optimize.curve_fit or lmcurce for using https://github.com/MockaWolke/py_lmcurve_ll5
    curve_fit_lib: str = "scipy"

    fix_f_parameter_ll5: Optional[float] = None

    keep_co_stressor_f_parameter_free: bool = False

    e_param_fac: float = 1.0

    def __post_init__(
        self,
    ):
        self.stress_to_survival: Callable = lambda x: stress_to_survival(
            x, p=self.beta_p, q=self.beta_q
        )
        self.survival_to_stress: Callable = lambda x: survival_to_stress(
            x, p=self.beta_p, q=self.beta_q
        )


STANDARD_SAM_SETTING = SAM_Settings(
    normalize_survival_for_stress_conversion=True,
    stress_form="div",
    fix_f_parameter_ll5=1.0,
    keep_co_stressor_f_parameter_free=False,
)
