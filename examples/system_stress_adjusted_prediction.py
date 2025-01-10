import matplotlib.pyplot as plt
from sam import (
    read_data,
    generate_sys_adjusted_sam_prediction,
    SysAdjustedSamPrediction,
)


data = read_data("data/2019 Naeem-Esf, Pro, food/21_days.xlsx")

prediction: SysAdjustedSamPrediction = generate_sys_adjusted_sam_prediction(
    control_data=data.main_series,
    co_stressor_data=data.additional_stress.get("Food_1% + Prochloraz_100"),
    additional_stress=0.18,
    hormesis_index=3,
    max_survival=data.meta.max_survival,
)

prediction.plot()
plt.show()
