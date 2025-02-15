import matplotlib.pyplot as plt
from sam import read_data
from sam.system_stress import generate_sys_adjusted_sam_prediction


def test_sys_adjusted_pred():
    data = read_data("data/2019 Naeem-Esf, Pro, food/21_days.xlsx")
    ser = data.additional_stress["Prochloraz_1 + Food_1%"]

    generate_sys_adjusted_sam_prediction(
        control_data=data.main_series,
        co_stressor_data=ser,
        additional_stress=0.18,
        hormesis_index=3,
        meta=data.meta,
    ).plot()
    plt.close()
