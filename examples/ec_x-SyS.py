from sam import CauseEffectData
from sam.ec_x_sys import generate_ecx_sys_prediction, ECxSySPrediction
import matplotlib.pyplot as plt

data = CauseEffectData(
    concentration=[0, 0.001, 0.01, 0.0316, 0.1, 0.316, 1, 3.16],
    survival_rate=[0.82, 0.74, 0.7, 0.8, 0.72, 0.53, 0.07, 0.0],
)

pred: ECxSySPrediction = generate_ecx_sys_prediction(
    data=data, max_survival=1, hormesis_index=3
)

# plot
fig = pred.plot()


plt.show()
