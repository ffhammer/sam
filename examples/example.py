from sam import *
import matplotlib.pyplot as plt

# Example CauseEffectData data
control_data = CauseEffectData(
    concentration=[0, 0.1, 0.5, 1.0, 5.0],
    survival_rate=[100, 98, 85, 50, 10],
    name="Control",
)

co_stressor_data = CauseEffectData(
    concentration=[0, 0.1, 0.5, 1.0, 5.0],
    survival_rate=[97, 95, 70, 30, 5],
    name="Stressor",
)


# Run SAM prediction
prediction = generate_sam_prediction(
    control_data=control_data,
    co_stressor_data=co_stressor_data,
    max_survival=100,
)


fig = prediction.plot(
    title="SAM Prediction Example", with_lcs=True, inlcude_control_addition=True
)

plt.show()
