from sam import *
import matplotlib.pyplot as plt

# Example DoseResponseSeries data
control_series = DoseResponseSeries(
    concentration=[0, 0.1, 0.5, 1.0, 5.0],
    survival_rate=[100, 98, 85, 50, 10],
    name="Control",
)

stressor_series = DoseResponseSeries(
    concentration=[0, 0.1, 0.5, 1.0, 5.0],
    survival_rate=[100, 95, 70, 30, 5],
    name="Stressor",
)


# Run SAM prediction
prediction = sam_prediction(
    main_series=control_series,
    stressor_series=stressor_series,
    max_survival=100,
)


# Plot results
fig = prediction.plot(title="SAM Prediction Example")

plt.show()
