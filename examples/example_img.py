from sam import generate_sam_prediction, SAMPrediction
import matplotlib.pyplot as plt


# generate SAM prediction
prediction: SAMPrediction = generate_sam_prediction(
    concentration=[0, 0.1, 0.5, 1.0, 5.0],
    control_survival=[100, 98, 85, 50, 10],
    co_stressor_survival=[97, 95, 70, 30, 5],
    max_survival=100,
)

# plot
fig = prediction.plot(
    title="SAM Prediction Example", with_lcs=True, inlcude_control_addition=True
)
fig.savefig("docs/imgs/example.png")
plt.close()
