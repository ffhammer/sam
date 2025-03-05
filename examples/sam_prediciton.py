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

# saving and loading
prediction.save_to_file("my_sam_prediction")

loaded_prediction: SAMPrediction = SAMPrediction.load_from_file("my_sam_prediction")


plt.show()
