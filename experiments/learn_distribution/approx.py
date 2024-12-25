from sam.plotting import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pysr import PySRRegressor


path = "better2.csv"

df = pd.read_csv(path)

x = df.right_borders.values[:, None]
y = np.cumsum(df.probs).values


pysr_model = PySRRegressor(
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["exp", "log", "tanh", "sqrt", "inv(x) = 1/x"],
    extra_sympy_mappings={"inv": lambda x: 1 / x},
    niterations=100,  # Adjust depending on the complexity of your function
    populations=50,  # More populations to better explore the equation space
    progress=True,  # Display progress bar
    random_state=False,
)

x = df.right_borders.values[:, None]
y = np.cumsum(df.probs).values


pysr_model.fit(x, y)

pysr_model_padded = PySRRegressor(
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["exp", "log", "tanh", "sqrt", "inv(x) = 1/x"],
    extra_sympy_mappings={"inv": lambda x: 1 / x},
    niterations=20,  # Adjust depending on the complexity of your function
    populations=20,  # More populations to better explore the equation space
    progress=True,  # Display progress bar
    random_state=0,
)

print("Model", pysr_model.latex())

with open(path.replace(".csv", "_latex.txt"), "w") as f:
    f.write(pysr_model.latex())

plt.plot(x, y, label="learned")

y_pred = pysr_model.predict(x)
plt.plot(x, y_pred, label="psyr")
plt.legend()
plt.show()
