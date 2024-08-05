import os
from model import dose_response_fit, ModelPredictions
from plotting import plot_complete
import pandas as pd
import numpy as np

for path in os.listdir("formatted_data"):

    df = pd.read_csv(f"formatted_data/{path}")

    conc = df.conc.values.astype(np.float64)
    survival = df["no stress"].values.astype(np.float64)
    hormesis = df.hormesis_concentration.iloc[0]

    if np.isnan(hormesis):
        hormesis = None

    res: ModelPredictions = dose_response_fit(
        conc, survival, hormesis_concentration=hormesis
    )

    title = f"{df.chemical.iloc[0]} - {df.organism.iloc[0]}"
    fig = plot_complete(model=res, title=title)

    save_path = f"python_plots/{path.replace('csv', 'png')}"

    fig.savefig(save_path)
