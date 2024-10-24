from glob import glob
import os
from sam import REPO_PATH
from tqdm import tqdm
import glob
from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple, Callable
import yaml
from pathlib import Path


@dataclass
class ExperimentMetaData:
    organism: str
    chemical: str
    max_survival: float
    path: str
    path: str
    days: int
    title: str
    hormesis_concentration: Optional[int] = None
    pub: Optional[str] = None


@dataclass()
class DoseResponseSeries:
    concentration: np.ndarray
    survival_rate: np.ndarray
    name: str
    meta: ExperimentMetaData

    def __post_init__(self):
        self.concentration = self.concentration.astype(np.float64)
        self.survival_rate = self.survival_rate.astype(np.float64)

        self.concentration.setflags(write=False)
        self.survival_rate.setflags(write=False)

        if len(self.concentration) != len(self.survival_rate):
            raise ValueError(
                "concentration and survival_observerd must have the same length."
            )
        if len(self.concentration) > len(set(self.concentration)):
            raise ValueError("Concentrations must be unique.")
        if (np.sort(self.concentration) != self.concentration).all():
            raise ValueError("The concentration values must be in sorted order.")
        if any(np.array(self.concentration) < 0):
            raise ValueError("Concentrations must be >= 0")
        if min(self.concentration) > 0:
            raise ValueError("No control is given. The first concentration must be 0.")
        if np.isnan(self.concentration).any():
            raise ValueError("concentration must be none NaN")
        if np.isnan(self.survival_rate).any():
            raise ValueError("survival_observerd must be none NaN")

    def __eq__(self, other):
        if not isinstance(other, DoseResponseSeries):
            return False
        return (
            self.name == other.name
            and np.all(self.concentration == other.concentration)
            and np.all(self.survival_rate == other.survival_rate)
        )


@dataclass
class ExperimentData:
    main_series: DoseResponseSeries
    additional_stress: Dict[str, DoseResponseSeries]
    meta: ExperimentMetaData


def read_metadata(path: str, df: pd.DataFrame) -> ExperimentMetaData:

    path: Path = Path(path)

    par_dir = path.parent
    meta_path = par_dir / "meta.yaml"

    if not os.path.exists(meta_path):
        raise ValueError(f"Cant find {meta_path}")

    with open(meta_path, "r") as file:
        meta_dict: dict = yaml.safe_load(file)

    expected_meta_columns = ["meta_category", "info"]
    if not (df.columns[-2:] == expected_meta_columns).all():
        raise ValueError(
            f"Expected last two columns to be {expected_meta_columns}, but got {df.columns[-2:].tolist()}"
        )

    meta = df[["meta_category", "info"]].dropna()
    
    if len(meta):    
        df_dict = dict(zip(meta.meta_category.str.strip(), meta["info"]))

        for key in df_dict:
            if key in meta_dict:
                raise ValueError(
                    f"You have set {key} in both the meta.yaml and the {path}. Duplicates are not allowed!"
                )
            meta_dict[key] = df_dict[key]

    # create name
    child_name = path.name.replace(".xlsx", "")

    title = par_dir.name

    if child_name != "data":
        title += "_" + child_name

    return ExperimentMetaData(**meta_dict, title=title, path=path)


def read_data(path: str) -> ExperimentData:
    df = pd.read_excel(path)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    expected_columns = ["concentration", "survival"]
    if not (df.columns[:2] == expected_columns).all():
        raise ValueError(
            f"Expected first two columns to be {expected_columns}, but got {df.columns[:2].tolist()}"
        )
    meta_data = read_metadata(path, df)
    main_series = DoseResponseSeries(
        df["concentration"].values,
        df["survival"].values,
        name="Toxicant",
        meta=meta_data,
    )

    additional_stresses = [
        col
        for col in df.columns
        if col not in ["concentration", "survival", "meta_category", "info"]
    ]
    additional_stress_dict = {
        name: DoseResponseSeries(
            df["concentration"].values,
            df[name].values,
            name=name,
            meta=meta_data,
        )
        for name in additional_stresses
    }

    return ExperimentData(
        main_series=main_series,
        additional_stress=additional_stress_dict,
        meta=meta_data,
    )


def load_files(filter: Optional[Callable] = None) -> Tuple[str, ExperimentData]:
    paths = glob.glob(os.path.join(REPO_PATH, "data/*/*.xlsx"))

    if filter is not None:
        paths = [i for i in paths if filter(i)]

    return [(path, read_data(path)) for path in paths]


def load_datapoints(
    filter: Optional[Callable] = None,
) -> list[Tuple[str, ExperimentData, str, DoseResponseSeries]]:

    files = load_files(filter=filter)

    return [
        (path, data, stress_name, stress_series)
        for path, data in files
        for stress_name, stress_series in data.additional_stress.items()
    ]


if __name__ == "__main__":

    load_datapoints()
