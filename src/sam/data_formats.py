from glob import glob
import os
from tqdm import tqdm
import glob
from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple, Callable, Type
import yaml
from pathlib import Path


@dataclass
class ExperimentMetaData:
    """
    Metadata for the experiment, including essential information such as organism, duration, and conditions.

    Notes:
        - Attributes such as `path`, `title`, and `experiment_name` are automatically inferred 
          when loading data. The `path` points to the `.xlsx` file location, while `title` is constructed 
          from `experiment_name` and the Excel file name (`xlsx_name`). If `xlsx_name` is "data", 
          only the `experiment_name` is used.
    """
    
    #: The name of the organism used in the experiment.
    organism: str 
    
    #: The primary stressor applied in the control series.
    main_stressor: str
    
    #: The maximum observed survival rate, representing the control condition.
    max_survival: float
    
    #: Duration of the experiment in days.
    days: int
    
    #: Name of the experiment for record-keeping purposes.
    experiment_name: str
    
    #: Constructed title from experiment name and Excel file name.
    title: str
    
    #: Path to the source `.xlsx` file.
    path: str
    
    #: Concentration corresponding to the hormesis effect in the control series, if applicable.
    hormesis_concentration: Optional[int] = None
    
    #: Optional name of the publication or journal where results are published, if applicable.
    pub: Optional[str] = None


@dataclass
class DoseResponseSeries:
    """
    Represents a dose-response data series for use in `sam.dose_response_fit`.

    This class holds concentration and survival rate data for dose-response modeling, 
    ensuring that input data meets specific requirements essential for accurate modeling.
    The `concentration` values must be a sorted, unique, and non-negative sequence starting 
    with a control value of 0, and both `concentration` and `survival_rate` arrays must be 
    of the same length. The `name` attribute is used for labeling plots, and `meta` provides 
    optional metadata for internal reference.

    Raises:
        ValueError: If the `concentration` and `survival_rate` lengths do not match.
        ValueError: If `concentration` values are not unique, sorted, non-negative, and starting with 0.
        ValueError: If any `concentration` or `survival_rate` value is NaN.

    Example:
        Create a dose-response series for use in the SAM model:
        
        >>> series = DoseResponseSeries(
        >>>     concentration=[0, 1.0, 2.5],
        >>>     survival_rate=[100, 95, 85],
        >>>     name="Example Data"
        >>> )
    """
    
    #: A non-negative, sorted array of unique concentration values (first value must be the control, i.e., 0).
    concentration: np.ndarray
    
    #: Array of survival rates corresponding to each concentration value, with matching length to `concentration`.
    survival_rate: np.ndarray
    
    #: Label for the dose-response series, used primarily in plotting.
    name: str
    
    #: Additional experimental metadata, used mainly for internal purposes.
    meta: Optional['ExperimentMetaData'] = None

    def __post_init__(self):
        self.concentration = np.array(self.concentration, dtype = np.float64)
        self.survival_rate = np.array(self.survival_rate, dtype = np.float64)

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


def read_meta_yaml(xlsx_path : Path) -> dict:
    meta_path = xlsx_path.parent / "meta.yaml"

    if not os.path.exists(meta_path):
        raise ValueError(f"Cant find {meta_path}")

    with open(meta_path, "r") as file:
        return  yaml.safe_load(file)



def read_metadata(path: str, df: pd.DataFrame) -> ExperimentMetaData:

    path: Path = Path(path)

    meta_dict = read_meta_yaml(path)


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

    title = path.parent.name

    if child_name != "data":
        title += "_" + child_name

    return ExperimentMetaData(**meta_dict, title=title, path=path, experiment_name=Path(path).parent.name)

@dataclass
class ExperimentData:
    """
    Represents data from a stress addition experiment, including control and additional stressor series.

    Methods:
        from_xlsx(path: str) -> ExperimentData: Class method to load data from an Excel file, constructing 
            the main and additional stress series based on the expected data template.
        to_markdown_table(): Converts data to a Markdown table format, including non-duplicate metadata.

    Notes:
        Meant to be created by calling from_xlsx like this:
        
    Example:
        >>> data = ExperimentData.from_xlsx("path/to/data_template.xlsx")
        >>> print(data.main_series)  # Access the main control series
        >>> print(data.to_markdown_table())  # Display data as a Markdown table
    """
    
    #: Dose-response data for the control series.
    main_series: DoseResponseSeries
    
    #: Dictionary of additional stressor series, with stressor names as keys.
    additional_stress: Dict[str, DoseResponseSeries]
    
    #: Metadata for the experiment, including organism, duration, and conditions.
    meta: ExperimentMetaData


    def to_markdown_table(self):
        
        
        cols = {"Concentration": self.main_series.concentration, 
                "Control Survival Rate": self.main_series.survival_rate}
        
        for name, ser in self.additional_stress.items():
            cols[name.replace("_"," ")] = ser.survival_rate
            
        df = pd.DataFrame.from_dict(cols, orient="columns")
        markdown_text = df.to_markdown()
        
        # if there is meta data show below the table
        yaml_meta = read_meta_yaml(self.meta.path)
        cur_meta = vars(self.meta).copy()
        del cur_meta["path"]
        del cur_meta["experiment_name"]
        del cur_meta["title"]
        
        non_dup_meta = {key:val for key,val in cur_meta.items() if key not in yaml_meta and val is not None}
        
        if len(non_dup_meta):
            yaml_text = "\n".join([f"{key}: '{val}'" for key,val in non_dup_meta.items()])
            yam_template ="```yaml\n{}\n```\n"
            markdown_text += "\n\nSpecific Settings:\n\n" + yam_template.format(yaml_text)
            
        return markdown_text
        
    @classmethod
    def from_xlsx(cls, path : str) -> 'ExperimentData':
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

        return cls(
            main_series=main_series,
            additional_stress=additional_stress_dict,
            meta=meta_data,
        )
        

def read_data(path: str) -> ExperimentData:
    return ExperimentData.from_xlsx(path)



def load_files(filter: Optional[Callable] = None, data_dir : Optional[str] = None) -> Tuple[str, ExperimentData]:
    if data_dir is None:
        data_dir = os.path.abspath("data")
        
    glob_path = str(Path(data_dir) / "*/*.xlsx")
    paths = glob.glob(str(glob_path))

    if len(paths) == 0:
        raise ValueError(f"Cand find any experiments at {data_dir}. You can control this via the data_dir argument. ")

    if filter is not None:
        paths = [i for i in paths if filter(i)]

    return [(path, read_data(path)) for path in paths]

def load_datapoints(
    filter: Optional[Callable] = None,  data_dir : Optional[str] = None,
) -> list[Tuple[str, ExperimentData, str, DoseResponseSeries]]:

    files = load_files(filter=filter, data_dir=data_dir)

    return [
        (path, data, stress_name, stress_series)
        for path, data in files
        for stress_name, stress_series in data.additional_stress.items()
    ]


if __name__ == "__main__":

    load_datapoints()
