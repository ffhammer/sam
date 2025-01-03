import os
from pathlib import Path
from sam import read_data
import argparse

import json

parser = argparse.ArgumentParser()
parser.add_argument("--dir4imgs", type=str)
args = parser.parse_args()
dir4imgs = Path(args.dir4imgs)

with open("docs/add_sys_examples.json", "r") as f:
    examples = json.load(f)

os.chdir(os.environ["SAM_REPO_PATH"])

PAGE_PATH = Path("docs/add_sys.md")

lines = [
    "# System Stress-Adjusted SAM Predictions for Experiments with Two Strong Co-Stressors\n\n",
    """In experiments involving two strong additional co-stressors, the SAM model tends to overestimate survival rates. 
We have developed an effective approach to address this issue using the following algorithm:

1. **Concentration-Response Fit on Adjusted Data**:
   - The input data is processed to exclude sub-hormesis effects, setting the control survival rate (at concentration = 0) to 100%.
   - A concentration-response fit is performed to approximate the effect without inherent system stress.

2. **Prediction Without System Stress**:
   - Survival rates for the `control_data.concentration` values are predicted using the "cleaned" model (with system stress removed).

3. **Conversion to Stress, Adding Additional Stress, and Back to Survival**:
   - Survival predictions are converted to stress values.
   - An additional manual stress of `+0.18` is applied to all values, which are then converted back to survival rates.

4. **SAM Prediction**:
   - The adjusted control survival data is used to generate a new SAM prediction incorporating the modified data.
""",
]

for path, stressor_name, hormesis_index in examples:
    data = read_data(path)
    base_path = f"{dir4imgs.name}/" + f"{data.meta.title}_{stressor_name}".replace(
        " ", "_"
    )

    nicer = lambda x: x.replace("_", " ")
    lines.append(f"""## {nicer(data.meta.title)} -  {nicer(stressor_name)}
                 
### Standard SAM Prediction:
![{data.meta.experiment_name} SAM Prediction](imgs/{dir4imgs.name}/{base_path}_sam.png)

### System Stress Adjusted Prediction: 
![{data.meta.experiment_name} SYS Adjusted Prediction](imgs/{dir4imgs.name}/{base_path}_sys.png)


""")

with open(PAGE_PATH, "w") as f:
    f.writelines(lines)
