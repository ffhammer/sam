import os
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from sam.ecx_sys import generate_ecx_sys_prediction
from sam import load_files, ExperimentData
from collections import defaultdict
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dir4imgs")
args = parser.parse_args()
dir4imgs = Path(args.dir4imgs)
dir4imgs.mkdir(exist_ok=True, parents=True)

os.chdir(os.environ["SAM_REPO_PATH"])
print(os.getcwd())

DOCS_PATH = Path("docs")
TEMPLATE_PATH = DOCS_PATH / "templates"
OUTPUT_PATH = DOCS_PATH / "ecx_sys.md"

# Load and configure Jinja2 environment
env = Environment(loader=FileSystemLoader(TEMPLATE_PATH))
template = env.get_template("ecx_sys.md")

resulst_by_experiments = defaultdict(list)

for _, data in load_files():
    data: ExperimentData
    if data.main_series.survival_rate[-1] != 0 or (data.hormesis_index or 0) < 2:
        continue

    try:
        prediction = generate_ecx_sys_prediction(
            data=data.main_series,
            max_survival=data.meta.max_survival,
            hormesis_index=data.hormesis_index,
        )

    except Exception:
        continue

    experiment_name = data.meta.experiment_name
    img_path = dir4imgs / f"{data.meta.title.replace(' ', '_')}.png"
    rel_doc_path = f"imgs/{dir4imgs.name}/{data.meta.title.replace(' ', '_')}.png"
    fig = prediction.plot()
    fig.savefig(img_path)
    plt.close()

    resulst_by_experiments[experiment_name].append(
        (data.meta.specific_name, rel_doc_path)
    )

# Render the markdown file
experiments = [
    {
        "title": experiment,
        "series": [
            {"name": specific_name, "img_path": str(img_path)}
            for specific_name, img_path in series
        ],
    }
    for experiment, series in resulst_by_experiments.items()
]

rendered_content = template.render(experiments=experiments)

# Write the rendered markdown to the output file
with open(OUTPUT_PATH, "w") as f:
    f.write(rendered_content)

print(f"Markdown file generated at {OUTPUT_PATH}")
