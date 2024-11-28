import os
from sam.data_formats import load_files, ExperimentData
from collections import defaultdict
from pathlib import Path
import yaml

os.chdir(os.environ["SAM_REPO_PATH"])

EXPERIMENTS_PATH = Path("docs/experiments")
EXPERIMENTS_PATH.mkdir(exist_ok=True)

experiments = defaultdict(list)

for _, data in load_files():
    data : ExperimentData
    experiments[data.meta.experiment_name].append(data)



def create_page_for_data(name, exps: list[ExperimentData]):
    
    
    exp_path = EXPERIMENTS_PATH / f"{name}.md".replace(" ", "_").replace("°C", "C")
    meta_yaml_file = Path(exps[0].meta.path).parent / "meta.yaml"
    
    # Read YAML file content
    with open(meta_yaml_file, 'r') as file:
        yaml_content = yaml.safe_load(file)
    
    # Convert YAML content to formatted text
    yaml_text = yaml.dump(yaml_content, default_flow_style=False)
    
    # Write to Markdown
    text = f"""# {name}\n\n## Experiment Metadata\n\n```yaml\n{yaml_text}\n```\n"""

    for exp in sorted(exps, key=lambda x: x.meta.title):
        table = exp.to_markdown_table()
        title = exp.meta.title.replace(" ","_")
        text += f"\n\n## {title}\n\n### Data\n\n{table}\n\n### SAM Predictions"
        
        
        for additional_stressor in exp.additional_stress:
            img_path = f"imgs/sam_predictions/{title}_{additional_stressor}.png".replace(" ","_")
            nicer = additional_stressor.replace("_", " ")
            
            if (Path("docs") / img_path).exists():
                text += f"### {nicer}\n\n![SAM Prediction](../{img_path})\n"
            else:
                print("Cant find:", img_path)

    with open(exp_path, "w") as f:
        f.write(text)

for name, exps in experiments.items():
    create_page_for_data(name, exps)