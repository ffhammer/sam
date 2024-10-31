import os
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
import os

os.chdir(os.environ["SAM_REPO_PATH"])
print(os.getcwd())

DOCS_PATH = Path("docs")
TEMPLATE_PATH = DOCS_PATH / "templates"
OUTPUT_PATH = DOCS_PATH / "index.md"

# List all experiment files in the "experiments" directory
experiments = os.listdir(DOCS_PATH / "experiments")

# Load and configure Jinja2 environment
env = Environment(loader=FileSystemLoader(TEMPLATE_PATH))
template = env.get_template("index.md")

# Render the template with the list of experiments
output_content = template.render(experiments=experiments)

# Save the rendered content to index.md
OUTPUT_PATH.write_text(output_content)