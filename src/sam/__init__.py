import os
from pathlib import Path


REPO_PATH = Path(__file__).parent.parent.parent

def chdir_to_repopath():
    os.chdir(REPO_PATH)