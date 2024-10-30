import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))


import pandas as pd

# # Load the Excel file
# df = pd.read_excel("../data_template.xlsx")

# # Convert to reStructuredText table format
# with open("data_template_table.rst", "w") as file:
#     file.write(df.to_markdown(index=False))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SAM'
copyright = '2024, Felix Hammer'
author = 'Felix Hammer'
release = '30.10.2024'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

html_theme = 'alabaster'             # You can change this to any other theme
html_static_path = ['_static']
extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',           # Automatically document from docstrings
    'sphinx.ext.napoleon',           # Support for Google-style and NumPy-style docstrings
    'sphinx_autodoc_typehints',      # Include type hints in the docs (optional)
]

master_doc = "home"
root_doc = "home"

templates_path = ['_templates']
exclude_patterns = [
    'experiments',
    'imgs',               # Ignore the imgs folder
    'templates',          # Ignore the templates folder
    'index.md',           # Ignore index.md
    '*.md',               # Ignore all other .md files
    '_build', 'Thumbs.db', '.DS_Store'  # Default exclusions
]



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_material' 
html_static_path = ['_static']
autodoc_member_order = 'bysource'