from setuptools import setup, find_packages

setup(
    name="sam",
    version="0.1",
    description="Python implementation of the Stress Addition Model (SAM) for modeling multi-stressor interactions",
    long_description=(
        "SAM (Stress Addition Model) is a Python implementation of the model proposed in the research paper "
        "'Predicting the synergy of multiple stress effects' by Matthias Liess, Kaarina Foit, Saskia Knillmann, "
        "Ralf B. Schäfer, and Hans-Dieter Liess. This package provides tools for modeling and predicting "
        "the combined effects of multiple environmental stressors on biological organisms using a tri-phasic concentration-response approach."
    ),
    long_description_content_type="text/plain",
    author="Felix Hammer",
    author_email="fhammer@uos.de",
    url="https://github.com/mockaWolke/sam",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "matplotlib",
        "numpy",
        "pandas",
        "PyYAML",
        "dataclass_json",
        "scikit_learn",
        "scipy",
        "seaborn",
        "openpyxl",
        "tqdm",
        "py_lmcurve_ll5 @ git+https://github.com/mockaWolke/py_lmcurve_ll5.git",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    license="GNU General Public License v3",
    project_urls={
        "Citation": "https://CRAN.R-project.org/package=stressaddition",
        "License": "https://www.gnu.org/licenses/gpl-3.0.en.html",
    },
)
