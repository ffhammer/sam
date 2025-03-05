# SAM - Stress Addition Model

SAM is a Python reimplementation of the Stress Addition Model from the paper "[Predicting the synergy of multiple stress effects](https://www.nature.com/articles/srep32965)" by Liess et al. It follows the methodology from [Systemecology/Indicate](https://www.systemecology.de/indicate/) and achieves nearly identical results across diverse datasets.

We also reimplemented the [$EC_{x-\text{sys}}$ model](https://doi.org/10.1038/s41598-019-51645-4) from Liess et al. (2019) as `sam.ecxsys`. This package lets you predict, visualize, and save outcomes for both SAM and $EC_{x-\text{sys}}$, even with minimal Python experience.

Examples:  
- `examples/sam.py`  
- `examples/ec_x-SyS.py`

## Install
```bash
pip install git+https://github.com/mockaWolke/sam.git
```

## Dataset and Publication

This repository is under active development for an upcoming publication. The `data` folder contains collated datasets on the effects of multiple stressors. More details will be available on our publication website: [SAM Publication](https://ffhamer.github.io/sam/).

## Detailed Documentation

Further documentation—including experimental setups and dataset handling—is available at: [SAM Documentation](https://ffhammer.github.io/sam_documentation/).

## Citation
```bibtex
@Manual{,
    title = {stressaddition: Modelling Tri-Phasic Concentration-Response Relationships},
    author = {Sebastian Henz and Matthias Liess},
    year = {2020},
    note = {R package version 3.1.0},
    url = {https://CRAN.R-project.org/package=stressaddition},
}
```

## Copyright and License

Copyright (C) 2025 Helmholtz-Zentrum fuer Umweltforschung GmbH - UFZ

This program is free software: you can redistribute it and/or modify  
it under the terms of the GNU General Public License as published by  
the Free Software Foundation, either version 3 of the License, or  
(at your option) any later version.  

This program is distributed in the hope that it will be useful,  
but WITHOUT ANY WARRANTY; without even the implied warranty of  
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the  
GNU General Public License for more details.  

You should have received a copy of the GNU General Public License  
along with this program. If not, see <https://www.gnu.org/licenses/>.  

<!-- 
## Academic Citation Requirement:
If you use this software in academic work, research, or publications,  
you must cite the following reference:  

[Your Name], "[Title of Your Work]," [Journal/Conference], [Year], [DOI or URL].  

Failure to provide appropriate citation in academic works constitutes  
a violation of this licensing agreement.   -->
