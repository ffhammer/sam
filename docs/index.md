# Stress Addition Model (SAM)


This is a Python reimplementation of the Stress Addition Model (SAM) from the research paper "Predicting the synergy of multiple stress effects" by Matthias Liess, Kaarina Foit, Saskia Knillmann, Ralf B. Schäfer, and Hans-Dieter Liess.



## Sub Pages

- **[Impact of Additional Stress on Lethal Concentrations (LCs)](lc_increase_with_stress.md)**  
  This page explores how additional environmental stress impacts lethal concentrations, visualizing how LC10 and LC50 levels change in response to stress according to SAM.

- **[Dose-Response Fits Across Experiments](dose_response_fits.md)**  
  This section compares dose-response fits across experiments. It includes both "raw" curves (direct SAM fits) and "cleaned" curves that account for inherent stress by normalizing survival rates.

## Experiments

The following is a list of individual experiments conducted, each with detailed data and SAM model predictions:


- [2024_Imrana-food.md](experiments/2024_Imrana-food.md)

- [2024_Imrana-Cu.md](experiments/2024_Imrana-Cu.md)

- [2023_Huang_Neon_Flupyradifurone.md](experiments/2023_Huang_Neon_Flupyradifurone.md)

- [2024_Imrana-salt,_food.md](experiments/2024_Imrana-salt,_food.md)

- [2022_Ayesha-Chlorantran.md](experiments/2022_Ayesha-Chlorantran.md)

- [2021_Ayesha-Cloth,_C,_Adap.md](experiments/2021_Ayesha-Cloth,_C,_Adap.md)

- [2024_Naeem-Esfe,_C,_food.md](experiments/2024_Naeem-Esfe,_C,_food.md)

- [2024_BPS,_Esf,_food.md](experiments/2024_BPS,_Esf,_food.md)

- [2019_Naeem-Esf,_Pro,_food.md](experiments/2019_Naeem-Esf,_Pro,_food.md)

- [2024_Naeem-Mix13,C,food.md](experiments/2024_Naeem-Mix13,C,food.md)

- [2023_Huang_Neon_Imidachloprid.md](experiments/2023_Huang_Neon_Imidachloprid.md)

- [2001_Liess-food,_UV_Cu.md](experiments/2001_Liess-food,_UV_Cu.md)

- [2024_Naeem-2Tox,_C,_Adapt.md](experiments/2024_Naeem-2Tox,_C,_Adapt.md)



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

Copyright (c) 2020,  
Helmholtz-Zentrum fuer Umweltforschung GmbH - UFZ.  
All rights reserved.

The code is a property of:

Helmholtz-Zentrum fuer Umweltforschung GmbH - UFZ  
Registered Office: Leipzig  
Registration Office: Amtsgericht Leipzig  
Trade Register: Nr. B 4703  
Chairman of the Supervisory Board: MinDirig’in Oda Keppler  
Scientific Director: Prof. Dr. Georg Teutsch  
Administrative Director: Dr. Sabine König

SAM (Stress Addition Model) is free software: you can redistribute it and/or modify  
it under the terms of the GNU General Public License as published by  
the Free Software Foundation, either version 3 of the License, or  
(at your option) any later version.

This program is distributed in the hope that it will be useful,  
but WITHOUT ANY WARRANTY; without even the implied warranty of  
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the  
GNU General Public License for more details.

You should have received a copy of the GNU General Public License  
along with this program. If not, see <https://www.gnu.org/licenses/>.