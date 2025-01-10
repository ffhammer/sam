# ECx-SyS: Enhanced System Stress Predictions

This page provides a detailed showcase of the ECx-SyS model for experiments with two strong co-stressors. The ECx-SyS model adjusts the predictions of the SAM model to account for system stress.

For further information or citations, see:

- [Liess, M., Henz, S. & Knillmann, S. Predicting low-concentration effects of pesticides. Sci Rep 9, 15248 (2019).](https://doi.org/10.1038/s41598-019-51645-4)

{% for experiment in experiments %}
## {{ experiment.title }}

{% for series in experiment.series %}
### {{ series.name }}

![{{ series.name }}](imgs/{{ series.img_path }})

{% endfor %}
{% endfor %}