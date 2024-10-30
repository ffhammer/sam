from sam.data_formats import read_data, DoseResponseSeries
from sam.dose_reponse_fit import dose_response_fit, ModelPredictions
from sam import REPO_PATH
import yaml
from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt

different_days = []

for file in glob("data/*/meta.yaml"):
    
    
    with open(file, "r") as f:
        meta = yaml.safe_load(f)

    different_days.append(Path(file).parent)
    
print(different_days)


def difference_plots(path):
    series : list[DoseResponseSeries]= [read_data(s).main_series for s in glob( str(path / "*.xlsx"))]
    
    days = {i.meta.days for i in series}
    if len(days) == 1:
        return
    
    plt.figure()

    color_map = {}  # To store the unique colors for each 'days'
    
    # Plot each series, and store the color for each day
    for s in series:
        scatter = plt.scatter(s.concentration, s.survival_rate, label=f'{s.meta.days} (scatter)')
        
        # Retrieve the color assigned by matplotlib
        color = scatter.get_facecolor()[0]
        
        if s.meta.days not in color_map:
            color_map[s.meta.days] = color  # Map the days to their assigned color
        
        fit: ModelPredictions = dose_response_fit(s)
        plt.plot(fit.concentrations, fit.survival_curve, label=f'{s.meta.days} (fit)', color=color)
    
    # Create custom legend elements using Line2D for each unique day with the respective colors
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[day], markersize=10, label=f'{day} days')
                       for day in color_map]
    
    # Add the custom legend
    plt.legend(handles=legend_elements)
    
    plt.title(path.name)
    plt.savefig(f"control_imgs/difference_of_days/{path.name}.png")
    plt.close()
            

for p in different_days:
    difference_plots(p)