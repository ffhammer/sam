# Visualizing Dose-Response Fits Across Experiments

This section visualizes the dose-response fits for all experiments. To enable effective comparisons, we have projected each curve into a "normalized" scale, where lethal concentrations (LC) range from LC1 to LC99.

### Raw Dose-Response Curves
In the "raw" visualization, we applied the `sam.dose_response_fit` method directly to each experiment without any adjustments. This view shows the unprocessed fits as provided by SAM.

> **Tip**: Click the "ALL" button in the viewer for optimal visualization.
<iframe src="imgs/dose_response_curves/raw_dosecurves.html" width="100%"height="600px" style="max-width: 1200px;"></iframe>

### Cleaned Dose-Response Curves
For the "cleaned curves," we adjusted each dose-response curve by removing subhormesis points (system inherent stress effects) and setting the control survival rate to 100%. This provides a clearer view of the dose-response relationship in isolation from inherent stress factors.

<iframe src="imgs/dose_response_curves/cleaned_dosecurves.html" width="100%"height="600px" style="max-width: 1200px;"></iframe>
