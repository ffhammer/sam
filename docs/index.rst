SAM Documentation
=================

Welcome to the documentation for the SAM project. Below are the main sections:

Welcome to the documentation for the SAM project. Below are the main sections:

.. toctree::
   :maxdepth: 2
   :caption: Contents

   DataHandling
   DoseResponseFits
   sam
   utils


Example
-------

.. code-block:: python

   from sam import * 
   import matplotlib.pyplot as plt

   # Example DoseResponseSeries data
   control_series = DoseResponseSeries(
      concentration=[0, 0.1, 0.5, 1.0, 5.0],
      survival_rate=[100, 98, 85, 50, 10],
      name="Control"
   )

   stressor_series = DoseResponseSeries(
      concentration=[0, 0.1, 0.5, 1.0, 5.0],
      survival_rate=[100, 95, 70, 30, 5],
      name="Stressor"
   )


   # Run SAM prediction
   main_fit, stress_fit, sam_sur, sam_stress, additional_stress = sam_prediction(
      main_series=control_series,
      stressor_series=stressor_series,
      settings=STANDARD_SAM_SETTING,
      max_survival=100,
   )


   # Plot results
   fig = plot_sam_prediction(
      main_fit,
      stress_fit,
      sam_sur,
      sam_stress,
      survival_max=100,
      title="SAM Prediction Example"
   )

   plt.show()