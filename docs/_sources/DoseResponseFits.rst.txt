Dose-Response Fitting
=====================


Steps
-----

1. **Prepare Dose-Response Data**: Start with an instance of `DoseResponseSeries`, containing concentration and survival data for the main stressor.
2. **Configure Settings**: Customize the fit by specifying `DRF_Settings`
3. **Run the Fit**: Call `dose_response_fit`. The function returns `ModelPredictions`, giving you survival and stress curves, predicted survival values, and lethal concentration metrics.

Classes and Functions
---------------------

Below are the primary classes and functions used for dose-response fitting.

Dose Response Fit Settings
--------------------------

.. autoclass:: sam.DRF_Settings
   :members:
   :undoc-members:
   :show-inheritance:
Model Predictions
-----------------

.. autoclass:: sam.ModelPredictions
   :members:
   :undoc-members:
   :show-inheritance:

Dose Response Fit Function
--------------------------

.. autofunction:: sam.dose_response_fit

Example Usage
-------------

Here’s how to use `dose_response_fit` with sample data and custom settings:


.. code-block:: python

   from sam import dose_response_fit, DRF_Settings, DoseResponseSeries

   # Sample dose-response data
   data = DoseResponseSeries(concentration=[0, 0.5, 1.0, 5.0], survival_rate=[100, 90, 75, 20], name = "example")

   # Custom settings (optional)
   cfg = DRF_Settings(max_survival=100)

   # Perform dose-response fit
   predictions = dose_response_fit(data, cfg)

   # Access the model predictions
   print(predictions.survival_curve)