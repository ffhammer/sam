Data Handling
=============

For trying out your own data quickly, it's best to use the :class:`sam.DoseResponseSeries` class, as it is easy and quick.

.. automodule:: sam
   :members: DoseResponseSeries,
   :undoc-members:

Our Data Format
-------------
For our work, we use a simple data format. It can be a bit confusing; it's best to have a look at our data folder in our repo.

The structure is:

.. code-block:: text

   data/
   ├── experiment1/
   │   ├── meta.yaml
   │   ├── days_2.xlsx
   │   └── days_14.xlsx
   └── experiment2/
       ├── meta.yaml
       ├── reference.xlsx
       └── agricultural.xlsx

Each experiment has a dedicated folder with specific measurements, such as those for different time durations or different populations (e.g., agricultural or reference populations). We work with .xlsx files since they are easy to read, write, and inspect.


THe .xlsx files follow this template:

.. include:: data_template_table.rst

The `concentration` and `survival` columns correspond to concentration amounts and survival rates, respectively, for the main stressor. 

Metadata attributes and their values, as specified in ExperimentMetaData, can be included either directly in the `.xlsx` file (with `meta_category` as the attribute name and `info` as the value) or in a `meta.yaml` file within the experiment directory. Important: Each attribute key should appear in either meta.yaml or the .xlsx file—duplicates across files are not allowed.

.. automodule:: sam
:members: ExperimentMetaData,
:undoc-members:

Loading Data
-------------

Once data is in the correct format, you can create an instance of :class:`sam.ExperimentData`:

.. automodule:: sam
   :members: ExperimentData,
   :undoc-members:

**Note**: The term "experiment" can refer to a folder containing multiple `.xlsx` files (e.g., for different durations) and may contain multiple `ExperimentData` instances.
