library(dplyr)
library(readxl)
library(tidyr)
library(purrr)
library(stringr)

# Source the prediction functions
source("SAM-R-v05-08-2024/SAM.R")

# Define function to read and structure experiment data
read_data <- function(file_path) {
  data <- read_excel(file_path)
  
  # Filter out unnamed columns
  data <- data[, !grepl("^\\.\\.\\.|Unnamed", names(data))]
  data <- data[, colSums(is.na(data)) != nrow(data)]  # Remove columns that are all NA

  # Assuming 'concentration' and 'survival' are always the first two columns
  # and there's an expected structure to the data
  if (!all(c("concentration", "survival") %in% names(data))) {
    stop("Expected 'concentration' and 'survival' as the first two columns")
  }

  
  # Assuming 'concentration' and 'survival' are always the first two columns
  main_series <- list(
    concentration = data[["concentration"]],
    survival_rate = data[["survival"]],
    name = "Main"
  )
  
  # Additional stressors
  stressor_cols <- setdiff(names(data), c("concentration", "survival", "meta_category", "info"))
  additional_stresses <- lapply(stressor_cols, function(col) {
    list(
      concentration = data[["concentration"]],
      survival_rate = data[[col]],
      name = col
    )
  })
  
  names(additional_stresses) <- stressor_cols
  
  list(main_series = main_series, additional_stress = additional_stresses)
}

# Define the transformation and curve fitting options
curve_fitting_options <- c('lmcurve')
transform_options <- c('williams_and_linear_interpolation') #'none', 'linear_interpolation', 'williams',

# Define a more robust clean path function to sanitize file names
clean_path <- function(x) {
  file_name <- tools::file_path_sans_ext(basename(x))
  file_name
}

# Ensure the output directory exists before attempting to save files
output_dir <- "migration/r"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Iterate over each file and configuration
file_paths <- list.files(path = "data", pattern = "\\.xlsx$", full.names = TRUE)
for (file in file_paths) {
  experiment_data <- read_data(file)
  
  for (curve_fitting in curve_fitting_options) {
    for (transform in transform_options) {
      
      for (stress_name in names(experiment_data$additional_stress)) {
        stress <- experiment_data$additional_stress[[stress_name]]
        
        # Use tryCatch to handle potential errors during prediction
        result <- tryCatch({
          get_prediction("some name", experiment_data$main_series$concentration, experiment_data$main_series$survival_rate, stress$survival_rate, transform)
        }, error = function(e) {
          message(sprintf("Error in processing %s with %s using %s transformation: %s", stress_name, curve_fitting, transform, e$message))
          NULL  # Explicitly return NULL on error
        })
        
        # Check if result is not NULL before saving to CSV
        if (!is.null(result)) {
          output_file_name <- sprintf("%s_%s_%s_%s.csv", clean_path(file), stress_name, transform, curve_fitting)
          output_path <- file.path(output_dir, output_file_name)
          if (!dir.exists(dirname(output_path))) {
            dir.create(dirname(output_path), recursive = TRUE)
          }
          write.csv(result, output_path, row.names = FALSE)
          message(sprintf("Processed %s with %s using %s transformation. Results saved to %s", stress_name, curve_fitting, transform, output_path))
        }
      }
    }
  }
}