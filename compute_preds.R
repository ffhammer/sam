library(stressaddition)

source("helpers.R")

plot_survival_with_data <- function(model, data, filename, log = FALSE) {
  png(filename, width = 800, height = 400)
  par(mfrow = c(1, 2))  # Layout for two plots side by side
  
  # First plot: Survival curve
  plot_survival_custom(model, show_legend = TRUE, which = "survival_tox", log_x_axis = log)
  points(data$conc, data$no.stress, pch = 19)
  if (!is.null(data$hormesis_concentration[1]) && data$hormesis_concentration[1] > 0) {
    points(data$hormesis_concentration[1], data$no.stress[which(data$conc == data$hormesis_concentration[1])], col = "red", pch = 19)
  }
  title <- paste(data$chemical[1], data$organism[1], sep = " - ")
  mtext(title, side = 3, line = 0.5, cex = 1.2)
  
  # Second plot: Stress curve
  plot_stress_custom(model, show_legend = TRUE, which = "stress_tox", log_x_axis = log)
  
  dev.off()
}

# Loop over all CSV files in the directory
files <- list.files(path = "formatted_data", pattern = "*.csv", full.names = TRUE)
for (file in files) {
  data <- try(read.csv(file), silent = TRUE)
  if (inherits(data, "try-error")) {
    print(paste("Error reading file:", file))
    next
  } 

  base_name <- tools::file_path_sans_ext(basename(file))
  filename_log <- paste0("plots/", base_name, "_log.png")
  filename_linear <- paste0("plots/", base_name, "_linear.png")
  filename_preds <- paste0("r_preds/", base_name, ".csv")

  tryCatch({
    model <- ecxsys(
      concentration = data$conc,
      hormesis_concentration = data$hormesis_concentration[1],
      survival_tox_observed = data$no.stress
    )

    plot_survival_with_data(model, data, filename_log, log = TRUE)
    plot_survival_with_data(model, data, filename_linear, log = FALSE)

    # Save predictions to CSV
    curves <- model$curves
    predictions <- data.frame(
      concentration_for_plots = curves$concentration_for_plots,
      stress_tox = curves$stress_tox,
      survival_tox = curves$survival_tox,
      survival_tox_LL5 = curves$survival_tox_LL5
    )
    write.csv(predictions, file = filename_preds, row.names = FALSE)

  }, error = function(e) {
    print(paste("Error plotting data for file:", file, "with error message:", e$message))
  })
}
