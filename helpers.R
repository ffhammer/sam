library(stressaddition)

plot_stress_custom <- function(model,
                               which = c(
                                 "sys_tox",
                                 "sys_tox_observed",
                                 "sys_tox_env",
                                 "sys_tox_env_observed"
                               ),
                               show_legend = FALSE,
                               xlab = "concentration",
                               ylab = "stress",
                               main = NULL,
                               log_x_axis = False) {
  stopifnot(inherits(model, "ecxsys"))
  
  curve_names <- names(model$curves)
  valid_names <- c(
    curve_names[startsWith(curve_names, "stress") | startsWith(curve_names, "sys")],
    "sys_tox_observed"
  )
  if (model$with_env) {
    valid_names <- c(valid_names, "sys_tox_env_observed")
  }
  if ("all" %in% which) {
    which <- valid_names
  } else {
    which <- intersect(which, valid_names)
  }
  
  curves <- model$curves
  ticks <- log10_ticks(curves$concentration_for_plots)
  point_concentration <- c(
    curves$concentration_for_plots[1],
    model$args$concentration[-1]
  )
  
  if (is.null(which)) {
    ymax = 1
  } else {
    which_lines <- which[!endsWith(which, "observed")]
    if (length(which_lines) == 0) {
      ymax <- 1
    } else {
      ymax <- max(curves[, which_lines], 1, na.rm = TRUE)
      # No need to include the observed stress in the call to max() no
      # matter if those are in "which" or not because these vectors are
      # clamped to [0, 1] anyway.
    }
  }
  
  if (log_x_axis) {
    plot(
      NA,
      NA,
      xlim = range(curves$concentration_for_plots, na.rm = TRUE),
      ylim = c(0, ymax),
      xlab = xlab,
      ylab = ylab,
      main = main,
      xaxt = "n",
      yaxt = "n",
      bty = "L",
      log = "x"
    )
  } else {
     plot(
      NA,
      NA,
      xlim = range(curves$concentration_for_plots, na.rm = TRUE),
      ylim = c(0, ymax),
      xlab = xlab,
      ylab = ylab,
      main = main,
      xaxt = "n",
      yaxt = "n",
      bty = "L",
    )
  }
  
  # The lines are drawn in this order to ensure that dotted and dashed lines
  # are on top of solid lines for better visibility.
  if ("sys_tox_observed" %in% which) {
    points(
      point_concentration,
      model$sys_tox_observed,
      pch = 16,
      col = "steelblue3"
    )
  }
  if ("stress_tox_sys" %in% which) {
    lines(
      curves$concentration_for_plots,
      curves$stress_tox_sys,
      col = "blue"
    )
  }
  if ("stress_tox" %in% which) {
    lines(
      curves$concentration_for_plots,
      curves$stress_tox,
      col = "deepskyblue",
      lty = 2
    )
  }
  if ("sys_tox" %in% which) {
    lines(
      curves$concentration_for_plots,
      curves$sys_tox,
      col = "steelblue3"
    )
  }
  if (model$with_env) {
    if ("sys_tox_env_observed" %in% which) {
      points(
        point_concentration,
        model$sys_tox_env_observed,
        pch = 16,
        col = "violetred"
      )
    }
    if ("stress_tox_env_sys" %in% which) {
      lines(
        curves$concentration_for_plots,
        curves$stress_tox_env_sys,
        col = "red"
      )
    }
    if ("stress_env" %in% which) {
      lines(
        curves$concentration_for_plots,
        curves$stress_env,
        col = "forestgreen",
        lty = 3
      )
    }
    if ("stress_tox_env" %in% which) {
      lines(
        curves$concentration_for_plots,
        curves$stress_tox_env,
        col = "orange",
        lty = 2
      )
    }
    if ("sys_tox_env" %in% which) {
      lines(
        curves$concentration_for_plots,
        curves$sys_tox_env,
        col = "violetred"
      )
    }
  }
  
  # The setting of col = NA and col.ticks = par("fg") is to prevent ugly line
  # thickness issues when plotting as a png with type = "cairo" and at a low
  # resolution.
  axis(1, at = ticks$major, labels = ticks$major_labels,
       col = NA, col.ticks = par("fg"))
  axis(1, at = ticks$minor, labels = FALSE, tcl = -0.25,
       col = NA, col.ticks = par("fg"))
  plotrix::axis.break(1, breakpos = model$axis_break_conc)
  axis(2, col = NA, col.ticks = par("fg"), las = 1)
  
  if (show_legend) {
    legend_df <- data.frame(
      name = c( "sys_tox_observed", "stress_tox", "sys_tox",
                "stress_tox_sys", "sys_tox_env_observed", "stress_env",
                "stress_tox_env", "sys_tox_env", "stress_tox_env_sys"),
      text = c("sys (tox, observed)", "tox", "sys (tox)", "tox + sys",
               "sys (tox + env, observed)", "env", "tox + env",
               "sys (tox + env)", "tox + env + sys"),
      pch = c(16, NA, NA, NA, 16, NA, NA, NA, NA),
      lty = c(0, 2, 1, 1, 0, 3, 2, 1, 1),
      col = c("steelblue3", "deepskyblue", "steelblue3", "blue",
              "violetred", "forestgreen", "orange", "violetred", "red"),
      stringsAsFactors = FALSE
    )
    legend_df <- legend_df[legend_df$name %in% which, ]
    if (nrow(legend_df) > 0) {
      legend(
        "topright",
        legend = legend_df$text,
        pch = legend_df$pch,
        lty = legend_df$lty,
        col = legend_df$col
      )
    }
  }
  invisible(NULL)  # suppress all possible return values
}


plot_survival_custom <- function(model,
                                 which = c(
                                   "survival_tox",
                                   "survival_tox_sys",
                                   "survival_tox_observed",
                                   "survival_tox_env",
                                   "survival_tox_env_sys",
                                   "survival_tox_env_observed"
                                 ),
                                 show_legend = FALSE,
                                 xlab = "concentration",
                                 ylab = "survival",
                                 main = NULL,
                                 log_x_axis = FALSE) {
  stopifnot(inherits(model, "ecxsys"))
  
  curve_names <- names(model$curves)
  valid_names <- c(
    curve_names[startsWith(curve_names, "survival")],
    "survival_tox_observed"
  )
  if (model$with_env) {
    valid_names <- c(valid_names, "survival_tox_env_observed")
  }
  if ("all" %in% which) {
    which <- valid_names
  } else {
    which <- intersect(which, valid_names)
  }
  
  curves <- model$curves
  ticks <- log10_ticks(curves$concentration_for_plots)
  point_concentration <- c(
    curves$concentration_for_plots[1],
    model$args$concentration[-1]
  )
  
  if (log_x_axis){

  plot(
    NA,
    NA,
    xlim = range(curves$concentration_for_plots, na.rm = TRUE),
    ylim = c(0, model$args$survival_max),
    xlab = xlab,
    ylab = ylab,
    main = main,
    xaxt = "n",
    yaxt = "n",
    bty = "L",
    log = "x",
  )
  } else {
     plot(
    NA,
    NA,
    xlim = range(curves$concentration_for_plots, na.rm = TRUE),
    ylim = c(0, model$args$survival_max),
    xlab = xlab,
    ylab = ylab,
    main = main,
    xaxt = "n",
    yaxt = "n",
    bty = "L",
  )
  }

  
  # The lines are drawn in this order to ensure that dotted and dashed lines
  # are on top of solid lines for better visibility.
  if ("survival_tox_observed" %in% which) {
    points(
      point_concentration,
      model$args$survival_tox_observed,
      pch = 16,
      col = "blue"
    )
  }
  if ("survival_tox_sys" %in% which) {
    lines(
      curves$concentration_for_plots,
      curves$survival_tox_sys,
      col = "blue"
    )
  }
  if ("survival_tox" %in% which) {
    lines(
      curves$concentration_for_plots,
      curves$survival_tox,
      col = "deepskyblue",
      lty = 2
    )
  }
  if ("survival_tox_LL5" %in% which) {
    lines(
      curves$concentration_for_plots,
      curves$survival_tox_LL5,
      col = "darkblue",
      lty = 3
    )
  }
  if (model$with_env) {
    if ("survival_tox_env_observed" %in% which) {
      points(
        point_concentration,
        model$args$survival_tox_env_observed,
        pch = 16,
        col = "red"
      )
    }
    if ("survival_tox_env_sys" %in% which) {
      lines(
        curves$concentration_for_plots,
        curves$survival_tox_env_sys,
        col = "red"
      )
    }
    if ("survival_tox_env" %in% which) {
      lines(
        curves$concentration_for_plots,
        curves$survival_tox_env,
        col = "orange",
        lty = 2
      )
    }
    if ("survival_tox_env_LL5" %in% which) {
      lines(
        curves$concentration_for_plots,
        curves$survival_tox_env_LL5,
        col = "darkred",
        lty = 3
      )
    }
  }
  
  # The setting of col = NA and col.ticks = par("fg") is to prevent ugly line
  # thickness issues when plotting as a png with type = "cairo" and at a low
  # resolution.
  axis(1, at = ticks$major, labels = ticks$major_labels,
       col = NA, col.ticks = par("fg"))
  axis(1, at = ticks$minor, labels = FALSE, tcl = -0.25,
       col = NA, col.ticks = par("fg"))
  plotrix::axis.break(1, breakpos = model$axis_break_conc)
  axis(2, col = NA, col.ticks = par("fg"), las = 1)
  
  if (show_legend) {
    legend_df <- data.frame(
      name = c("survival_tox_observed", "survival_tox", "survival_tox_sys",
               "survival_tox_LL5", "survival_tox_env_observed",
               "survival_tox_env", "survival_tox_env_sys", "survival_tox_env_LL5"),
      text = c("tox (observed)", "tox", "tox + sys", "tox (LL5)",
               "tox + env (observed)", "tox + env", "tox + env + sys",
               "tox + env (LL5)"),
      pch = c(16, NA, NA, NA, 16, NA, NA, NA),
      lty = c(0, 2, 1, 3, 0, 2, 1, 3),
      col = c("blue", "deepskyblue", "blue", "darkblue", "red", "orange",
              "red", "darkred"),
      stringsAsFactors = FALSE
    )
    legend_df <- legend_df[legend_df$name %in% which, ]
    if (nrow(legend_df) > 0) {
      legend(
        "topright",
        legend = legend_df$text,
        pch = legend_df$pch,
        lty = legend_df$lty,
        col = legend_df$col
      )
    }
  }
  invisible(NULL)  # suppress all possible return values
}

