library(caret)
library(pROC)

#' Compute classification metrics at a given probability cutoff and Youden's optimal threshold.
#'
#' @param y_true Factor vector of true outcomes with levels `c("no", "yes")`.
#' @param y_prob Numeric vector of predicted probabilities for the positive class `"yes"`.
#' @param model_name Character label identifying the model.
#' @param default_cutoff Probability threshold for standard classification (default 0.50).
#' @return A list containing the `roc` object and a 1-row metrics `data.frame`.
evaluate_binary_model <- function(y_true, y_prob, model_name, default_cutoff = 0.50) {
  roc_obj <- roc(response = y_true, predictor = y_prob, levels = c("no", "yes"), direction = "<", quiet = TRUE)
  auc_val <- as.numeric(auc(roc_obj))

  # Standard 0.50 threshold metrics
  pred_std <- factor(ifelse(y_prob >= default_cutoff, "yes", "no"), levels = c("no", "yes"))
  cm_std <- confusionMatrix(data = pred_std, reference = y_true, positive = "yes")

  # Youden's J optimal threshold (maximizes Sensitivity + Specificity - 1)
  best_coords <- coords(roc_obj, x = "best", best.method = "youden", ret = c("threshold", "sensitivity", "specificity"))
  opt_cutoff <- as.numeric(best_coords$threshold[1])

  pred_opt <- factor(ifelse(y_prob >= opt_cutoff, "yes", "no"), levels = c("no", "yes"))
  cm_opt <- confusionMatrix(data = pred_opt, reference = y_true, positive = "yes")

  summary_row <- data.frame(
    Model             = model_name,
    ROC_AUC           = round(auc_val, 4),
    Accuracy_050      = round(as.numeric(cm_std$overall["Accuracy"]), 4),
    NIR               = round(as.numeric(cm_std$overall["AccuracyNull"]), 4),
    Precision_050     = round(as.numeric(cm_std$byClass["Pos Pred Value"]), 4),
    Recall_050        = round(as.numeric(cm_std$byClass["Sensitivity"]), 4),
    Specificity_050   = round(as.numeric(cm_std$byClass["Specificity"]), 4),
    Youden_Threshold  = round(opt_cutoff, 4),
    Recall_Youden     = round(as.numeric(cm_opt$byClass["Sensitivity"]), 4),
    Specificity_Youden = round(as.numeric(cm_opt$byClass["Specificity"]), 4),
    Precision_Youden  = round(as.numeric(cm_opt$byClass["Pos Pred Value"]), 4),
    stringsAsFactors  = FALSE
  )

  list(roc = roc_obj, metrics = summary_row)
}

#' Plot overlaid ROC curves for all evaluated models.
#'
#' @param roc_list Named list of `roc` objects.
#' @param output_path Destination PNG file path.
plot_roc_comparison <- function(roc_list, output_path = file.path("assets", "roc_curves_comparison.png")) {
  dir.create(dirname(output_path), showWarnings = FALSE, recursive = TRUE)
  palette <- c("#718096", "#3182CE", "#2F855A", "#D69E2E")

  png(output_path, width = 1000, height = 750, res = 120)
  plot(
    roc_list[[1]],
    col = palette[1],
    lwd = 2.5,
    main = "ROC Curve Comparison on Test Set (UCI Bank Marketing, N = 45,211)"
  )
  if (length(roc_list) > 1) {
    for (i in 2:length(roc_list)) {
      plot(roc_list[[i]], col = palette[(i - 1) %% length(palette) + 1], lwd = 2.5, add = TRUE)
    }
  }

  legend_labels <- vapply(names(roc_list), function(nm) {
    sprintf("%s (AUC = %.4f)", nm, as.numeric(auc(roc_list[[nm]])))
  }, character(1))

  legend("bottomright", legend = legend_labels, col = palette[seq_along(roc_list)], lwd = 2.5, bty = "n")
  dev.off()
}
