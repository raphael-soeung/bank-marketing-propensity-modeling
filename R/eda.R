#' Export exploratory data analysis figures to the assets directory.
#'
#' @param df Cleaned data.frame with factor target `y`.
#' @param output_dir Directory where PNG plots will be saved.
generate_eda_plots <- function(df, output_dir = "assets") {
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

  # 1. Balance distribution by subscription outcome (log-modulus / clipped view)
  png(file.path(output_dir, "eda_balance_boxplot.png"), width = 900, height = 600, res = 120)
  boxplot(
    balance ~ y,
    data = df,
    outline = FALSE,
    col = c("#D9E2EC", "#2B6CB0"),
    main = "Account Balance Distribution by Subscription Status (Outliers Hidden)",
    xlab = "Term Deposit Subscription (y)",
    ylab = "Average Yearly Balance (EUR)"
  )
  dev.off()

  # 2. Subscription rate across education levels
  png(file.path(output_dir, "eda_education_mosaic.png"), width = 950, height = 600, res = 120)
  plot(
    table(df$education, df$y),
    col = c("#CBD5E0", "#2B6CB0"),
    main = "Term Deposit Subscription Share by Education Level",
    xlab = "Education Level",
    ylab = "Subscription Outcome"
  )
  dev.off()

  # 3. Subscription rate by prior campaign outcome (poutcome)
  png(file.path(output_dir, "eda_poutcome_mosaic.png"), width = 950, height = 600, res = 120)
  plot(
    table(df$poutcome, df$y),
    col = c("#CBD5E0", "#2B6CB0"),
    main = "Subscription Share by Previous Campaign Outcome",
    xlab = "Previous Campaign Outcome (poutcome)",
    ylab = "Subscription Outcome"
  )
  dev.off()
}
