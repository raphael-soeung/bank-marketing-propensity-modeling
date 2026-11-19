#' Load and preprocess the full UCI Bank Marketing dataset (bank-full.csv).
#'
#' Automatically downloads and extracts `bank-full.csv` from the UCI Machine
#' Learning Repository if not present locally, validates schema integrity,
#' removes post-hoc target leakage (`duration`), and casts character predictors
#' to factors for tree-based and ensemble models.
#'
#' @param data_path Path to `bank-full.csv`.
#' @param exclude_duration Logical; if TRUE (default), drops `duration` to prevent
#'   post-call target leakage in pre-call propensity modeling.
#' @return A cleaned data.frame ready for stratified splitting.
load_bank_marketing_data <- function(
  data_path = file.path("data", "bank-full.csv"),
  exclude_duration = TRUE
) {
  if (!file.exists(data_path)) {
    dir.create(dirname(data_path), showWarnings = FALSE, recursive = TRUE)
    zip_url <- "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
    outer_zip <- tempfile(fileext = ".zip")
    inner_dir <- tempfile()

    message("Downloading UCI Bank Marketing dataset...")
    download.file(zip_url, destfile = outer_zip, mode = "wb", quiet = TRUE)
    unzip(outer_zip, exdir = inner_dir)

    nested_zip <- file.path(inner_dir, "bank.zip")
    if (file.exists(nested_zip)) {
      unzip(nested_zip, files = "bank-full.csv", exdir = dirname(data_path))
    } else {
      csv_found <- list.files(inner_dir, pattern = "bank-full\\.csv$", recursive = TRUE, full.names = TRUE)
      if (length(csv_found) == 0) {
        stop("Failed to locate bank-full.csv inside the downloaded UCI archive.")
      }
      file.copy(csv_found[1], data_path)
    }
  }

  df <- read.csv(data_path, header = TRUE, sep = ";", stringsAsFactors = FALSE)

  if (!"y" %in% names(df)) {
    stop("Expected target column 'y' not found in dataset.")
  }
  if (anyNA(df)) {
    stop("Unexpected NA values detected in dataset.")
  }

  if (exclude_duration && "duration" %in% names(df)) {
    df$duration <- NULL
  }

  # Explicitly coerce character columns to factors (required for randomForest in R >= 4.0)
  char_cols <- vapply(df, is.character, logical(1))
  df[char_cols] <- lapply(df[char_cols], as.factor)
  df$y <- factor(df$y, levels = c("no", "yes"))

  return(df)
}

#' Split dataset into deterministic training and test partitions.
#'
#' @param df Preprocessed data.frame containing target `y`.
#' @param train_ratio Proportion of rows allocated to training (default 0.70).
#' @param seed Integer random seed for reproducibility.
#' @return A list with `train` and `test` data.frames.
split_train_test <- function(df, train_ratio = 0.70, seed = 123L) {
  set.seed(seed)
  n_obs <- nrow(df)
  train_idx <- sample(seq_len(n_obs), size = round(train_ratio * n_obs))

  list(
    train = df[train_idx, , drop = FALSE],
    test  = df[-train_idx, , drop = FALSE]
  )
}
