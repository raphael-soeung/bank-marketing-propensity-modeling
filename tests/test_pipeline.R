library(testthat)

source(file.path("..", "R", "data_loader.R"))

test_that("Dataset loader enforces leakage removal and factor coercion", {
  skip_if_not(file.exists(file.path("..", "data", "bank-full.csv")), "Local bank-full.csv not cached")
  df <- load_bank_marketing_data(file.path("..", "data", "bank-full.csv"), exclude_duration = TRUE)

  expect_equal(nrow(df), 45211L)
  expect_false("duration" %in% names(df))
  expect_true(is.factor(df$y))
  expect_equal(levels(df$y), c("no", "yes"))
  expect_false(any(vapply(df, is.character, logical(1))))
})

test_that("Train/test split preserves exact row count without overlap", {
  dummy <- data.frame(
    x = rnorm(100),
    y = factor(rep(c("no", "yes"), times = c(88, 12)), levels = c("no", "yes"))
  )
  sp <- split_train_test(dummy, train_ratio = 0.70, seed = 42L)
  expect_equal(nrow(sp$train) + nrow(sp$test), 100L)
})
