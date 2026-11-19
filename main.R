#!/usr/bin/env Rscript

source(file.path("R", "data_loader.R"))
source(file.path("R", "eda.R"))
source(file.path("R", "models.R"))
source(file.path("R", "evaluation.R"))

SEED <- 123L
TRAIN_RATIO <- 0.70
EXCLUDE_DURATION <- TRUE
ASSET_DIR <- "assets"

message("[1/5] Loading and validating full UCI Bank Marketing dataset...")
bank_df <- load_bank_marketing_data(exclude_duration = EXCLUDE_DURATION)
message(sprintf(
  "Loaded %d rows x %d predictors | Positive class ('yes') rate: %.2f%%",
  nrow(bank_df), ncol(bank_df) - 1L, 100 * mean(bank_df$y == "yes")
))

message("[2/5] Exporting exploratory data analysis figures...")
generate_eda_plots(bank_df, output_dir = ASSET_DIR)

splits <- split_train_test(bank_df, train_ratio = TRAIN_RATIO, seed = SEED)
train_df <- splits$train
test_df  <- splits$test

message("[3/5] Training CART, Logistic Regression, randomForest, and ranger...")
cart_model   <- train_pruned_cart(train_df, seed = SEED, asset_dir = ASSET_DIR)
logit_model  <- train_logistic_regression(train_df)
rf_model     <- train_random_forest(train_df, ntree = 500L, seed = SEED, asset_dir = ASSET_DIR)
ranger_model <- train_ranger_forest(train_df, num_trees = 500L, mtry = rf_model$mtry, seed = SEED)

message("[4/5] Computing test-set probabilities and ROC metrics...")
probs <- list(
  "Pruned CART"         = predict(cart_model, newdata = test_df, type = "prob")[, "yes"],
  "Logistic Regression" = predict(logit_model, newdata = test_df, type = "response"),
  "Random Forest (RF)"  = predict(rf_model, newdata = test_df, type = "prob")[, "yes"],
  "Random Forest (Ranger)" = predict(ranger_model, data = test_df)$predictions[, "yes"]
)

eval_results <- lapply(names(probs), function(nm) {
  evaluate_binary_model(y_true = test_df$y, y_prob = probs[[nm]], model_name = nm)
})
names(eval_results) <- names(probs)

comparison_table <- do.call(rbind, lapply(eval_results, `[[`, "metrics"))
roc_objects <- lapply(eval_results, `[[`, "roc")

message("[5/5] Exporting benchmark summary and ROC comparison plot...")
plot_roc_comparison(roc_objects, output_path = file.path(ASSET_DIR, "roc_curves_comparison.png"))
write.csv(comparison_table, file = file.path(ASSET_DIR, "model_comparison.csv"), row.names = FALSE)

print(comparison_table, row.names = FALSE)
