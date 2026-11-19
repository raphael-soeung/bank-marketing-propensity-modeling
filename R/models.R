library(rpart)
library(rpart.plot)
library(randomForest)
library(ranger)

#' Fit and cost-complexity prune a CART decision tree via 10-fold CV.
#'
#' @param train_df Training data.frame with target `y`.
#' @param seed Random seed for internal cross-validation splits.
#' @param asset_dir Directory to export the pruned tree diagram.
#' @return Fitted and pruned `rpart` model object.
train_pruned_cart <- function(train_df, seed = 123L, asset_dir = "assets") {
  set.seed(seed)
  full_tree <- rpart(y ~ ., data = train_df, method = "class", cp = 0.0005)

  cp_table <- full_tree$cptable
  best_row <- which.min(cp_table[, "xerror"])
  optimal_cp <- cp_table[best_row, "CP"]

  pruned_tree <- prune(full_tree, cp = optimal_cp)

  dir.create(asset_dir, showWarnings = FALSE, recursive = TRUE)
  png(file.path(asset_dir, "pruned_decision_tree.png"), width = 1400, height = 900, res = 130)
  rpart.plot(
    pruned_tree,
    type = 2,
    extra = 104,
    fallen.leaves = TRUE,
    main = sprintf("Cost-Complexity Pruned CART (optimal cp = %.6f)", optimal_cp)
  )
  dev.off()

  return(pruned_tree)
}

#' Fit a baseline binomial Logistic Regression model.
#'
#' @param train_df Training data.frame with target `y`.
#' @return Fitted `glm` object.
train_logistic_regression <- function(train_df) {
  glm(y ~ ., data = train_df, family = binomial(link = "logit"))
}

#' Tune `mtry` via OOB error and fit a `randomForest` classifier.
#'
#' @param train_df Training data.frame with target `y`.
#' @param ntree Number of trees in the ensemble (default 500).
#' @param seed Random seed for reproducibility.
#' @param asset_dir Directory to export variable importance plot.
#' @return Fitted `randomForest` object.
train_random_forest <- function(train_df, ntree = 500L, seed = 123L, asset_dir = "assets") {
  set.seed(seed)
  x_train <- train_df[, setdiff(names(train_df), "y"), drop = FALSE]
  y_train <- train_df$y

  tuned <- tuneRF(
    x = x_train,
    y = y_train,
    stepFactor = 1.5,
    improve = 0.01,
    trace = FALSE,
    plot = FALSE
  )
  best_mtry <- tuned[which.min(tuned[, "OOBError"]), "mtry"]

  set.seed(seed)
  rf_model <- randomForest(
    y ~ .,
    data = train_df,
    ntree = ntree,
    mtry = best_mtry,
    importance = TRUE
  )

  dir.create(asset_dir, showWarnings = FALSE, recursive = TRUE)
  png(file.path(asset_dir, "rf_variable_importance.png"), width = 1100, height = 700, res = 120)
  varImpPlot(rf_model, main = sprintf("Random Forest Feature Importance (mtry = %d)", best_mtry))
  dev.off()

  return(rf_model)
}

#' Fit a fast C++ probability Random Forest using `ranger`.
#'
#' @param train_df Training data.frame with target `y`.
#' @param num_trees Number of trees (default 500).
#' @param mtry Number of candidate variables per split.
#' @param seed Random seed for reproducibility.
#' @return Fitted `ranger` probability model object.
train_ranger_forest <- function(train_df, num_trees = 500L, mtry = 3L, seed = 123L) {
  ranger(
    formula = y ~ .,
    data = train_df,
    num.trees = num_trees,
    mtry = mtry,
    probability = TRUE,
    importance = "impurity",
    seed = seed
  )
}
