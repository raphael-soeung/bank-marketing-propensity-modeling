required_packages <- c(
  "rpart",
  "rpart.plot",
  "caret",
  "pROC",
  "randomForest",
  "ranger",
  "testthat"
)

installed <- rownames(installed.packages())
missing_pkgs <- setdiff(required_packages, installed)

if (length(missing_pkgs) > 0) {
  message("Installing missing CRAN packages: ", paste(missing_pkgs, collapse = ", "))
  install.packages(missing_pkgs, repos = "https://cloud.r-project.org")
} else {
  message("All required packages are already installed.")
}
