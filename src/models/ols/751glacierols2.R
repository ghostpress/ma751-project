### 751, Statistical Machine Learning
### Loc, Lucia, Firas
### April 28th, 2025

# Fitting an OLS model for the glacier data
# Packages
# install.packages("leaps")
# install.packages("car")
library(leaps)
library(car)

# Data upload
glacierdata <- read.csv("glacier.csv")

# Variables for OLS: 12 rain, 12 temp, PDD, area, lat, 
# rain, slope, snow, zmax, zmin, zmed 
# note: icecap not included because all values = 0
# note: snow removed for perfect multicolinearity
predictors <- c("PDD", "area", "lat", "prcp_01", "prcp_02", 
                "prcp_03", "prcp_04", "prcp_05", "prcp_06", "prcp_07", 
                "prcp_08", "prcp_09", "prcp_10", "prcp_11", "prcp_12", 
                "rain", "slope", "temp_01", "temp_02", "temp_03", 
                "temp_04", "temp_05", "temp_06", "temp_07", "temp_08", 
                "temp_09", "temp_10", "temp_11", "temp_12", "zmax", 
                "zmed", "zmin")

# Building design matrix X for OLS
X <- glacierdata[, predictors]

# Standardizing predictors
Xscale <- scale(X)

# Adding column of 1s, final design matrix
Xdesign <- cbind(beta0 = 1, Xscale)

# Y variable (SMB, called dmdtda in data)
smb <- glacierdata$dmdtda

# OLS model
glacierOLS <- lm(smb ~ ., data = as.data.frame(Xdesign))
summary(glacierOLS)

# Best subset (method used in paper)
dataXY <- data.frame(smb = smb, Xscale)
subset_fit <- regsubsets(smb ~ ., data = dataXY, nvmax = 32, method = "exhaustive")
subset_summary <- summary(subset_fit)

# Fitting the best models at each subset size and looking at their Radj^2
# Store good models
good_models <- list()
adjr2_list <- c()

# Loop over models of sizes 1 to 29
for (k in 1:29) {
  vars <- names(coef(subset_fit, k))
  vars <- vars[vars != "(Intercept)"]
  formula_k <- as.formula(paste("smb ~", paste(vars, collapse = " + ")))
  model_k <- lm(formula_k, data = dataXY)
  
  pvals <- summary(model_k)$coefficients[, 4]
  vif_vals <- if (length(vars) > 1) vif(model_k) else 1
  
  if (all(pvals[-1] < 0.5) && all(vif_vals < 100)) {
    adjr2 <- summary(model_k)$adj.r.squared
    
    # Use a numeric index and store the model's subset size
    model_label <- paste0("size_", length(vars))
    
    good_models[[model_label]] <- list(
      formula = formula_k,
      model = model_k,
      adjr2 = adjr2
    )
    
    adjr2_list <- c(adjr2_list, adjr2)
  }
}

# Now sort using names of good_models
sorted_indices <- order(adjr2_list, decreasing = TRUE)
sorted_names <- names(good_models)[sorted_indices]
top_n <- min(3, length(sorted_names))
top_models <- good_models[sorted_names[1:top_n]]

# Display nicely
for (i in seq_along(top_models)) {
  cat("\n====================\n")
  cat("Top Model:", sorted_names[i], "\n")
  print(top_models[[i]]$formula)
  cat("Adjusted R^2:", top_models[[i]]$adjr2, "\n")
}

model_11 <- top_models[[1]]$model
model_10 <- top_models[[2]]$model
model_9 <- top_models[[3]]$model
plot(model_11)
plot(model_10)
plot(model_9)
summary(model_11)
summary(model_10)
summary(model_9)
vif(model_11)
vif(model_10)
vif(model_9)
