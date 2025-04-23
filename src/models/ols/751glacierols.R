### 751, Statistical Machine Learning
### Loc, Lucia, Firas
### April 28th, 2025

# Fitting an OLS model for the glacier data
# Packages
install.packages("leaps")

install.packages("car")
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
  
  # Get p-values
  pvals <- summary(model_k)$coefficients[, 4]
  
  # Get VIFs (skip if model is too small)
  if (length(vars) > 1) {
    vif_vals <- vif(model_k)
  } else {
    vif_vals <- 1  # single-variable model: no VIF issue
  }
  
  # Apply filters
  if (all(pvals[-1] < 0.01) && all(vif_vals < 1.2)) {
    adjr2 <- summary(model_k)$adj.r.squared
    good_models[[paste("Model", k)]] <- list(
      formula = formula_k,
      model = model_k,
      adjr2 = adjr2
    )
    adjr2_list <- c(adjr2_list, adjr2)
  }
}

# Sort models by adjusted R^2
# sorted_indices <- order(adjr2_list, decreasing = TRUE)
# top_models <- good_models[sorted_indices]

# Get names of top models
sorted_names <- names(good_models)[sorted_indices]

# Pull top 3 models by name
top_models <- good_models[sorted_names[1:3]]

# Add some diagnostic plots


























