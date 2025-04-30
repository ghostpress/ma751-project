# 751 Final Project
# Decision tree regression
# Loading data
glactrain <- read.csv("glaciertrain.csv") # Data set with all predictors
glactrainy <- read.csv("glaciertrainy.csv") # Data set with SMB

# Merging data into one data frame
myglacdata <- cbind(glactrainy, glactrain)
myglacdata <- myglacdata[, -c(1, 3, 4, 5)]

#install.packages("rpart")
#install.packages("rpart.plot")
#install.packages("gbm")
#install.packages("xgboost")
library(gbm)
library(rpart)
library(rpart.plot)
library(xgboost)

# Surface Mass Balance data
SMB <- myglacdata$dmdtda
myglacdata$SMB <- myglacdata$dmdtda

# Predictors 
pdd <- myglacdata$PDD
area <- myglacdata$area
lat <- myglacdata$lat
rain <- myglacdata$rain
slope <- myglacdata$slope
snow <- myglacdata$snow

zmax <- myglacdata$zmax
zmed <- myglacdata$zmed
zmin <- myglacdata$zmin

p1 <- myglacdata$prcp_01
p2 <- myglacdata$prcp_02
p3 <- myglacdata$prcp_03
p4 <- myglacdata$prcp_04
p5 <- myglacdata$prcp_05
p6 <- myglacdata$prcp_06
p7 <- myglacdata$prcp_07
p8 <- myglacdata$prcp_08
p9 <- myglacdata$prcp_09
p10 <- myglacdata$prcp_10
p11 <- myglacdata$prcp_11
p12 <- myglacdata$prcp_12

t1 <- myglacdata$temp_01
t2 <- myglacdata$temp_02
t3 <- myglacdata$temp_03
t4 <- myglacdata$temp_04
t5 <- myglacdata$temp_05
t6 <- myglacdata$temp_06
t7 <- myglacdata$temp_07
t8 <- myglacdata$temp_08
t9 <- myglacdata$temp_09
t10 <- myglacdata$temp_10
t11 <- myglacdata$temp_11
t12 <- myglacdata$temp_12

# Making my decision tree model
treemodel <- rpart(SMB ~ pdd + area + lat + rain + slope + snow +
                   zmax + zmin + zmed +
                   p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9 + p10 + p11 + p12+
                   t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 + t9 + t10 + t11 + t12,
                   data = myglacdata,
                   method = "anova",
                   control = rpart.control(maxdepth = 5, minsplit = 5, cp = 0))
# anova method for regression tree
# maxdepth prevents tree with more than 5 levels
# minsplit requires at least 5 data points for new branch

# Visualizing our decision tree
rpart.plot(treemodel)

# Making predictions on the same data set using our decision tree regression
predicted <- predict(treemodel, newdata = myglacdata)

# Noting my actual SMB values
actual <- SMB

# Plot actual vs predicted
plot(actual, predicted,
     xlim = c(-2.5, 1.5),
     ylim = c(-2.5, 1.5),
     xlab = "Reference SMB",
     ylab = "Modeled SMB",
     main = "Decision Tree Regression: Model Evaluation",
     pch = 19, col = adjustcolor("seagreen", alpha.f = 0.2),
     xaxt = "n", yaxt = "n")
abline(0, 1, col = "black", lwd = 2)  # Reference line predicted = actual
abline(h = 0, lty = 2, col = "gray37")
abline(v = 0, lty = 2, col = "gray37")
ticks <- seq(-2.5, 1.5, by = 0.5)
axis(1, at = ticks)
axis(2, at = ticks)

# R^2, RMSE, and MAE
# RMSE (Root Mean Squared Error)
rmse <- sqrt(mean((predicted - actual)^2))
print(rmse)

# R-squared (coefficient of determination)
sst <- sum((actual - mean(actual))^2)
ssr <- sum((actual - predicted)^2)
rsquared <- 1 - ssr / sst
print(rsquared)


# Gradient Boost using xgboost
# Drop the response variable from your dataset
treemodelgbX <- model.matrix(SMB ~ . - dmdtda - 1, data = myglacdata)  # -1 removes intercept
treemodelgbY <- myglacdata$SMB
xgb_model <- xgboost(
  data = treemodelgbX,
  label = treemodelgbY,
  objective = "reg:squarederror",
  nrounds = 300,
  max_depth = 4,
  min_child_weight = 5,
  eta = 0.1,
  subsample = 0.8,
  verbose = 0
)

predictedgb <- predict(xgb_model, newdata = treemodelgbX)

plot(treemodelgbY, predictedgb,
     xlab = "Reference SMB",
     ylab = "Modeled SMB",
     main = "XGBoost: Model Evaluation",
     xlim = c(-2.5, 1.5), ylim = c(-2.5, 1.5),
     col = adjustcolor("darkgreen", alpha.f = 0.3),
     pch = 19, xaxt = "n", yaxt = "n")

axis(1, at = seq(-2.5, 1.5, by = 0.5))
axis(2, at = seq(-2.5, 1.5, by = 0.5))
abline(0, 1, col = "black", lwd = 2)
abline(h = 0, lty = 2)
abline(v = 0, lty = 2)

# R^2 and RMSE for gradient boosting
# RMSE (Root Mean Squared Error)
rmsegb <- sqrt(mean((predictedgb - actual)^2))
print(rmsegb)

# R-squared (coefficient of determination)
sstgb <- sum((actual - mean(actual))^2)
ssrgb <- sum((actual - predictedgb)^2)
rsquaredgb <- 1 - ssrgb / sstgb
print(rsquaredgb)

# Importance ranks the effect of each variable in my gradient boosting model
importance <- xgb.importance(model = xgb_model)
print(importance)



## Could be over-fitted, so let's try on testing data set
# Loading in testing data
glactestx <- read.csv("glaciertestx.csv") # Data set with all predictors
glactesty <- read.csv("glaciertesty.csv") # Data set with SMB

# Merging data into one data frame
myglacdatatest <- cbind(glactesty, glactestx)
myglacdatatest <- myglacdatatest[, -c(1, 3, 4, 5)]

# Surface Mass Balance data
SMB <- myglacdatatest$dmdtda
myglacdatatest$SMB <- myglacdatatest$dmdtda

# Predictors 
pdd <- myglacdatatest$PDD
area <- myglacdatatest$area
lat <- myglacdatatest$lat
rain <- myglacdatatest$rain
slope <- myglacdatatest$slope
snow <- myglacdatatest$snow

zmax <- myglacdatatest$zmax
zmed <- myglacdatatest$zmed
zmin <- myglacdatatest$zmin

p1 <- myglacdatatest$prcp_01
p2 <- myglacdatatest$prcp_02
p3 <- myglacdatatest$prcp_03
p4 <- myglacdatatest$prcp_04
p5 <- myglacdatatest$prcp_05
p6 <- myglacdatatest$prcp_06
p7 <- myglacdatatest$prcp_07
p8 <- myglacdatatest$prcp_08
p9 <- myglacdatatest$prcp_09
p10 <- myglacdatatest$prcp_10
p11 <- myglacdatatest$prcp_11
p12 <- myglacdatatest$prcp_12

t1 <- myglacdatatest$temp_01
t2 <- myglacdatatest$temp_02
t3 <- myglacdatatest$temp_03
t4 <- myglacdatatest$temp_04
t5 <- myglacdatatest$temp_05
t6 <- myglacdatatest$temp_06
t7 <- myglacdatatest$temp_07
t8 <- myglacdatatest$temp_08
t9 <- myglacdatatest$temp_09
t10 <- myglacdatatest$temp_10
t11 <- myglacdatatest$temp_11
t12 <- myglacdatatest$temp_12

# Predicting using decision tree
predictedtest <- predict(treemodel, newdata = myglacdatatest)
actualtest <- SMB

# Plotting actual vs predicted of model on testing set, decision tree
plot(actualtest, predictedtest,
     xlim = c(-2.5, 1.5),
     ylim = c(-2.5, 1.5),
     xlab = "Reference SMB",
     ylab = "Modeled SMB",
     main = "Decision Tree Regression: Model used on Testing Data",
     pch = 19, col = adjustcolor("seagreen", alpha.f = 0.2),
     xaxt = "n", yaxt = "n")
abline(0, 1, col = "black", lwd = 2)  # Reference line predicted = actual
abline(h = 0, lty = 2, col = "gray37")
abline(v = 0, lty = 2, col = "gray37")
ticks <- seq(-2.5, 1.5, by = 0.5)
axis(1, at = ticks)
axis(2, at = ticks)

# R^2, RMSE, MAE, MSE for decision tree model used on testing data
# RMSE (Root Mean Squared Error)
rmsetest <- sqrt(mean((predictedtest - actualtest)^2))
print(rmsetest)

# R-squared (coefficient of determination)
ssttest <- sum((actualtest - mean(actualtest))^2)
ssrtest <- sum((actualtest - predictedtest)^2)
rsquaredtest <- 1 - ssrtest / ssttest
print(rsquaredtest)

# MAE and MSE for decision tree
maetree <- mean(abs(predictedtest - actualtest))
print(maetree)
msetree <- mean((predictedtest - actualtest)^2)
print(msetree)


# Predicting using gradient boosting
treemodelgbXtest <- model.matrix(SMB ~ . - dmdtda - 1, data = myglacdatatest)
predictedgbtest <- predict(xgb_model, newdata = treemodelgbXtest)

actualtest <- SMB


# Plotting actual vs predicted of model on testing data, gradient boosting
plot(actualtest, predictedgbtest,
     xlim = c(-2.5, 1.5),
     ylim = c(-2.5, 1.5),
     xlab = "Reference SMB",
     ylab = "Modeled SMB",
     main = "XGBoost Regression: Model used on Testing Data",
     pch = 19, col = adjustcolor("darkgreen", alpha.f = 0.3),
     xaxt = "n", yaxt = "n")
abline(0, 1, col = "black", lwd = 2)
abline(h = 0, lty = 2, col = "gray37")
abline(v = 0, lty = 2, col = "gray37")
ticks <- seq(-2.5, 1.5, by = 0.5)
axis(1, at = ticks)
axis(2, at = ticks)

# R^2, RMSE, MAE, MSE for gradient boosting model used on testing data
# RMSE on testing set
rmsegbtest <- sqrt(mean((predictedgbtest - actualtest)^2))
print(rmsegbtest)

# R^2 on testing set
sstgbtest <- sum((actualtest - mean(actualtest))^2)
ssrgbtest <- sum((actualtest - predictedgbtest)^2)
rsquaredgbtest <- 1 - ssrgbtest / sstgbtest
print(rsquaredgbtest)

# MAE and MSE for gradient boosting
maegb <- mean(abs(predictedgbtest - actualtest))
print(maegb)
msegb <- mean((predictedgbtest - actualtest)^2)
print(msegb)

