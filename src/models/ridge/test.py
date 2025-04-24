import numpy as np
import matplotlib.pyplot as plt

from ridge import RidgeRegression
import utils

simulated = utils.simulate_data()
sim_X, sim_y = simulated[0], simulated[1]

ridge_test = RidgeRegression.new_model(sim_X, sim_y)
ridge_test.fit()

lambda_vals = [0.0, 0.001, 0.1, 1.0, 10.0, 100.0, 1000.0]
  
# Now fit lambda regularization term
ridge_test.fit(cross_val=True, nfolds=10, param_vals=lambda_vals, iterations=100)

# Predict
simulated_ = utils.simulate_data()
newX, newy = simulated_[0], simulated_[1]

yhat = ridge_test.predict(newX)

# Test whether predictions == true values
fig, ax = plt.subplots(figsize=(10,10))
ax.scatter(newy, yhat, c='crimson')
ax.axline((1, 1), slope=1)
#plt.xlabel('True Values', fontsize=15)
#plt.ylabel('Predictions', fontsize=15)
plt.show()
quit()