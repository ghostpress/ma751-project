import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.initializers import HeUniform
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt

# 1. Load & preprocess
df_X = pd.read_csv("train_X.csv")
df_y = pd.read_csv("train_y.csv")
# quick peek
print(df_X.shape, df_y.shape)
print(df_X.head())
print(df_y.head())

# 1. Parse the “period” column into two new numeric features
df_X[['period_min', 'period_max']] = (
    df_X['period']
      .str.split('-', expand=True)    # split "2000-2010" → ["2000","2010"]
      .astype(int)                    # convert to integers
)

# 2. Drop only the truly non-predictor columns
X = df_X.drop(['Unnamed: 0', 'rgi_id', 'period'], axis=1).values
y = df_y['dmdtda'].values.reshape(-1,1) #.reshape(-1,1) makes it a column vector of shape so Keras sees it as a single‐output regression.


# 3. Build the 6-layer ANN, Dense: compute z=Wx+b. Order matters, they do Dense → BatchNorm → Activation → Dropout
model = Sequential([
    Dense(70, kernel_initializer=HeUniform(), input_shape=(X.shape[1],)), #He uniform initialization (He et al. 2015),
    BatchNormalization(), LeakyReLU(alpha=0.1), Dropout(0.05),
    #Each batch was normalized before applying the activation function in order to accelerate the training
    Dense(200, kernel_initializer=HeUniform()),
    BatchNormalization(), LeakyReLU(alpha=0.1), Dropout(0.3),

    Dense(150, kernel_initializer=HeUniform()),
    BatchNormalization(), LeakyReLU(alpha=0.1), Dropout(0.25),

    Dense(1, kernel_initializer=HeUniform(), activation='linear') #output layer
])

# 4. Compile the model with RMSprop optimizer
model.compile(
    optimizer=RMSprop(learning_rate=0.0008),
    loss='mse',
    metrics=['mae']
)

# 5. Train on the full set (with an internal validation split for early stopping)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True
)
# patience = 10: stop training if no improvement for 10 epochs
# restore_best_weights=True: roll back to the epoch with lowest validation loss

history = model.fit(
    X, y,
    epochs=100, #the paper trains the network after 100 epochs
    batch_size=32, #implement standard minibatch of 32 in gradient descent
    callbacks=[early_stop],
    validation_split=0.2,
    verbose=2 #report per-epoch training/validation curves to illustrate convergence and the effect of regularization
)

# Generate predictions on the “full test” data
y_pred = model.predict(X)

# (Optional) save them to CSV
pd.DataFrame(y_pred, columns=['dmdtda_pred']).to_csv('predictions.csv', index=False)

# Last‐epoch training error
last_train_mse = history.history['loss'][-1]
last_train_mae = history.history['mae'][-1]
print(f"Final training MSE: {last_train_mse:.4f}")
print(f"Final training MAE: {last_train_mae:.4f}")



# assuming y and y_pred are flat 1-D arrays of length n_samples
# Flatten into 1-D arrays
y_true = y.flatten()
y_hat  = y_pred.flatten()

# 1) Scatter of predicted vs. actual
plt.figure()
plt.scatter(y_true, y_hat, alpha=0.5)
plt.plot([y_true.min(), y_true.max()],
         [y_true.min(), y_true.max()],
         'r--', linewidth=1)  # 45° line
plt.xlabel('Actual SMB')
plt.ylabel('Predicted SMB')
plt.title('Predicted vs. Actual SMB (Training Data)')
plt.tight_layout()
plt.show()

# 2) Residuals vs. actual (optional)
residuals = abs(y_hat - y_true)
plt.figure()
plt.scatter(y_true, residuals, alpha=0.5)
plt.hlines(0, y_true.min(), y_true.max(), colors='red', linestyles='--')
plt.xlabel('Actual SMB')
plt.ylabel('Residual |Pred - True|')
plt.title('Residuals vs. Actual SMB')
plt.tight_layout()
plt.show()
