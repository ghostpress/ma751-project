import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras.initializers import HeUniform
from tensorflow.keras.optimizers import RMSprop
import kerastuner as kt

# 1. Load & preprocess
df_X = pd.read_csv("train_X.csv")
df_y = pd.read_csv("train_y.csv")

# 1. Parse the “period” column into two new numeric features
df_X[['period_min', 'period_max']] = (
    df_X['period']
      .str.split('-', expand=True)    # split "2000-2010" → ["2000","2010"]
      .astype(int)                    # convert to integers
)

# 2. Drop only the truly non-predictor columns
X = df_X.drop(['Unnamed: 0', 'rgi_id', 'period'], axis=1).values
y = df_y['dmdtda'].values.reshape(-1,1) #.reshape(-1,1) makes it a column vector of shape so Keras sees it as a single‐output regression.

def build_model(hp):
    model = Sequential()
    # Tune alpha
    alpha = hp.Float("alpha", min_value=0.01, max_value=0.3, step=0.01)

    # Tune how many layers: from 1 to 6
    for i in range(hp.Int("num_layers", 1, 10)):
        # Tune number of units in this layer
        units = hp.Int(f"units_{i}", min_value=10, max_value=200, step=10)
        
        # First layer needs input_shape
        if i == 0:
            model.add(Dense(units,
                            kernel_initializer=HeUniform(),
                            input_shape=(X.shape[1],)))
        else:
            model.add(Dense(units, kernel_initializer=HeUniform()))
        
        # Always use BatchNorm → LeakyReLU → Dropout
        model.add(BatchNormalization())
        model.add(LeakyReLU(negative_slope=alpha))
        
        # Tune dropout rate per layer
        dropout_rate = hp.Float(f"dropout_{i}", 0.0, 0.5, step=0.05)
        model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(1, activation="linear", kernel_initializer=HeUniform()))
    
    # Compile: tune learning rate if you like
    lr = hp.Float("lr", 1e-4, 1e-2, sampling="log")
    model.compile(
        optimizer=RMSprop(learning_rate=lr),
        loss="mse",
        metrics=["mae"]
    )
    return model

# --- 3. Set up the tuner ---
tuner = kt.RandomSearch(
    build_model,
    objective="val_loss",
    max_trials=20,             # try 20 different architectures
    executions_per_trial=1,    # train each once
    directory="tuner_dir",
    project_name="glacier_ann_arch", #It will save the old result if dont have overwrite = true
    overwrite=True 
)

# --- 4. Run the search ---
tuner.search(
    X, y,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[tf.keras.callbacks.EarlyStopping("val_loss", patience=5)],
    verbose=2
)

# --- 5. Inspect the best model & hyperparameters ---
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

print("Best number of layers:", best_hp.get("num_layers"))
for i in range(best_hp.get("num_layers")):
    print(
        f" Layer {i} units:", best_hp.get(f"units_{i}"),
        " dropout:", best_hp.get(f"dropout_{i}")
    )

# This gets your single global LeakyReLU slope:
print("Best LeakyReLU alpha:", best_hp.get("alpha"))

print("Best learning rate:", best_hp.get("lr"))

# Retrieve the best model
best_model = tuner.get_best_models(num_models=1)[0]
