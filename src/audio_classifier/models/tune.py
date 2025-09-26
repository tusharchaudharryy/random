import os
import yaml
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from audio_classifier.logger import logger

def build_model(hp):
    """
    Builds a model with tunable hyperparameters.
    """
    model = tf.keras.models.Sequential()
    
    # Input Layer
    model.add(tf.keras.layers.Input(shape=(128, 157, 1))) # Use fixed input shape for tuning
    
    # Tune the number of filters and kernel size for Conv layers
    hp_filters_1 = hp.Int('filters_1', min_value=8, max_value=32, step=8)
    model.add(tf.keras.layers.Conv2D(filters=hp_filters_1, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    
    hp_filters_2 = hp.Int('filters_2', min_value=16, max_value=64, step=16)
    model.add(tf.keras.layers.SeparableConv2D(filters=hp_filters_2, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(tf.keras.layers.Flatten())
    
    # Tune the number of units in the Dense layer
    hp_units = hp.Int('units', min_value=32, max_value=128, step=32)
    model.add(tf.keras.layers.Dense(units=hp_units, activation='relu'))
    
    # Tune the dropout rate
    hp_dropout = hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)
    model.add(tf.keras.layers.Dropout(rate=hp_dropout))
    
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    # Tune the learning rate
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

# --- Main Tuning Logic ---
logger.info("Starting hyperparameter tuning...")
with open("params.yaml") as f:
    config = yaml.safe_load(f)

# Load data
features_dir = config["data"]["features_path"]
features = np.load(os.path.join(features_dir, "features.npy"))
labels = np.load(os.path.join(features_dir, "labels.npy"))
features = np.expand_dims(features, axis=-1)

# Split data
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

# Instantiate the tuner
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10, # Number of different model configurations to test
    executions_per_trial=1, # Number of times to train each model
    directory='tuner_results',
    project_name='audio_classification'
)

# Start the search
tuner.search(X_train, y_train, epochs=20, validation_data=(X_val, y_val))

# Get and print the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
logger.info(f"""
The hyperparameter search is complete. The optimal number of filters in the first conv layer is {best_hps.get('filters_1')} 
and the second is {best_hps.get('filters_2')}. The optimal number of units in the dense layer is {best_hps.get('units')}, 
the optimal dropout rate is {best_hps.get('dropout')}, and the optimal learning rate is {best_hps.get('learning_rate')}.
""")