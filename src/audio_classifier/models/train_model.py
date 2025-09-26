import os
import yaml
import numpy as np
import tensorflow as tf
import librosa 
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from audio_classifier.logger import logger
from audio_classifier.models.architecture import create_model


def add_noise(signal, noise_factor=0.005):
    noise = np.random.randn(len(signal))
    augmented_signal = signal + noise_factor * noise
    return augmented_signal

def time_stretch(signal, stretch_rate=0.8):
    return librosa.effects.time_stretch(y=signal, rate=stretch_rate)



def train_model(config_path="params.yaml"):
    logger.info("Starting model training pipeline...")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    features_dir = config["data"]["features_path"]
    model_dir = config["data"]["model_dir"]
    random_state = config["base"]["random_state"]
    
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "model.keras") 
    
    logger.info("Loading features and labels...")
    features = np.load(os.path.join(features_dir, "features.npy"))
    labels = np.load(os.path.join(features_dir, "labels.npy"))
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=random_state
    )


    logger.info("Applying data augmentation to the training set...")
    X_train_augmented = []
    y_train_augmented = []
    
    for i in range(len(X_train)):
        signal = X_train[i]
        label = y_train[i]
        

        X_train_augmented.append(signal)
        y_train_augmented.append(label)
        
        if np.random.rand() < 0.5: 
            augmented_signal = add_noise(signal.flatten()).reshape(signal.shape)
        else:

            augmented_signal = add_noise(signal.flatten(), noise_factor=0.007).reshape(signal.shape)

        X_train_augmented.append(augmented_signal)
        y_train_augmented.append(label)

    X_train = np.array(X_train_augmented)
    y_train = np.array(y_train_augmented)
    logger.info(f"Augmentation complete. New training set shape: {X_train.shape}")


    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(features, axis=-1)[len(X_train):] 
    X_test = np.expand_dims(X_test, axis=-1) 
    y_test = labels[len(X_train):] 

    
    if os.path.exists(model_path):
        logger.info("Existing model found. Loading it...")
        model = tf.keras.models.load_model(model_path)
    else:
        logger.info("No existing model found. Creating a new one...")
        input_shape = X_train.shape[1:]
        model = create_model(input_shape=input_shape)
    
    model.summary(print_fn=logger.info)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), 
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    logger.info("Model compiled successfully.")

    logger.info("Starting model training...")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=20,
        batch_size=32,
        callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)] 
    )

    model.save(model_path)
    logger.info(f"Model training complete. Model saved to: {model_path}")

if __name__ == '__main__':
    train_model()