import tensorflow as tf
from tensorflow.keras import models, layers

def create_model(input_shape):
    """
    Builds and returns a lightweight CNN model.

    Args:
        input_shape (tuple): The shape of the input data (height, width, channels).

    Returns:
        A Keras Sequential model.
    """
    model = models.Sequential([

        layers.Input(shape=input_shape),


        layers.Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.SeparableConv2D(16, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        

        layers.SeparableConv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        

        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.5), 
        

        layers.Dense(1, activation='sigmoid') 
    ])

    return model