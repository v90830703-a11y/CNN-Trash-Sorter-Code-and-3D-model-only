from tensorflow import keras
from tensorflow.keras import layers

def build_model(input_shape=(32, 32, 1), num_classes=4):

    model = keras.Sequential([
        layers.Input(shape=input_shape),

        # FLATTEN EARLY (small input only!)
        layers.Flatten(),

        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),

        layers.Dense(num_classes, activation='softmax')
    ])

    return model