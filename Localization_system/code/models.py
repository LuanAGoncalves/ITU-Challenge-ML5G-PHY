import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, concatenate, Dropout


def build_default_mlp(n_inputs):
    inputs = tf.keras.Input(shape=(n_inputs,))

    model = tf.keras.Sequential(
        [
            Dense(n_inputs, activation="relu", name="layer1"),
            Dropout(0.3),
            Dense(n_inputs * 2, activation="relu", name="layer2"),
            Dropout(0.3),
            Dense(n_inputs * 4, activation="relu", name="layer3"),
            Dropout(0.3),
            Dense(n_inputs * 2, activation="relu", name="layer4"),
            Dropout(0.3),
            Dense(n_inputs, activation="relu", name="layer5"),
            Dropout(0.3),
            Dense(2, activation=None, name="layer6"),
        ]
    )

    x = model(inputs)
    outputs = tf.keras.activations.relu(x, threshold=-1)

    return Model(inputs=inputs, outputs=outputs)
