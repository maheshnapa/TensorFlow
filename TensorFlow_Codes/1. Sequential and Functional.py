import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

#physical_devices = tf.config.list_physical_devices("GPU")
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)

# import sys
# sys.exit()

x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0

print(x_train.shape)


              ##################################################
              ##########    Sequential Layer ###################
              ##################################################

'''The sequential API allows you to create models layer-by-layer for most problems. 
It is limited in that it does not allow you to create models that share layers or have multiple inputs or outputs.'''

# Sequential API (Very convenient, not very flexible)
model = keras.Sequential(
    [
        keras.Input(shape=(28 * 28)),
        layers.Dense(512, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(10),
    ]
)

model = keras.Sequential()
model.add(keras.Input(shape=(784)))
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dense(256, activation="relu", name="my_layer"))
model.add(layers.Dense(10))

                ##################################################
                ##########    Functional layer ###################
                ##################################################

'''the functional API allows you to create models that have a lot more flexibility
as you can easily define models where layers connect to more than just the previous and next layers.
In fact, you can connect layers to (literally) any other layer.
As a result, creating complex networks such as siamese networks and residual networks become possible.'''

# Functional API (A bit more flexible)

# method-1

inputs = keras.Input(shape=(784))
x = layers.Dense(512, activation="relu", name="first_layer")(inputs)
x = layers.Dense(256, activation="relu", name="second_layer")(x)
outputs = layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)


# Method 2
def create_model():
    inputs = keras.Input(shape=(784))
    x = layers.Dense(512, activation="relu", name="first_layer")(inputs)
    x = layers.Dense(256, activation="relu", name="second_layer")(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

model = create_model()

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"],
)

model.fit(x_train, y_train, batch_size=32, epochs=1, verbose=2)
model.evaluate(x_test, y_test, batch_size=32, verbose=2)
