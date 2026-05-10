import tensorflow_hub as hub
import tensorflow as tf
import tensorflow.keras as keras

embed = hub.load("https://tfhub.dev/google/Wiki-words-250/2")
embeddings = embed(["cat is on the mat", "dog is in the fog"])


hub_layer = hub.KerasLayer("https://tfhub.dev/google/Wiki-words-250/2",
                           input_shape=[], dtype=tf.string)

model = keras.Sequential()
model.add(hub_layer)
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()