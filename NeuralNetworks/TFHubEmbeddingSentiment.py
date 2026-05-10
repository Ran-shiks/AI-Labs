import tensorflow_hub as hub
import tensorflow as tf
import tensorflow.keras as keras

#embed = hub.load("https://tfhub.dev/google/Wiki-words-250/2")
#embeddings = embed(["cat is on the mat", "dog is in the fog"])

#print(embeddings)

#exit(0)
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import callbacks


def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

dataset=pd.read_csv("IMDB Dataset.csv")

#NB_WORDS = 30000  # Parameter indicating the number of words we'll put in the dictionary
NB_EPOCHS = 20  # Number of epochs we usually start to train with
BATCH_SIZE = 50  # Size of the batches used in the mini-batch gradient descent
MAX_LEN = 100  # Maximum number of words in a sequence
#FILTER_STRING='!"#$%&()*+,-./:;<=>?@[\]^_`{"}~\t\n'
#EMBEDDING_SIZE=100 # Size of the word embedding
PATIENCE=10 # Patience level
DROP_RATE=0.4 # Dropout rate


dataset['review']=dataset['review'].map(strip_html)

X_trainAll, X_test, y_trainAll, y_test = train_test_split(dataset['review'], dataset['sentiment'],
                                                          test_size=0.10, random_state=10)

X_train, X_valid, y_train, y_valid = train_test_split(X_trainAll, y_trainAll,
                                                      test_size=0.20, random_state=10)

le = LabelEncoder()
y_train_le=le.fit_transform(y_train)
y_valid_le=le.transform(y_valid)
y_test_le=le.transform(y_test)


hub_layer = hub.KerasLayer("https://tfhub.dev/google/Wiki-words-250/2",
                           input_shape=[], dtype=tf.string,trainable=False)

model = keras.Sequential()
model.add(hub_layer)
model.add(layers.Dropout(DROP_RATE))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(layers.Dropout(DROP_RATE))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(layers.Dropout(DROP_RATE))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="tb_dir", histogram_freq=1)

checkpoint_cb = callbacks.ModelCheckpoint("my_keras_model.h5", save_best_only=True)

early_stopping_cb = callbacks.EarlyStopping(patience=PATIENCE,
                                            restore_best_weights=True)
history = model.fit(X_train, y_train_le, epochs=NB_EPOCHS,
                    validation_data=(X_valid, y_valid_le),
                    callbacks=[tensorboard_callback,early_stopping_cb, checkpoint_cb],batch_size=BATCH_SIZE)
#here you can apply on the test set

#model = models.load_model("my_keras_model.h5") # rollback to best model


loss, accuracy = model.evaluate(X_train, y_train_le, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))

loss, accuracy = model.evaluate(X_test, y_test_le, verbose=True)
print("Testing Accuracy:  {:.4f}".format(accuracy))

import matplotlib.pyplot as plt
pd.DataFrame(history.history).plot()
plt.grid(True)
plt.show()