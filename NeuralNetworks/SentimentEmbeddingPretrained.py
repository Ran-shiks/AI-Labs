import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from keras import models
from keras import layers
from keras import callbacks
from sklearn.model_selection import train_test_split


def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

dataset=pd.read_csv("IMDB Dataset.csv")

NB_WORDS = 40000  # Parameter indicating the number of words we'll put in the dictionary
NB_EPOCHS = 50  # Number of epochs we usually start to train with
BATCH_SIZE = 25  # Size of the batches used in the mini-batch gradient descent
MAX_LEN = 200  # Maximum number of words in a sequence
FILTER_STRING='!"#$%&()*+,-./:;<=>?@[\]^_`{"}~\t\n'
PATIENCE=20
DROP_RATE=0.5



dataset['review']=dataset['review'].map(strip_html)

X_trainAll, X_test, y_trainAll, y_test = train_test_split(dataset['review'], dataset['sentiment'],
                                                    test_size=0.10, random_state=10)

X_train, X_valid, y_train, y_valid = train_test_split(X_trainAll, y_trainAll,
                                                          test_size=0.20, random_state=10)

tokenizer = Tokenizer(num_words=NB_WORDS,filters=FILTER_STRING,lower=True, split=" ",oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
word_index=tokenizer.word_index
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_valid_seq = tokenizer.texts_to_sequences(X_valid)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_seq_trunc = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post')
X_valid_seq_trunc = pad_sequences(X_valid_seq, maxlen=MAX_LEN, padding='post')
X_test_seq_trunc = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding='post')


le = LabelEncoder()
y_train_le=le.fit_transform(y_train)
y_valid_le=le.transform(y_valid)
y_test_le=le.transform(y_test)



import gensim.downloader
glove_vectors = gensim.downloader.load('glove-wiki-gigaword-100')
EMBEDDING_SIZE=len(glove_vectors[0])

voc_len=len(word_index)+1

embedding_matrix = np.zeros((voc_len, EMBEDDING_SIZE))
for word, i in word_index.items():
    if word in glove_vectors:
        embedding_vector = glove_vectors[word]
        embedding_matrix[i] = embedding_vector



model = models.Sequential()
model.add(layers.Embedding(
    voc_len,
    EMBEDDING_SIZE,
    embeddings_initializer=keras.initializers.Constant(embedding_matrix),
    trainable=False,input_length=MAX_LEN))
model.add(layers.Dropout(DROP_RATE))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(DROP_RATE))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(DROP_RATE))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


model.summary()

checkpoint_cb = callbacks.ModelCheckpoint("my_keras_model.h5", save_best_only=True)
# history = model.fit(X_train, y_train, epochs=10,
#                     validation_data=(X_valid, y_valid),
#                     callbacks=[checkpoint_cb])
# model = keras.models.load_model("my_keras_model.h5") # rollback to best model


early_stopping_cb = callbacks.EarlyStopping(patience=PATIENCE,
                                                  restore_best_weights=True)
history = model.fit(X_train_seq_trunc, y_train_le, epochs=NB_EPOCHS,
                    validation_data=(X_valid_seq_trunc, y_valid_le),
                    callbacks=[checkpoint_cb, early_stopping_cb],batch_size=BATCH_SIZE)
#here you can apply on the test set


loss, accuracy = model.evaluate(X_train_seq_trunc, y_train_le, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))

loss, accuracy = model.evaluate(X_test_seq_trunc, y_test_le, verbose=True)
print("Testing Accuracy:  {:.4f}".format(accuracy))

import matplotlib.pyplot as plt
pd.DataFrame(history.history).plot()
plt.grid(True)
plt.show()