import pandas as pd
from bs4 import BeautifulSoup
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import callbacks






NB_WORDS = 30000  # Parameter indicating the number of words we'll put in the dictionary
NB_EPOCHS = 20  # Number of epochs we usually start to train with
BATCH_SIZE = 50  # Size of the batches used in the mini-batch gradient descent
MAX_LEN = 50  # Maximum number of words in a sequence
FILTER_STRING='!"#$%&()*+,-./:;<=>?@[\]^_`{"}~\t\n'
EMBEDDING_SIZE=100 # Size of the word embedding
PATIENCE=10 # Patience level
DROP_RATE=0.4 # Dropout rate


dataset=pd.read_csv("a.csv")
print(dataset.head())


X_trainAll, X_test, y_trainAll, y_test = train_test_split(dataset['clean_comment'], dataset['category'],
                                                    test_size=0.10, random_state=10)

X_train, X_valid, y_train, y_valid = train_test_split(X_trainAll, y_trainAll,
                                                          test_size=0.20, random_state=10)


tokenizer = Tokenizer(split=" ",oov_token="<OOV>")

tokenizer.fit_on_texts(X_train) #fits the sentences, creating the dictionary
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_valid_seq = tokenizer.texts_to_sequences(X_valid)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_seq_trunc = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post')
X_valid_seq_trunc = pad_sequences(X_valid_seq, maxlen=MAX_LEN, padding='post')
X_test_seq_trunc = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding='post')

from sklearn.preprocessing import OneHotEncoder

oh=OneHotEncoder(sparse=False)
y_train_oh=oh.fit_transform([[x] for x in y_train])
y_valid_oh=oh.transform([[x] for x in y_valid])
y_test_oh=oh.transform([[x] for x in y_test])



voc_len=len(tokenizer.word_index)

model = models.Sequential()
model.add(layers.Embedding(voc_len+1,EMBEDDING_SIZE,input_length=MAX_LEN))
model.add(layers.Dropout(DROP_RATE))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(DROP_RATE))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(DROP_RATE))
#model.add(layers.GRU(128,return_sequences=True))
#model.add(layers.GRU(128))
model.add(layers.Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

checkpoint_cb = callbacks.ModelCheckpoint("my_keras_model.h5", save_best_only=True)
# history = model.fit(X_train, y_train, epochs=10,
#                     validation_data=(X_valid, y_valid),
#                     callbacks=[checkpoint_cb])
# model = keras.models.load_model("my_keras_model.h5") # rollback to best model


early_stopping_cb = callbacks.EarlyStopping(patience=PATIENCE,
                                                  restore_best_weights=True)
history = model.fit(X_train_seq_trunc, y_train_oh, epochs=NB_EPOCHS,
                    validation_data=(X_valid_seq_trunc, y_valid_oh),
                    callbacks=[checkpoint_cb, early_stopping_cb],batch_size=BATCH_SIZE)
#here you can apply on the test set

model = models.load_model("my_keras_model.h5") # rollback to best model

loss, accuracy = model.evaluate(X_train_seq_trunc, y_train_oh, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))

loss, accuracy = model.evaluate(X_test_seq_trunc, y_test_oh, verbose=True)
print("Testing Accuracy:  {:.4f}".format(accuracy))

import matplotlib.pyplot as plt
pd.DataFrame(history.history).plot()
plt.grid(True)
plt.show()


#Prediction metrics
from sklearn.metrics import classification_report
import numpy as np

y_pred = model.predict(X_test_seq_trunc, verbose=1)
print(y_pred)
y_pred_cat=np.argmax(y_pred,axis=1)-1
y_test_cat=np.argmax(y_test_oh,axis=1)-1
print("Confusion matrix: ",pd.crosstab(y_test_cat,y_pred_cat))

print(classification_report(y_test_cat,y_pred_cat))
