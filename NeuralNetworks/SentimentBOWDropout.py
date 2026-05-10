import pandas as pd

dataset=pd.read_csv("IMDB_Table.csv")

from sklearn.model_selection import train_test_split
X_trainAll, X_test, y_trainAll, y_test = train_test_split(dataset['review'], dataset['sentiment'],
                                                    test_size=0.10, random_state=10)

X_train, X_valid, y_train, y_valid = train_test_split(X_trainAll, y_trainAll,
                                                          test_size=0.20, random_state=10)

#Build the counting corpus
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
count_vect = CountVectorizer(min_df=30)
tfidf_transformer = TfidfTransformer()

X_train = count_vect.fit_transform(X_train)
X_train = tfidf_transformer.fit_transform(X_train).toarray()

X_valid=count_vect.transform(X_valid)
X_valid=tfidf_transformer.transform(X_valid).toarray()

X_test=count_vect.transform(X_test)
X_test=tfidf_transformer.transform(X_test).toarray()


from keras.models import Sequential
from keras import layers
import keras.utils
input_dim = X_train.shape[1]  # Number of features

model = Sequential()
model.add(layers.Dropout(0.3, input_shape=(input_dim,)))
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


model.summary()

checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5", save_best_only=True)
# history = model.fit(X_train, y_train, epochs=10,
#                     validation_data=(X_valid, y_valid),
#                     callbacks=[checkpoint_cb])
# model = keras.models.load_model("my_keras_model.h5") # rollback to best model


early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,
                                                  restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100,
                    validation_data=(X_valid, y_valid),
                    callbacks=[checkpoint_cb, early_stopping_cb])
#here you can apply on the test set

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))

loss, accuracy = model.evaluate(X_test, y_test, verbose=True)
print("Testing Accuracy:  {:.4f}".format(accuracy))

import matplotlib.pyplot as plt
pd.DataFrame(history.history).plot()
plt.grid(True)
plt.show()