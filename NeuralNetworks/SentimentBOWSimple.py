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
model.add(layers.Dense(100, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()


history = model.fit(X_train, y_train, epochs=10, verbose=True,
                    validation_data=(X_valid, y_valid), batch_size=10)

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))

loss, accuracy = model.evaluate(X_test, y_test, verbose=True)
print("Testing Accuracy:  {:.4f}".format(accuracy))

import matplotlib.pyplot as plt
pd.DataFrame(history.history).plot()
plt.grid(True)
plt.show()