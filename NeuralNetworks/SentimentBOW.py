import pandas as pd
from nltk.corpus import stopwords
import gensim
import numpy as np
from bs4 import BeautifulSoup



def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def transformText(text):
    stops = set(stopwords.words("english"))
    # Remove HTML
    text = strip_html(text)
    # Convert text to lowercase
    text = text.lower()
    # Strip multiple whitespaces
    text = gensim.corpora.textcorpus.strip_multiple_whitespaces(text)
    # Removing all the stopwords
    filtered_words = [word for word in text.split() if word not in stops]
    # Preprocessed text after stop words removal
    text = " ".join(filtered_words)
    # Remove the punctuation
    text = gensim.parsing.preprocessing.strip_punctuation(text)
    # Strip all the numerics
    text = gensim.parsing.preprocessing.strip_numeric(text)
    # Removing all the words with less than 3 characters
    text = gensim.parsing.preprocessing.strip_short(text, minsize=3)
    # Strip multiple whitespaces
    text = gensim.corpora.textcorpus.strip_multiple_whitespaces(text)
    # Stemming
    return gensim.parsing.preprocessing.stem_text(text)


def transformLabel(text):
    return 1 if text=="positive" else 0

dataset=pd.read_csv("IMDB Dataset.csv")
print(dataset.describe())

print(dataset['sentiment'].value_counts())

dataset['review'] = dataset['review'].map(transformText)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
dataset['sentiment'] =le.fit_transform(dataset['sentiment'])

print(dataset['review'])
from sklearn.model_selection import train_test_split
X_trainAll, X_test, y_trainAll, y_test = train_test_split(dataset['review'], dataset['sentiment'],
                                                    test_size=0.10, random_state=10)

X_train, X_valid, y_train, y_valid = train_test_split(X_trainAll, y_trainAll,
                                                          test_size=0.20, random_state=10)

#Build the counting corpus
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
count_vect = CountVectorizer(min_df=30)
X_train = count_vect.fit_transform(X_train)
## Get the TF-IDF vector representation of the data
tfidf_transformer = TfidfTransformer()
X_train = tfidf_transformer.fit_transform(X_train).toarray()

X_valid=count_vect.transform(X_valid)
X_valid=tfidf_transformer.transform(X_valid).toarray()

X_test=count_vect.transform(X_test)
X_test=tfidf_transformer.transform(X_test).toarray()

from keras.models import Sequential
from keras import layers
input_dim = X_train.shape[1]  # Number of features
print("Input:",input_dim)
model = Sequential()
#model.add(layers.Dropout(0.5, input_shape=(input_dim,)))
#model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
#model.add(layers.Dropout(0.3))
model.add(layers.Dense(100, activation='relu',input_shape=(input_dim,)))
#model.add(layers.Dropout(0.5))
model.add(layers.Dense(100, activation='relu'))
#model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train, epochs=20, verbose=True,
                    validation_data=(X_valid, y_valid), batch_size=10)

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))

loss, accuracy = model.evaluate(X_test, y_test, verbose=True)
print("Testing Accuracy:  {:.4f}".format(accuracy))


#Prediction metrics
from sklearn.metrics import classification_report

y_pred = model.predict(X_test, verbose=1)
pred_threshold=0.5
print("Y pred",y_pred)
print("Y test",y_test)
y_pred_bool = [int(x+0.5) for [x] in y_pred]


print(y_pred_bool)
print(classification_report(y_test, y_pred_bool))