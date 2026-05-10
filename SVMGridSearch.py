import pandas as pd
from nltk.corpus import stopwords
import gensim
import numpy as np


dataset=pd.read_csv("sms_spam.csv")
#print(dataset.head())
#print ("Shape:", dataset.shape, '\n')



def transformText(text):
    stops = set(stopwords.words("english"))
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

#applies transformText to all rows of text
dataset['text'] = dataset['text'].map(transformText)
print(dataset['text'].head())


## Split the data
from sklearn.model_selection import train_test_split

#separate the test set
X_train, X_test, y_train, y_test = train_test_split(dataset['text'], dataset['type'],
                                                    test_size=0.33, random_state=10)

print ("Training Sample Size:", len(X_train), ' ', "Test Sample Size:" ,len(X_test))


#Build the counting corpus
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(min_df=30)
X_train_counts = count_vect.fit_transform(X_train)

## Get the TF-IDF vector representation of the data
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print ('Dimension of TF-IDF vector :' , X_train_tfidf.shape)

from sklearn.model_selection import GridSearchCV
from sklearn import svm
from time import time

#creates a model instance with no parameters
svc=svm.SVC()

# prints the list of parameters for the model
print(svc.get_params().keys())

# create a dictionary with possible values for some parameters
parameters = {
    'C': [1, 10, 100, 1000],
    'gamma': [0.001, 0.0001],
    'kernel': ['rbf','linear']
}

#instantiates the grid search
# using the svc model and the parameters above defined
grid_search = GridSearchCV(svc, parameters, n_jobs=-1, verbose=10)

print("Performing grid search...")
print("parameters:")
print(parameters)
t0 = time()
# Starts the grid search
grid_search.fit(X_train_tfidf, y_train)
# Prints the required time
print("done in %0.3fs" % (time() - t0))
print()

# Prints the best score
print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

#Creating the model:

#instantiating the model using the grid search best estimator
clf= grid_search.best_estimator_
clf.fit(X_train_tfidf, y_train)

#indexing the test set
X_new_counts = count_vect.transform(X_test)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

#performing the actual prediction
predicted = clf.predict(X_new_tfidf)

from sklearn import metrics
print(pd.crosstab(y_test,predicted))
print(metrics.classification_report(y_test, predicted))