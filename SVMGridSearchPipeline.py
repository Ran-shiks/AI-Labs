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
#print(dataset['text'].head())


## Split the data
from sklearn.model_selection import train_test_split

#separate the test set
X_train, X_test, y_train, y_test = train_test_split(dataset['text'], dataset['type'],
                                                    test_size=0.33, random_state=10)
print ("Training Sample Size:", len(X_train), ' ', "Test Sample Size:" ,len(X_test))


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import svm
from time import time

#creates a model instance with no parameters
svc=svm.SVC()

# defines the steps of the pipeline, each with
# a name and the model object
pipeline = Pipeline(
    [
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("selector",SelectPercentile()),
        ("clf", svc),
    ]
)

# create a dictionary with possible values for some parameters
# each parameter name is composed as
# pipelineStepName__componentParameter
parameters = {
    "vect__ngram_range": ((1, 1), (1, 2)),
    "vect__min_df": (20,30,40),
    'tfidf__use_idf': (True, False),
    'selector__score_func': [chi2], #selector function needs a list
    'selector__percentile': (20,30,40),
    'clf__C': [1, 10, 100, 1000],
    'clf__gamma': [0.001, 0.0001],
    'clf__kernel': ['rbf','linear']
}

#instantiates the grid search
# using the svc model and the parameters above defined
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=10)

print("Performing grid search...")
print("parameters:")
print(parameters)
t0 = time()
# Starts the grid search
grid_search.fit(X_train, y_train)
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

#instantiating the model using the grid search best parameters
clf=best_pipe = grid_search.best_estimator_
clf.fit(X_train, y_train)

#performing the actual prediction
predicted = clf.predict(X_test)

from sklearn import metrics
print(pd.crosstab(y_test,predicted))
print(metrics.classification_report(y_test, predicted))