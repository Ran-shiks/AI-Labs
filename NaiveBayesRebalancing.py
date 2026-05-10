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
#import time
X_train, X_test, y_train, y_test = train_test_split(dataset['text'], dataset['type'],
                                                    test_size=0.33, random_state=10)

print ("Training Sample Size:", len(X_train), ' ', "Test Sample Size:" ,len(X_test))

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectPercentile
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#Build the counting corpus
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)

## Get the TF-IDF vector representation of the data
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print ('Dimension of TF-IDF vector :' , X_train_tfidf.shape)


#selector = SelectKBest(chi2, k=2000)
#X_new=selector.fit_transform(X_train_tfidf, y_train)

#from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
# instantiates the undersampler
#undersample = RandomUnderSampler(sampling_strategy='majority')
# undersamples the training set
#X_new, y_train = undersample.fit_resample(X_train_tfidf, y_train)
# prints the dataset composition
#counter=Counter(y_train)
#print(counter)

from imblearn.over_sampling import SMOTE, RandomOverSampler

#instantiate the SMOTE oversampler
#oversample = SMOTE()

#instantiate the random oversampler class
oversample = RandomOverSampler(sampling_strategy='minority')

# Rebalances the training set by creating artificial instances
# of the minority class. "minority" means that the minority class
# will be oversampled to match the majority class
X_new, y_train = oversample.fit_resample(X_train_tfidf, y_train)
# prints the dataset composition
counter=Counter(y_train)
print(counter)




clf = MultinomialNB()
clf.fit(X_new, y_train)
#Performing the prediction

#indexing the test set
X_new_counts = count_vect.transform(X_test)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
#xsel=selector.transform(X_new_tfidf)
#performing the actual prediction
predicted = clf.predict(X_new_tfidf)

print(predicted)
print(np.mean(predicted==y_test))

