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


#Build the counting corpus
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(min_df=30)
X_train_counts = count_vect.fit_transform(X_train)

## Get the TF-IDF vector representation of the data
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print ('Dimension of TF-IDF vector :' , X_train_tfidf.shape)

#Creating the classifier
#MultinomialNB accepts weights instead of Boolean
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(alpha=1)
# the fit() function of any classifier takes the features from the
# training set X_train_tfidf and the labels from the training set
# y_train
clf.fit(X_train_tfidf, y_train)

#Performing the prediction

#indexing the test set
X_new_counts = count_vect.transform(X_test)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

#performing the actual prediction
predicted = clf.predict(X_new_tfidf)

print(predicted)
print(np.mean(predicted==y_test))


#Getting feature importance
neg_class_prob_sorted = clf.feature_log_prob_[0,: ].argsort()[::-1]
pos_class_prob_sorted = clf.feature_log_prob_[1,: ].argsort()[::-1]
print("Ham top 20 features:",np.take(count_vect.get_feature_names_out(), neg_class_prob_sorted[: 20]))
print("Spam top 20 features:",np.take(count_vect.get_feature_names_out(), pos_class_prob_sorted[: 20]))


