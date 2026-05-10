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

from sklearn.preprocessing import LabelEncoder
dataset=pd.read_csv("IMDB Dataset.csv")
print(dataset.describe())
print(dataset['sentiment'].value_counts())

dataset['review'] = dataset['review'].map(transformText)

le = LabelEncoder()
le.fit(dataset['sentiment'])
dataset['sentiment']=le.transform(dataset['sentiment'])

dataset.to_csv("IMDB_Table.csv",index=False)

