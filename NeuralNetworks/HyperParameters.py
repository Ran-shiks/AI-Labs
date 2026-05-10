import pandas as pd
from bs4 import BeautifulSoup
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
from keras_tuner import HyperModel

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

class EmbeddingHyperModel(HyperModel):
    def __init__(self,voc_size,emb_size,max_len,numNodes,
                 minLayers,maxLayers,minDrop,maxDrop):
        self.voc_size=voc_size
        self.emb_size=emb_size
        self.max_len=max_len
        self.numNodes=numNodes
        self.minLayers=minLayers
        self.maxLayers=maxLayers
        self.minDrop=minDrop
        self.maxDrop=maxDrop

    def build(self,hp):
        drop_rate=hp.Float(name="dropout",min_value=self.minDrop,
                           max_value=self.maxDrop,sampling='linear')
        nodes_hidden=hp.Choice("units",self.numNodes)
        model = keras.Sequential()
        model.add(layers.Embedding(voc_len+1,100,input_length=100))
        model.add(layers.Dropout(drop_rate))
        model.add(layers.Flatten())
        for i in range(hp.Int(name="layers",min_value=self.minLayers,
                              max_value=self.maxLayers,sampling='linear')):
            model.add(layers.Dense(nodes_hidden, activation='relu'))
            model.add(layers.Dropout(drop_rate))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

dataset=pd.read_csv("IMDB Dataset.csv")

NB_WORDS = 30000  # Parameter indicating the number of words we'll put in the dictionary
NB_EPOCHS = 20  # Number of epochs we usually start to train with
BATCH_SIZE = 50  # Size of the batches used in the mini-batch gradient descent
MAX_LEN = 100  # Maximum number of words in a sequence
FILTER_STRING='!"#$%&()*+,-./:;<=>?@[\]^_`{"}~\t\n'
EMBEDDING_SIZE=100 # Size of the word embedding
PATIENCE=10 # Patience level


dataset['review']=dataset['review'].map(strip_html)

X_trainAll, X_test, y_trainAll, y_test = train_test_split(dataset['review'], dataset['sentiment'],
                                                          test_size=0.10, random_state=10)

#X_train, X_valid, y_train, y_valid = train_test_split(X_trainAll, y_trainAll,
#                                                      test_size=0.20, random_state=10)


tokenizer = Tokenizer(num_words=NB_WORDS,filters=FILTER_STRING,
                      lower=True, split=" ",oov_token="<OOV>")

tokenizer.fit_on_texts(X_trainAll) #fits the sentences, creating the dictionary
X_train_seq = tokenizer.texts_to_sequences(X_trainAll)
#X_valid_seq = tokenizer.texts_to_sequences(X_valid)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_seq_trunc = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post')
#X_valid_seq_trunc = pad_sequences(X_valid_seq, maxlen=MAX_LEN, padding='post')
X_test_seq_trunc = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding='post')

le = LabelEncoder()
y_train_le=le.fit_transform(y_trainAll)
#y_valid_le=le.transform(y_valid)
y_test_le=le.transform(y_test)


voc_len=len(tokenizer.word_index)

hm=EmbeddingHyperModel(voc_len,EMBEDDING_SIZE,MAX_LEN,[128,256,512],1,5,0,0.4)

tuner = kt.Hyperband(hm,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='my_dir',
                     project_name='intro_to_kt')


stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner.search(X_train_seq_trunc, y_train_le, epochs=50, validation_split=0.2, callbacks=[stop_early])

best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. 
The optimal number of layers is {best_hps.get('layers')}
The units in the densely-connected layers are {best_hps.get('units')} 
The optimal dropout rate is {best_hps.get('dropout')} 
and the optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.
""")

model = tuner.hypermodel.build(best_hps)

# From here, you can just use the model to fit it and use it