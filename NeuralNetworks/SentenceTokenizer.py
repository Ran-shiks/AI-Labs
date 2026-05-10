from tensorflow.keras.preprocessing.text import Tokenizer

NB_WORDS = 40000  # Parameter indicating the number of words we'll put in the dictionary
MAX_LEN = 20  # Maximum number of words in a sequence
FILTER_STRING='!"#$%&()*+,-./:;<=>?@[\]^_`{"}~\t\n'

sentences=["John is going to the bus stop",
           "The technician is repairing the F512"]


newSent=["Mary is going to the bus stop"]

tokenizer = Tokenizer(num_words=NB_WORDS,filters=FILTER_STRING,lower=True, split=" ",oov_token="<OOV>")
tokenizer.fit_on_texts(sentences) #fits the sentences, creating the dictionary
print("Word index:",tokenizer.word_index)

t=tokenizer.texts_to_sequences(sentences)
print("Sequences:",t)

t2=tokenizer.texts_to_sequences(newSent)
print("Test set",t2)

exit(0)
print("Reconstructed sentences",tokenizer.sequences_to_texts(t))

from tensorflow.keras.preprocessing.sequence import pad_sequences
padded_sentences = pad_sequences(t, maxlen=MAX_LEN,padding='post', truncating='post')
print(padded_sentences)


