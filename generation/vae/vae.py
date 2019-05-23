from keras.layers import Bidirectional, Dense, Embedding, Input, Lambda, LSTM, RepeatVector, TimeDistributed, Layer, Activation, Dropout
from keras.preprocessing.sequence import pad_sequences
from keras.layers.advanced_activations import ELU
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K
from keras.models import Model
from scipy import spatial
import tensorflow as tf
import pandas as pd
import numpy as np
import codecs
import csv
import os


BASE_DIR = 'C:/Users/gianc/Desktop/PhD/Progetti/vae/'
TRAIN_DATA_FILE = BASE_DIR + 'train.csv'#'train_micro.csv'
GLOVE_EMBEDDING = BASE_DIR + 'glove.6B.50d.txt'
VALIDATION_SPLIT = 0.2
MAX_SEQUENCE_LENGTH = 15
MAX_NB_WORDS = 12000
EMBEDDING_DIM = 50

texts = []
with codecs.open(TRAIN_DATA_FILE, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        texts.append(values[3])
        texts.append(values[4])
print('Found %s texts in train.csv' % len(texts))

tokenizer = Tokenizer(MAX_NB_WORDS+1, oov_token='unk') #+1 for 'unk' token
tokenizer.fit_on_texts(texts)
print('Found %s unique tokens' % len(tokenizer.word_index))
## **Key Step** to make it work correctly otherwise drops OOV tokens anyway!
tokenizer.word_index = {e:i for e,i in tokenizer.word_index.items() if i <= MAX_NB_WORDS} # <= because tokenizer is 1 indexed
tokenizer.word_index[tokenizer.oov_token] = MAX_NB_WORDS + 1
word_index = tokenizer.word_index #the dict values start from 1 so this is fine with zeropadding
index2word = {v: k for k, v in word_index.items()}
sequences = tokenizer.texts_to_sequences(texts)
data_1 = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', data_1.shape)
NB_WORDS = (min(tokenizer.num_words, len(word_index))+1) #+1 for zero padding

data_val = data_1[775000:783000]
data_train = data_1[:775000]

def sent_generator(TRAIN_DATA_FILE, chunksize):
    reader = pd.read_csv(TRAIN_DATA_FILE, chunksize=chunksize, iterator=True)
    for df in reader:
        # print(df.shape)
        # df=pd.read_csv(TRAIN_DATA_FILE, iterator=False)
        val3 = df.iloc[:, 3:4].values.tolist()
        val4 = df.iloc[:, 4:5].values.tolist()
        flat3 = [item for sublist in val3 for item in sublist]
        flat4 = [str(item) for sublist in val4 for item in sublist]
        texts = []
        texts.extend(flat3[:])
        texts.extend(flat4[:])

        sequences = tokenizer.texts_to_sequences(texts)
        data_train = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        yield [data_train, data_train]

