# -*- coding: utf-8 -*-
import sys
import numpy as np
from collections import Counter
import codecs
import csv
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import itertools
import pickle

class default_dict(dict):
    """ Dictionary that returns a default value when looking for a missing key """

    def __init__(self, default_value=None, *args, **kwargs):
        super(default_dict, self).__init__(*args, **kwargs)
        self.default_value = default_value

    def __missing__(self, key):
        return self.default_value


""" Selecting file to encode based on paramaters fed to the program"""
if len(sys.argv) < 2:
    filename = "train_stories.csv"
else:
    filename = sys.argv[1]

""" Selecting vocabulary to use from encoding based on paramaters fed to the program"""
if len(sys.argv) < 3:
    vocabname = None  # If nothing is specified, generate a new vocab!
else:
    vocabname = sys.argv[2]

if filename != "train_stories.csv" and vocabname == None:
    print("You probably  did not intend to generate a new vocabulary for a file other than train_stories file. Exiting")
    print("Did you mean to specify: processed/sentences.train_vocab.npy ?")
    exit(1)

#generates an array of the lines present in the file
texts = []
with codecs.open("data/"+ filename, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        texts.append(values[2:])

MAX_NB_WORDS = 20000 - 1 # cause pad
MAX_SEQUENCE_LENGTH = 30

flattened = [item for sublist in texts for item in sublist]

tokenizer = Tokenizer(MAX_NB_WORDS, oov_token='<unk>', filters='!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(flattened)
word_index = tokenizer.word_index #the dict values start from 1 so this is fine with zeropadding
index2word = {v: k for k, v in word_index.items()}
print('Found %s unique tokens' % len(word_index))
sequences = []
for story in texts:
    # print(story)
    sequences.append(
        # tokenizer.texts_to_sequences(story)
        pad_sequences(
            tokenizer.texts_to_sequences(story), maxlen=MAX_SEQUENCE_LENGTH, padding='post'
        )
    )

# data_1 = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
word_index["<pad>"] = 0

print(dict(itertools.islice(word_index.items(), MAX_NB_WORDS)))

# print(sequences[0:4])
# if vocabname is not None:
#     print(f"Loading vocab from data/{vocabname}")
#     vocabfile = dict(np.load("data/processed/" + vocabname).item().items())
#     vocab = default_dict(vocabfile["<unk>"], vocabfile)
# else:
#     vocab = {",": -1, "<unk>" : 0, "<bos>": 1, "<eos>": 2, "<pad>": 3}
#     id = 4
#     for w, _ in counter.most_common(20000 - 4):
#         vocab.update({w:id})
#         id += 1
#
#     #The words that are not present in the vocab should be encodded as the symbol <unk>
#     vocab = default_dict(vocab["<unk>"], vocab)

data = np.array(sequences, dtype=int)
np.save("data/processed/" + filename + "_vocab", dict(itertools.islice(word_index.items(), MAX_NB_WORDS)))
np.save("data/processed/" + filename, data)

with open('data/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

