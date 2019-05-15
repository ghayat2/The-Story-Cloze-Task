# -*- coding: utf-8 -*-
import sys
import numpy as np
from collections import Counter
import codecs
import csv


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
    vocabname = None # If nothing is specified, generate a new vocab!
else:
    vocabname = sys.argv[2]

if filename != "train_stories.csv" and vocabname == None:
    print("You probably  did not intend to generate a new vocabulary for a file other than train_stories file. Exiting")
    print("Did you mean to specify: processed/sentences.train_vocab.npy ?")
    exit(1)

#generates an array of the lines present in the file
file = open("data/" + filename, "r")
lines = file.readlines()
del lines[0]
file.close()

#computes a dictionary with the 20K most frequent words in the file
counter = Counter()
for line in lines:
    sentences = line.strip().split(',')
    del sentences[0]
    for sentence in sentences:
        words = sentence.strip().split(' ')
        counter.update(words)
    
if vocabname is not None:
    print(f"Loading vocab from data/{vocabname}")
    vocabfile = dict(np.load("data/processed/" + vocabname).item().items())
    vocab = default_dict(vocabfile["<unk>"], vocabfile)
else:
    vocab = {",": -1, "<unk>" : 0, "<bos>": 1, "<eos>": 2, "<pad>": 3}
    id = 4
    for w, _ in counter.most_common(20000 - 4):
        vocab.update({w:id})
        id += 1
        
    #The words that are not present in the vocab should be encodded as the symbol <unk>
    vocab = default_dict(vocab["<unk>"], vocab)

#Encodes each word as its id in the training data
data = []

with codecs.open("data/"+ filename, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    
    for sentences in reader: 
        """ Deleting the story id and title"""
        del sentences[0]
        del sentences[1]
        
        encoded_words = []
        for i, sentence in enumerate(sentences):
            
            words = sentence.strip().split(' ')
            words.insert(0, "<bos>")
            words.append("<eos>")
            
            if len(words) <= 30 - 2: #Ignore sentences longer than 30 words (the 2 is to take symbols into account <bos> <eos>)
                for w in words:
                    encoded_words.append(vocab[w])
                            
                #pads symbol <pad> to have encoded sentences of the same length
                for j in range(30 - len(words)):
                    encoded_words.append(vocab["<pad>"])
                    
                if i != len(sentences)-1:
                    encoded_words.append(vocab[","])
              
        assert len(encoded_words) == 5*31 - 1
        
        data.append(encoded_words)
    
    

data = np.array(data, dtype=int)
np.save("data/processed/" + filename + "_vocab", dict(vocab))
np.save("data/processed/" + filename, data)





