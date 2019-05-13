import os
import glob
import functools
import tensorflow as tf
import math
import numpy as np

CONTEXT_LENGTH = 4

class RandomPicker:

    # Initialize with a random dictionary of sentences
    def __init__(self, dictionary):
        self.dictionary = dictionary

    # Pick a random sample sentence
    def pick(self):
        return np.random.sample(self.dictionary)


def split_sentences(sentences):
    # Split sentences into [context], ending
    return sentences[0:CONTEXT_LENGTH], [sentences[CONTEXT_LENGTH]]

def augment_data(context, endings,
                 randomPicker = None): # Augment the data

    if randomPicker is not None:
        randomSentence = randomPicker.pick()
        endings = [endings[0], randomSentence]

    return context, endings


def get_data_iterator(sentences,
                        augment_fn=functools.partial(augment_data),
                        threads=5,
                        batch_size=1,
                        repeat_train_dataset=5):

    # Create dataset from image and label paths
    dataset = tf.data.Dataset.from_tensor_slices(sentences) \
        .map(split_sentences, num_parallel_calls=threads) \
        .map(augment_fn, num_parallel_calls=threads) \
        .batch(batch_size, drop_remainder=True) \
        .repeat(repeat_train_dataset)

    return dataset