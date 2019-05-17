import numpy as np
import functools
import tensorflow as tf
import data_utils as d

FLAGS = tf.flags.FLAGS

CONTEXT_LENGTH = 4

class RandomPicker:

    # Initialize with a random dictionary of sentences
    def __init__(self, dictionary, length):
        self.dictionary = dictionary
        self.length = length

    # Pick a random sample sentence
    def pick(self, N = 1):
        # picks = []
        rand_index = tf.random.uniform([N], 0, self.length, dtype=tf.int32)
        return tf.gather(self.dictionary, rand_index)
    

def augment_data(context, endings,
                 randomPicker = None): # Augment the data

    ending1 = endings[0] # set, correct ending
    #ending2 = endings[1] # not set, all 0s

    print("Ending", endings)
    if randomPicker is not None:
        randomSentences = randomPicker.pick(N = FLAGS.classes - 1)
        print("Random", randomSentences)
        print("ending1", tf.expand_dims(ending1, axis = 0))
        # together = [endings, randomSentence]

    all_endings = tf.concat([tf.expand_dims(ending1, axis = 0), randomSentences], axis = 0)        
    print("All Endings", all_endings)

    randomized_endings, labels = d.randomize_labels(all_endings)

    print("Randomized endings", randomized_endings)
    print("labels", labels)

    return tf.concat([context, randomized_endings], axis=0), labels


def get_data_iterator(sentences,
                        augment_fn=functools.partial(augment_data),
                        threads=5,
                        batch_size=1,
                        repeat_train_dataset=5):

    # Create dataset from image and label paths
    dataset = tf.data.Dataset.from_tensor_slices(sentences) \
        .map(d.split_sentences, num_parallel_calls=threads) \
        .map(augment_fn, num_parallel_calls=threads) \
        .batch(batch_size, drop_remainder=True) \
        .repeat(repeat_train_dataset) \
        .shuffle(buffer_size=50)

    return dataset

def transform_labels_onehot(sentences, labels):
    one_hot = tf.one_hot(labels, FLAGS.classes, dtype=tf.int32)
    return sentences, one_hot

def get_eval_iterator(sentences, labels,
                        threads=5,
                        batch_size=1,
                        repeat_eval_dataset=5):

    # Create dataset from image and label paths
    dataset = tf.data.Dataset.from_tensor_slices((sentences, labels)) \
        .batch(batch_size, drop_remainder=True) \
        .repeat(repeat_eval_dataset) \
        .shuffle(buffer_size=50)\

    return dataset
