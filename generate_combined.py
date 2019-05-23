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
    def pick(self, context, N = 1):
        # picks = []
        rand_index = tf.random.uniform([N], 0, self.length, dtype=tf.int32)
        return tf.gather(self.dictionary, rand_index)


class BackPicker:

    # Pick a random sample sentence
    def pick(self, context, N = 1):
        rand_index = tf.random.uniform([N], 0, FLAGS.num_context_sentences, dtype=tf.int32)
        return tf.gather(context, rand_index)
    

def augment_data(context, endings,
                 randomPicker,
                 backPicker,
                 ratio_random = 0,
                 ratio_back = 0): # Augment the data

    ending1 = endings[0] # set, correct ending
    #ending2 = endings[1] # not set, all 0s

    print("Ending", endings)
    if ratio_random > 0 and ratio_back == 0:
        generatedSentence = randomPicker.pick(context, N = 1)
    elif ratio_back > 0 and ratio_random == 0:
        generatedSentence = backPicker.pick(context, N = 1)
    else:
        prob = tf.random.uniform([1], 0, 1, dtype=tf.float32)
        print(f"Picking both. Ratio for random: {ratio_random / (ratio_random + ratio_back)}")
        generatedSentence = tf.cond(tf.less(prob[0], ratio_random / (ratio_random + ratio_back)),
                                                       lambda: randomPicker.pick(context, N = 1),
                                                       lambda: backPicker.pick(context, N = 1)
                                                       )

    print("Random", generatedSentence)

    all_endings = tf.concat([tf.expand_dims(ending1, axis = 0), generatedSentence], axis = 0)
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
        .shuffle(buffer_size=5000) \
        .repeat(repeat_train_dataset) \
        .batch(batch_size, drop_remainder=True)

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
        .shuffle(buffer_size=5000) \
        .repeat(repeat_eval_dataset) \
        .batch(batch_size, drop_remainder=True) \

    return dataset
