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
        return [self.dictionary[np.random.randint(0, self.length), :] for i in range(N)]
    

def augment_data(context, endings,
                 randomPicker = None): # Augment the data

    print("Ending", endings)
    if randomPicker is not None:
        randomSentence = randomPicker.pick()
        print("Random", randomSentence)
        # together = [endings, randomSentence]

    allEndings = tf.concat([tf.expand_dims(endings, axis=0), randomSentence], axis=0)

    randomized_endings, labels = randomize_labels(allEndings)

    print("Randomized endings", randomized_endings)
    print("labels", labels)

    return tf.concat([context, randomized_endings], axis=0), labels

def randomize_labels(sentences):
    # The index-4'th sentence is the correct one
    classes = FLAGS.classes
    indexes = tf.range(classes, dtype=tf.int32)
    shuffled_indexes = tf.random.shuffle(indexes)
    index_of_zero = tf.cast(tf.argmin(shuffled_indexes), dtype=tf.int32)
    return tf.gather(sentences, shuffled_indexes), index_of_zero
    # return sentences, 0

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
