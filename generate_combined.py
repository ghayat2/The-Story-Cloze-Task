import numpy as np
import functools
import tensorflow as tf
import data_utils as d

FLAGS = tf.flags.FLAGS

CONTEXT_LENGTH = 4


class Picker:

    def pick(self, context, N):
        raise NotImplementedError()


class RandomPicker(Picker):

    # Initialize with a random dictionary of sentences
    def __init__(self, dictionary, length):
        self.dictionary = dictionary
        self.length = length

    # Pick a random sample sentence
    def pick(self, context, N=1):
        # picks = []
        rand_index = tf.random.uniform([N], 0, self.length, dtype=tf.int32)
        return tf.gather(self.dictionary, rand_index)


class BackPicker(Picker):

    # Pick a random sample sentence
    def pick(self, context, N=1):
        rand_index = tf.random.uniform([N], 0, FLAGS.num_context_sentences, dtype=tf.int32)
        return tf.gather(context, rand_index)


class EmbeddedRandomPicker(Picker):

    def __init__(self, tf_dataset, *args, **kwargs):
        super(EmbeddedRandomPicker, self).__init__(*args, **kwargs)
        self.tf_dataset = tf_dataset

    def pick(self, context, N=1):
        return tf.stack(
            [tf.data.experimental.sample_from_datasets([self.tf_dataset]).make_one_shot_iterator().get_next()["sentence5"]]
        )


class EmbeddedBackPicker(Picker):
    def pick(self, context, N=1):
        rand_index = tf.random.uniform([N], 0, FLAGS.num_context_sentences, dtype=tf.int32)
        return tf.gather(context, rand_index)


def augment_data(context, endings, pickers=()):
    """
    Augment the data
    """
    ending1 = endings[0] # set, correct ending
    #ending2 = endings[1] # not set, all 0s

    print("Ending", endings)
    allGenerated = []
    for picker in pickers:
        generatedSentences = picker.pick(context, N = 1)
        allGenerated.append(generatedSentences)
        print("Random", generatedSentences)

    allGenerated = tf.concat(allGenerated, axis=0)

    all_endings = tf.concat([tf.expand_dims(ending1, axis = 0), allGenerated], axis = 0)
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


def get_skip_thoughts_data_iterator(augment_fn, threads=5, batch_size=1, repeat_train_dataset=5):
    from embedding.sentence_embedder import SkipThoughtsEmbedder
    return SkipThoughtsEmbedder.get_train_tf_dataset()\
        .map(d.split_skip_thoughts_sentences, num_parallel_calls=5)\
        .map(augment_fn, num_parallel_calls=threads)\
        .shuffle(5000).repeat(repeat_train_dataset)\
        .batch(batch_size, drop_remainder=True)


def transform_labels_onehot(sentences, labels, threads=5):
    one_hot = tf.one_hot(labels, FLAGS.classes, dtype=tf.int32).map(d.split_sentences, num_parallel_calls=threads)
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


def get_skip_thoughts_eval_iterator(labels, threads=5, batch_size=1, repeat_eval_dataset=5):
    from embedding.sentence_embedder import SkipThoughtsEmbedder
    eval_dataset = SkipThoughtsEmbedder.get_eval_tf_dataset().map(d.tensorize_dict, num_parallel_calls=threads)
    labels_dataset = tf.data.Dataset.from_tensor_slices(labels)
    # Zips the embeddings with the labels
    return tf.data.Dataset.zip((eval_dataset, labels_dataset))\
        .shuffle(buffer_size=5000)\
        .repeat(repeat_eval_dataset)\
        .batch(batch_size, drop_remainder=True)
