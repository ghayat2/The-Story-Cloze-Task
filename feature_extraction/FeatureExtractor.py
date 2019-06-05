from functools import reduce

import pandas as pd
import tensorflow_transform as tft
import tensorflow as tf
from data_pipeline.data_utils import sentences_to_sparse_tensor as to_sparse
import numpy as np
import nltk
from definitions import ROOT_DIR

"""
Inspired by https://aclweb.org/anthology/W17-0908
"""


class FeatureExtractor:

    def __init__(self, context):
        """

        :param context: First 4 sentences of the story. Array of strings.
        """
        self.story = context

    @staticmethod
    def generate_feature_records_train_set(tf_session, for_methods=("pronoun_contrast", "n_grams_overlap")):
        """
        Generates .tfrecords files containing the extracted features for all the endings in the file at
        the given filepath.
        :param for_methods: Feature extraction methods.
        """
        data = pd.read_csv(ROOT_DIR + "/data/train_stories.csv", delimiter=",")

        features = {method: [] for method in for_methods}
        for ind, sentences in data.iterrows():
            story = list(sentences[f"sentence{i}"] for i in range(1, 5))
            ending = sentences["sentence5"]
            fe = FeatureExtractor(story)
            for method in for_methods:
                if method == "pronoun_contrast":
                    features[method].append(fe.pronoun_contrast(ending))
                elif method == "n_grams_overlap":
                    features[method].append(fe.n_grams_overlap(ending))
                else:
                    raise NotImplementedError("Feature extraction method not implemented.")

        for method in for_methods:
            tensors = features[method]
            # Evens out tensors' dimensions by padding with 0s
            max_dim = max(tensor.shape[0].value for tensor in tensors)
            for i in range(len(tensors)):
                tensors[i] = tf.pad(
                    tensors[i],
                    tf.constant([[max_dim - tensors[i].shape[0].value, 0]])
                )
            ds = tf.data.Dataset.from_tensor_slices(tensors)
            with tf.python_io.TFRecordWriter(f'{ROOT_DIR}/data/features/{method}_train.tfrecords') as writer:
                # Writes a feature to a .tfrecords file
                def write_data(feature_val):
                    tf_example = tf.train.Example(features=tf.train.Features(feature={
                        "extracted_feature": tf.train.Feature(int64_list=tf.train.Int64List(value=feature_val.numpy()))
                    }))
                    writer.write(tf_example.SerializeToString())
                    return tf.constant([1])

                def write_tensor(t):
                    return tf.py_function(
                        write_data,
                        inp=[t],
                        Tout=tf.int32
                    )

                it = ds.map(write_tensor).make_initializable_iterator()
                tf_session.run(tf.global_variables_initializer())
                tf_session.run(tf.local_variables_initializer())
                tf_session.run(it.initializer)
                while True:
                    try:
                        tf_session.run(it.get_next())
                    except tf.errors.OutOfRangeError:
                        break

    @staticmethod
    def generate_feature_records_eval_set(for_methods=("pronoun_contrast", "n_grams_overlap")):
        raise NotImplementedError()

    def pronoun_contrast(self, ending):
        get_pronouns = lambda strings: list(map(
            lambda word_and_tag: word_and_tag[0],
            filter(
                lambda word_and_tag: "PRP" in word_and_tag[1],
                nltk.pos_tag(nltk.word_tokenize(strings))
            )
        ))
        story_pronouns = get_pronouns(self._merged_story())
        ending_pronouns = get_pronouns(ending)
        ending_pronouns_mismatch = []
        for story_pronoun in story_pronouns:
            ending_pronouns_mismatch.append(1 if (story_pronoun in ending_pronouns) else 0)
        return tf.constant(sum(ending_pronouns_mismatch), shape=[1])

    def n_grams_overlap(self, ending, ngram_range=(1, 3), character_count=True):
        """
        Counts the number of overlapping n-grams between the story and the two endings.

        :param ngram_range: Range of n-grams to inspect.
        :parameter character_count: If the values of the feature vector will be 0s and 1s or 0s and the varying
        number of characters of the matched n-grams.

        :return: One feature vector for each ending, having the length of the number of ngrams in the story. Each
        component of the vector is 0 if the n-gram is not in the ending. Otherwise, if word_count is True, the value
        is the amount of letters in the n-gram. If word_count is False, it is simply 1.
        """
        story_ngrams_vec, ending_ngrams_vec = self._n_grams(ending, ngram_range)
        # Retrieves the ngram 1d tensor of strings from the sparse vectors.
        story_ngrams = tf.unstack(story_ngrams_vec.values, num=story_ngrams_vec.indices.shape[0])

        # Our two feature vectors
        feature_vector_shape = story_ngrams_vec.indices.shape[0]
        ending_ngram_overlap = tf.Variable(initial_value=np.zeros(feature_vector_shape),
                                           expected_shape=feature_vector_shape,
                                           dtype=tf.int32,
                                           name="ending_ngram_overlap")

        # Checks for n-gram matchings.
        index = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32)
        for story_ngram in story_ngrams:
            ending_ngram_overlap = tf.scatter_update(
                ending_ngram_overlap,
                index,
                tf.multiply(
                    # Produces 1 if any ngram in the ending matches story_ngram and 0 otherwise
                    tf.cast(tf.math.reduce_any(tf.strings.regex_full_match(
                        ending_ngrams_vec.values,
                        story_ngram
                    )), dtype=tf.int32),
                    # Returns either 1 if word_count if False or the number of characters in the matched n-gram
                    tf.strings.length(story_ngram, unit="UTF8_CHAR") if character_count else tf.constant(1, tf.int32)
                )
            )

            index = index + tf.constant(1, dtype=tf.int32)
        return tf.reduce_sum(ending_ngram_overlap, keep_dims=True)

    def _n_grams(self, ending, ngram_range):
        """
        :return A list containing, in order: n-gram feature vector of story, n-gram feature vector of first
        ending and finally n-gram feature vector of second ending.
        """
        return (
            tft.ngrams(
                tokens=to_sparse(self._merged_story()),
                ngram_range=ngram_range,
                separator=" ",
                name="story_ngram_feature_vector"
            ),
            tft.ngrams(
                tokens=to_sparse(ending),
                ngram_range=ngram_range,
                separator=" ",
                name="ending_ngram_feature_vector"
            )
        )

    def _merged_story(self):
        return reduce(lambda sen1, sen2: sen1 + " " + sen2, self.story)


def save_all_features():
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            FeatureExtractor.generate_feature_records_train_set(sess)


def test_ngrams():
    with tf.Graph().as_default():
        story = ["My man took a hat.", "He gave me the hat."]
        ending1 = "I took the hat."
        ending2 = "I tame tigers with hat."
        sess = tf.Session()
        with sess.as_default():
            fe = FeatureExtractor(story)
            ending1_ngram_overlaps = fe.n_grams_overlap(ending1)
            ending2_ngram_overlaps = fe.n_grams_overlap(ending2)
            init = tf.global_variables_initializer()
            sess.run(init)
            print(ending1_ngram_overlaps.eval())
            print(ending2_ngram_overlaps.eval())


def test_pronoun_contrast():
    with tf.Graph().as_default():
        story = ["The man saw a boat.", "He bought it."]
        ending1 = "He then proceeded to sail the boat"
        ending2 = "She then proceeded to sail the boat"
        sess = tf.Session()
        with sess.as_default() as default_sess:
            fe = FeatureExtractor(story)
            mis1 = fe.pronoun_contrast(ending1)
            mis2 = fe.pronoun_contrast(ending2)
            print(mis1.eval())
            print(mis2.eval())
