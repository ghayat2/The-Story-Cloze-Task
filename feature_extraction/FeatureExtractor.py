from functools import reduce

import tensorflow_transform as tft
import tensorflow as tf
from data_utils import sentences_to_sparse_tensor as to_sparse
import numpy as np
import nltk

"""
Inspired by https://aclweb.org/anthology/W17-0908
"""
class FeatureExtractor():

    def __init__(self, tf_session, story, ending1, ending2):
        """

        :param tf_session: Tensorflow session
        :param story: First 4 sentences of the story. Array of strings.
        :param ending1: First possible ending (str).
        :param ending2: Second possible ending (str).
        """
        self.sess = tf_session
        self.story = story
        self.ending1 = ending1
        self.ending2 = ending2


    def pronounContrast(self):
        get_pronouns = lambda strings: list(map(
            lambda word_and_tag: word_and_tag[0],
            filter(
                lambda word_and_tag: "PRP" in word_and_tag[1],
                nltk.pos_tag(nltk.word_tokenize(strings))
            )
        ))
        story_pronouns = get_pronouns(self._merged_story())
        endings_pronouns = (get_pronouns(self.ending1), get_pronouns(self.ending2))
        endings_pronouns_mismatch = ([], [])
        for story_pronoun in story_pronouns:
            for i in range(2):
                endings_pronouns_mismatch[i].append(0 if (story_pronoun in endings_pronouns[i]) else 1)
        return list(map(lambda mismatches: tf.constant(mismatches), endings_pronouns_mismatch))


    def nGramsOverlap(self, ngram_range=(1,3), character_count=True):
        """
        Counts the number of overlapping n-grams between the story and the two endings.

        :param ngram_range: Range of n-grams to inspect.
        :parameter character_count: If the values of the feature vector will be 0s and 1s or 0s and the varying
        number of characters of the matched n-grams.

        :return: One feature vector for each ending, having the length of the number of ngrams in the story. Each
        component of the vector is 0 if the n-gram is not in the ending. Otherwise, if word_count is True, the value
        is the amount of letters in the n-gram. If word_count is False, it is simply 1.
        """
        story_ngrams_vec, ending1_ngrams_vec, ending2_ngrams_vec = self._nGrams(ngram_range)
        # Retrieves the ngram 1d tensor of strings from the sparse vectors.
        story_ngrams = tf.unstack(story_ngrams_vec.values, num=story_ngrams_vec.indices.shape[0])

        # Our two feature vectors
        feature_vector_shape = story_ngrams_vec.indices.shape[0]
        ending1_ngram_overlap = tf.Variable(initial_value=np.zeros(feature_vector_shape),
                                            expected_shape=feature_vector_shape,
                                            dtype=tf.int32,
                                            name="ending1_ngram_overlap")
        ending2_ngram_overlap = tf.Variable(initial_value=np.zeros(feature_vector_shape),
                                            expected_shape=feature_vector_shape,
                                            dtype=tf.int32,
                                            name="ending2_ngram_overlap")

        # Checks for n-gram matchings.
        index = tf.Variable(0, dtype=tf.int32)
        for story_ngram in story_ngrams:
            update = lambda ending_ngram_overlap, ending_ngrams_vec: tf.scatter_update(
                ending_ngram_overlap,
                index,
                tf.multiply(
                    # Produces 1 if any ngram in the ending matches story_ngram and 0 otherwise
                    tf.cast(tf.math.reduce_any(tf.strings.regex_full_match(
                        ending_ngrams_vec.values,
                        story_ngram
                    )), dtype=tf.int32),
                    # Returns either 1 if word_count if False or the number of characters in the matched n-gram
                    tf.strings.length(story_ngram, unit="UTF8_CHAR") if (character_count) else tf.constant(1, tf.int32)
                )
            )

            ending1_ngram_overlap = update(ending1_ngram_overlap, ending1_ngrams_vec)
            ending2_ngram_overlap = update(ending2_ngram_overlap, ending2_ngrams_vec)

            index = index + tf.constant(1, dtype=tf.int32)
        return ending1_ngram_overlap, ending2_ngram_overlap


    def _nGrams(self, ngram_range):
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
                tokens=to_sparse(self.ending1),
                ngram_range=ngram_range,
                separator=" ",
                name="ending1_ngram_feature_vector"
            ),
            tft.ngrams(
                tokens=to_sparse(self.ending2),
                ngram_range=ngram_range,
                separator=" ",
                name="ending2_ngram_feature_vector"
            )
        )

    def _merged_story(self):
        return reduce(lambda sen1, sen2: sen1 + " " + sen2, self.story)


def test_ngrams():
    with tf.Graph().as_default():
        story = ["My man took a hat.", "He gave me the hat."]
        ending1 = "I took the hat."
        ending2 = "I tame tigers with hat."
        sess = tf.Session()
        with sess.as_default():
            ending1_ngram_overlaps, ending2_ngram_overlaps = FeatureExtractor(sess, story, ending1, ending2).nGramsOverlap()
            init = tf.global_variables_initializer()
            sess.run(init)
            print(ending1_ngram_overlaps.eval())
            print(ending2_ngram_overlaps.eval())


def test_pronoun_contrast():
    with tf.Graph().as_default():
        story = ["The man saw a boat.", "He bought the it."]
        ending1 = "He then proceeded to sail the boat"
        ending2 = "She then proceeded to sail the boat"
        sess = tf.Session()
        with sess.as_default():
            mis1, mis2 = FeatureExtractor(sess, story, ending1, ending2).pronounContrast()
            print(mis1.eval())
            print(mis2.eval())

test_pronoun_contrast()
