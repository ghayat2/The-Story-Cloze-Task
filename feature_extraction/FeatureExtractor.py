from functools import reduce

import tensorflow_transform as tft
import tensorflow as tf
from data_utils import sentences_to_sparse_tensor as to_sparse
import numpy as np

class FeatureExtractor():

    def __init__(self, tf_session, story, ending1, ending2, ngram_range=(1,3)):
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
        self.ngram_range = ngram_range


    def nGramsOverlap(self, word_count=True):
        """
        Counts the number of overlapping n-grams between the story and the two endings.
        :return: One feature vector for each ending, having the length of the number of ngrams in the story. Each
        component of the vector is 0 if the n-gram is not in the ending. Otherwise, if word_count is True, the value
        is the amount of letters in the n-gram. If word_count is False, it is simply 1.
        """
        story_ngrams_vec, ending1_ngrams_vec, ending2_ngrams_vec = self._nGrams()
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
                    tf.cast(tf.math.reduce_any(tf.strings.regex_full_match(
                        ending_ngrams_vec.values,
                        story_ngram
                    )), dtype=tf.int32),
                    tf.strings.length(story_ngram, unit="UTF8_CHAR") if (word_count) else tf.constant(1, tf.int32)
                )
            )

            ending1_ngram_overlap = update(ending1_ngram_overlap, ending1_ngrams_vec)
            ending2_ngram_overlap = update(ending2_ngram_overlap, ending2_ngrams_vec)

            index = index + tf.constant(1, dtype=tf.int32)
        return ending1_ngram_overlap, ending2_ngram_overlap


    def _nGrams(self):
        """
        :return A list containing, in order: n-gram feature vector of story, n-gram feature vector of first
        ending and finally n-gram feature vector of second ending.
        """
        return (
            tft.ngrams(
                tokens=to_sparse(reduce(lambda sen1, sen2: sen1 + sen2, self.story)),
                ngram_range=self.ngram_range,
                separator=" ",
                name="story_ngram_feature_vector"
            ),
            tft.ngrams(
                tokens=to_sparse(self.ending1),
                ngram_range=self.ngram_range,
                separator=" ",
                name="ending1_ngram_feature_vector"
            ),
            tft.ngrams(
                tokens=to_sparse(self.ending2),
                ngram_range=self.ngram_range,
                separator=" ",
                name="ending2_ngram_feature_vector"
            )
        )




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
