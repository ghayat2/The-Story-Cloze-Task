from functools import reduce

import tensorflow_transform as tft
import tensorflow as tf
from data_utils import sentences_to_sparse_tensor as to_sparse

class FeatureExtractor():

    def __init__(self, story, ending1, ending2, ngram_range=(1,3)):
        """

        :param story: First 4 sentences of the story. Array of strings.
        :param ending1: First possible ending (str).
        :param ending2: Second possible ending (str).
        """
        self.story = story
        self.ending1 = ending1
        self.ending2 = ending2
        self.ngram_range = ngram_range

    def nGrams(self):
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
        story = ["My man has a hat.", "He gave me the hat."]
        ending1 = "I took the hat."
        ending2 = "I tame tigers."
        ngrams = FeatureExtractor(story, ending1, ending2).nGrams()
        sess = tf.Session()
        with sess.as_default():
            print(ngrams[2].eval())
            print(ngrams[0].eval())

test_ngrams()