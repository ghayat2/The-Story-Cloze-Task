import datetime
import operator

import pandas

from definitions import ROOT_DIR
from embedding.sentence_embedder import SkipThoughtsEmbedder as SentenceEmbedder
from generation.distance_tracker import DistanceTracker
from generation.ending_generator import EndingGenerator
import numpy as np
import tensorflow as tf


class NearGeneration(EndingGenerator):

    def __init__(self,
                 sentence_embeddings,
                 encoder=None,
                 embeddings_hashable=False,
                 distance_function=lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)),  # np.linalg.norm(np.subtract(a, b), 2),
                 *args, **kwargs):
        """
        :param sentence_embeddings: A vector of sentence embeddings of any length.
        :param embeddings_hashable: If the sentence embeddings are hashable python objects. If not, they'll be transformed to
        tuples on the fly.
        :param distance_function: A distance function to apply to the embeddings.
        """
        super(NearGeneration, self).__init__(*args, **kwargs)
        if encoder is None:
            encoder = SentenceEmbedder()
        self.sentence_embeddings = sentence_embeddings
        self.distances_from_optimal_dist = {}
        self.embeddings_hashable = embeddings_hashable
        self.encoder = encoder
        self.dist_function = distance_function

    def generate_endings(self,
                         correct_ending,
                         nb_samples,
                         optimal_endings_distance=0.78409,
                         is_encoded=True,
                         is_hashable=False):
        """
        :param correct_ending: A correct story ending.
        :param optimal_endings_distance: The ideal distance between :correct_ending and the returned generated ending.
        :param is_encoded: If correct_ending is already an embedded vector. If this is false, then skip thoughts
        is used to encode it.
        :param is_hashable: If correct_ending is hashable
        :return: An ending from self.sentence_embeddings that is as close as possible to the given optima distance.
        """
        # Embed the ending if it isn't already
        if not is_encoded:
            correct_ending = self._get_encoder().encode(correct_ending)[0]
        # Ensures the ending is a hashable object
        if not is_encoded or not is_hashable:
            correct_ending = tuple(correct_ending)
        # Checks if we have already calculated some distances for this ending
        if not (correct_ending in self.distances_from_optimal_dist):
            self.distances_from_optimal_dist[correct_ending] = {}
        # Retrieves the distances
        ending_distances = self.distances_from_optimal_dist[correct_ending]
        # Starts calculating or simply retrieving the distances if they've already been computed
        for sentence_embedding in self.sentence_embeddings:
            if not self.embeddings_hashable:
                sentence_embedding = tuple(sentence_embedding)
            if sentence_embedding != correct_ending:
                if sentence_embedding not in ending_distances:
                    # Computes distance from correct ending to current sentence embedding
                    # l1 norm between optimal distance and the actual distance for this sentence embedding
                    ending_distances[sentence_embedding] = \
                        abs(optimal_endings_distance - self.dist_function(correct_ending, sentence_embedding))
        # Takes the vectors having the closest to optimal distance
        closest_endings = sorted(self.distances_from_optimal_dist[correct_ending].items(), key=operator.itemgetter(1))[:nb_samples]
        return list(map(lambda close_ending: close_ending[0], closest_endings))

    def get_evaluation_set_avg_distance(self):
        eval_set = pandas.read_csv(ROOT_DIR + '/data/eval_stories.csv', header=0)
        avg_distance = 0.0
        set_size = eval_set.shape[0]
        for i, sentences in eval_set.iterrows():
            dist = self.dist_function(*self._get_encoder().encode([sentences["RandomFifthSentenceQuiz1"],
                                                                   sentences["RandomFifthSentenceQuiz2"]]))
            avg_distance += dist / set_size
        return avg_distance

    def _get_encoder(self):
        return self.encoder

    @staticmethod
    def generate_training_set_endings(nb_training_samples=88000):
        with tf.Graph().as_default():
            encoder = SentenceEmbedder()
            training_set_embeddings = SentenceEmbedder.get_train_tf_dataset()
            dt = DistanceTracker()
            training_endings = training_set_embeddings\
                .map(lambda data: data["sentence5"], num_parallel_calls=5)\
                .batch(batch_size=nb_training_samples)\
                .map(lambda all_endings: tf.py_function(
                    lambda endings: dt.save_closest_endings(
                        endings, NearGeneration(sentence_embeddings=endings, encoder=encoder)),
                    inp=[all_endings],
                    Tout=tf.int32
                ), num_parallel_calls=5)
            # Start fetching data
            it = training_endings.make_initializable_iterator()
            compute_next_near_endings = it.get_next()
            sess = tf.Session()
            with sess.as_default():
                sess.run(it.initializer)
                i = 0
                while True:
                    try:
                        sess.run(compute_next_near_endings)
                    except tf.errors.OutOfRangeError:
                        break
                print(i)

    @staticmethod
    def get_training_set_near_endings():
        filepath = ROOT_DIR + "/data/embeddings/endings/near_endings.tfrecords"

        def extract_fn(data_record):
            return tf.parse_single_example(
                data_record,
                {f"generated_ending_{ind}": tf.FixedLenFeature(shape=4800, dtype=tf.float32) for ind in range(1, 6)}
            )

        generated_endings = tf.data.TFRecordDataset(filepath)
        return generated_endings.map(extract_fn)


def test_distances():
    sentences = [
        "Tyler has released a new album called Igor.",
        "I've been listening to it all the time ever since.",
        "I like strawberries.",
        "Pizzas sometimes come with pineapple."
    ]
    embedder = SentenceEmbedder()
    embedded_sentences = embedder.encode(sentences)
    generator = NearGeneration(
        embedded_sentences
    )
    false_ending = generator.generate_endings(embedded_sentences[0], nb_samples=1)
    for i in range(len(embedded_sentences)):
        if abs(sum(embedded_sentences[i]) - sum(false_ending)) < 1e-2:
            print(sentences[i])


def test_avg_distance():
    ng = NearGeneration(sentence_embeddings=None)
    print(ng.get_evaluation_set_avg_distance())  # prints 0.7840946715410734


def test_eval_dataset():
    eval_dataset = np.load(ROOT_DIR + "/data/processed/eval_stories_skip_thoughts.npy")[:, 4, :]
    sentences = pandas.read_csv(ROOT_DIR + "/data/eval_stories.csv")["RandomFifthSentenceQuiz1"]
    print(sentences.iloc[480])
    ng = NearGeneration(sentence_embeddings=eval_dataset)
    new_ending = ng.generate_endings(eval_dataset[480], is_encoded=True, nb_samples=1)
    for i in range(len(eval_dataset)):
        if abs(sum(eval_dataset[i]) - sum(new_ending)) < 1e-2:
            print(sentences.iloc[i])


def read_near_endings_tfrecords():
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            near_endings = NearGeneration.get_training_set_near_endings()
            ite = near_endings.make_initializable_iterator()
            sess.run(ite.initializer)
            i = 0
            while True:
                try:
                    print(sess.run(ite.get_next()))
                    i += 1
                    print(i)
                except tf.errors.OutOfRangeError:
                    break


NearGeneration.generate_training_set_endings()
