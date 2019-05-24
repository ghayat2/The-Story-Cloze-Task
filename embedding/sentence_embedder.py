import sys

import pandas

import embedding.skipthoughts as skipthoughts
import datetime
from definitions import ROOT_DIR

import numpy as np


# Uses implementation from: https://github.com/ryankiros/skip-thoughts
class SentenceEmbedder:

    def __init__(self, *args, **kwargs):
        super(SentenceEmbedder, self).__init__(*args, **kwargs)
        model = skipthoughts.load_model()
        self.encoder = skipthoughts.Encoder(model)

    def encode(self, data_to_encode, batch_size=1):
        return self.encoder.encode(data_to_encode, batch_size=batch_size, verbose=False)

    def generate_embedded_training_set(self, training_set_path, save_file_path):
        """
        Generates skip-thoughts embeddings for the training dataset.
        :param training_set_path: Path to the training dataset.
        :param save_file_path: Path to save the results to.
        """
        self._generate_embedded_set(training_set_path, save_file_path, 5, 2)

    def generate_embedded_eval_set(self, testing_set_path, save_file_path):
        """
        Generates skip-thoughts embeddings for the evaluation dataset.
        :param testing_set_path: Path to the testing dataset.
        :param save_file_path: Path to save the results to.
        """
        self._generate_embedded_set(testing_set_path, save_file_path, 6, 1)

    @staticmethod
    def similarity(vec1, vec2):
        """Cosine similarity."""
        vec1 = vec1.reshape((4800))
        vec2 = vec2.reshape((4800))
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def _generate_embedded_set(self, set_path, save_file_path, nb_encodings_per_story, start_ind):
        dataset = pandas.io.parsers.read_csv(set_path).values
        embeddings = list()
        nbr_stories = dataset.shape[0]
        sys.stdout.write("Starting to encode " + str(nbr_stories) + " stories\n")
        batch_size = 1000
        for i in range(int(nbr_stories / batch_size)):
            a = datetime.datetime.now()
            ubound = min(batch_size, nbr_stories - i * batch_size)
            print(f"Encoding sentences {i*batch_size} to {i*batch_size+ubound-1}. {nbr_stories-i*batch_size} remaining.")
            to_encode = np.array(dataset[i*batch_size:i*batch_size+ubound, start_ind:start_ind+nb_encodings_per_story])
            to_encode = to_encode.flatten()
            encodings = self.encoder.encode(
                to_encode,
                batch_size=ubound,
                verbose=False
            )
            encodings = encodings.reshape((batch_size, nb_encodings_per_story, -1))
            for encoding in encodings:
                embeddings.append(encoding)
            b = datetime.datetime.now()
            sys.stdout.write(f"Time elapsed: {b-a}\n")
        np.save(save_file_path, np.array(embeddings))


def example_encode():
    embedder = SentenceEmbedder()
    s1 = embedder.encode(["My name is not what you think"])
    s2 = embedder.encode(["My username is different than what you think"])
    s4 = embedder.encode(["Beach or horses, give or take, life is full of extremes."])
    s3 = embedder.encode(["That is a totally unrelated sentence"])
    print("Similarity between s1 and s2: {}".format(embedder.similarity(s1, s2)))
    print("Similarity between s1 and s3: {}".format(embedder.similarity(s1, s3)))
    print("Similarity between s1 and s4: {}".format(embedder.similarity(s1, s4)))


def example_generate():
    SentenceEmbedder().generate_embedded_training_set(
        ROOT_DIR + "/data/train_stories.csv",
        ROOT_DIR + "/data/processed/train_stories_skip_thoughts"
    )


def example_load():
    sentences = np.load(ROOT_DIR + "/data/processed/train_stories_skip_thoughts.npy").astype(np.float32)
    print(sentences)
