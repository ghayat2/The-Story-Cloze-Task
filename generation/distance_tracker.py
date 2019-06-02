import datetime

from embedding.sentence_embedder import SkipThoughtsEmbedder
import tensorflow as tf
from definitions import ROOT_DIR
import numpy as np


class DistanceTracker:
    
    def save_closest_endings(self, endings, ng, buffer_size=50):
        with tf.python_io.TFRecordWriter(f'{ROOT_DIR}/data/embeddings/endings/near_endings.tfrecords') as writer:
            for endingIndex in range(len(endings)):
                a = datetime.datetime.now()
                np_endings = endings.numpy()
                np_endings = np_endings[np.random.randint(np_endings.shape[0], size=buffer_size), :]
                ng.sentence_embeddings = np_endings
                generated_endings = ng.generate_endings(correct_ending=endings[endingIndex], nb_samples=5)
                # Writes a feature to a .tfrecords file
                generated_endings_as_features = {}
                for i in range(len(generated_endings)):
                    generated_endings_as_features[f"generated_ending_{i + 1}"] = \
                        tf.train.Feature(float_list=tf.train.FloatList(value=generated_endings[i]))
                tf_example = tf.train.Example(features=tf.train.Features(feature=generated_endings_as_features))
                writer.write(tf_example.SerializeToString())
                print(f"One run: {datetime.datetime.now() - a}")
            # Dummy return value to please tensorflow
        return tf.constant(1, dtype=tf.int32)
