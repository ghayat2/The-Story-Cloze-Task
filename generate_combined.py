import numpy as np
import functools
import tensorflow as tf
import data_utils as d
from embedding.sentence_embedder import SkipThoughtsEmbedder
import feature_extraction as F
from definitions import ROOT_DIR



FLAGS = tf.flags.FLAGS

CONTEXT_LENGTH = 4

encoder = SkipThoughtsEmbedder()


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
        self.dataset_iterator = tf_dataset.shuffle(5000).map(lambda t: t["sentence5"]).make_one_shot_iterator()

    def pick(self, context, N=1):
        return tf.stack([self.dataset_iterator.get_next()])
    
class PlainRandomPicker(Picker):

    def __init__(self, *args, **kwargs):
        
        super(PlainRandomPicker, self).__init__(*args, **kwargs)
        csv_path = f"{ROOT_DIR}/data/train_stories.csv"

        train_stories = tf.data.experimental.CsvDataset(
        filenames=csv_path,
        record_defaults=[tf.string for _ in range(5)],
        select_cols=[2, 3, 4, 5, 6],
        field_delim=",",
        use_quote_delim=True,
        header=True
    )
        self.dataset_iterator = train_stories.shuffle(5000).map(lambda x: x[4]).make_one_shot_iterator()

    def pick(self, context, N=1):
        return tf.stack([self.dataset_iterator.get_next()])


class EmbeddedBackPicker(Picker):
    def pick(self, context, N=1):
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
        .repeat(repeat_train_dataset) \
        .map(d.split_sentences, num_parallel_calls=threads) \
        .map(augment_fn, num_parallel_calls=threads) \
        .shuffle(buffer_size=5000) \
        .batch(batch_size, drop_remainder=True)

    return dataset


def get_features(*sentences, for_methods=("pronoun_contrast", "n_grams_overlap")):
    
    features1 = {method: [] for method in for_methods}
    features2 = {method: [] for method in for_methods}
    story = sentences[0:3]
    ending1 = sentences[4]
    ending2 = sentences[5]

    fe = F.FeatureExtractor(story)
    for method in for_methods:
        if method == "pronoun_contrast":
            features1[method].append(fe.pronoun_contrast(ending1))
            features2[method].append(fe.pronoun_contrast(ending2))
        elif method == "n_grams_overlap":
            features1[method].append(fe.n_grams_overlap(ending1))
            features2[method].append(fe.n_grams_overlap(ending2))
        
    features1 = tf.stack(features1, axis=0)
    features2 = tf.stack(features2, axis=0)
    return (story, features1, features2)
    
    
def get_skip_thoughts_data_iterator(augment_fn, threads=5, batch_size=1, repeat_train_dataset=5):

    csv_path = f"{ROOT_DIR}/data/train_stories.csv"

    def embed_sentences(*sentences):
        sentences = list(map(lambda s: s.numpy().decode("utf-8"), sentences))
        embeddings = encoder.encode(sentences)
        return tuple(embeddings)

    def map_stories(*embedded_sentences):
        return {f'sentence{i+1}': embedded_sentences[i] for i in range(5)}

    train_stories = tf.data.experimental.CsvDataset(
        filenames=csv_path,
        record_defaults=[tf.string for _ in range(5)],
        select_cols=[2, 3, 4, 5, 6],
        field_delim=",",
        use_quote_delim=True,
        header=True
    )

    return train_stories.map(lambda *sentences: tf.py_function(
            embed_sentences,
            inp=sentences,
            Tout=[tf.float32 for _ in range(5)]
        ))\
        .map(map_stories, num_parallel_calls=threads)\
        .map(d.split_skip_thoughts_sentences, num_parallel_calls=5) \
        .map(augment_fn, num_parallel_calls=threads) \
        .repeat(repeat_train_dataset) \
        .shuffle(5000) \
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


def get_skip_thoughts_eval_iterator(threads=5, batch_size=1, repeat_eval_dataset=5):
    from definitions import ROOT_DIR

    csv_path = f"{ROOT_DIR}/data/eval_stories.csv"

    def embed_sentences(*sentences):
        story = list(map(lambda s: s.numpy().decode("utf-8"), sentences[:-1]))
        embeddings = encoder.encode(story)
        return tf.stack(embeddings), sentences[-1]

    train_stories = tf.data.experimental.CsvDataset(
        filenames=csv_path,
        record_defaults=[tf.string for _ in range(6)] + [tf.int32],
        select_cols=[1, 2, 3, 4, 5, 6, 7],
        field_delim=",",
        use_quote_delim=True,
        header=True
    )

    # Zips the embeddings with the labels
    return train_stories.map(map_func=lambda *sentences: tf.py_function(
            embed_sentences,
            inp=sentences,
            Tout=[tf.float32, tf.int32]
        ))\
        .shuffle(buffer_size=100)\
        .repeat(repeat_eval_dataset)\
        .batch(batch_size, drop_remainder=True)
