import tensorflow as tf
from data_pipeline.story import Story
from embedding.sentence_embedder import SkipThoughtsEmbedder
from feature_extraction.FeatureExtractor import FeatureExtractor

CONTEXT_LENGTH = 4
encoder = SkipThoughtsEmbedder()


def create_story(*story_tensors, random_picker=None, back_picker=None, ratio_random=0, ratio_back=0):
    def create_story_python(*raw_data):
        if len(raw_data) != 5 and len(raw_data) != 7:
            raise AssertionError("Data point should contain at either 5 (train) or 7 (eval) elements.")

        data = list(map(lambda t: t.numpy().decode("utf-8 "), raw_data[:6]))
        # Takes care of adding the label
        if len(raw_data) == 7:
            data.append(raw_data[6].numpy())

        story = Story(context=data[:4], ending1=data[4])
        if len(data) == 5:
            # Story from the training set
            story.set_labels((1, 0))
            story = augment_data(story, random_picker, back_picker, ratio_random, ratio_back)
        if len(data) == 7:
            # Story from the evaluation set
            story.ending2 = data[5]
            story.set_labels((1, 0) if data[6] == 1 else (0, 1))
        # return story
        story = get_features(story)
        return compute_sentence_embeddings(story)

    return tf.py_function(create_story_python, inp=story_tensors, Tout=[
        tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.int32
    ])


def augment_data(story, random_picker, back_picker, ratio_random=0, ratio_back=0):  # Augment the data

    if ratio_random > 0 and ratio_back == 0:
        generated_ending = random_picker.pick(story.context, N=1)
    elif ratio_back > 0 and ratio_random == 0:
        generated_ending = back_picker.pick(story.context, N=1)
    else:
        prob = tf.random.uniform([1], 0, 1, dtype=tf.float32)
        generated_ending = tf.cond(
            tf.less(prob[0], ratio_random / (ratio_random + ratio_back)),
            lambda: random_picker.pick(story.context),
            lambda: back_picker.pick(story.context)
        )
    # Sets fake ending
    story.ending2 = generated_ending
    # Randomize labels for training
    return story.randomize_labels()


def get_features(story, for_methods=("pronoun_contrast", "n_grams_overlap")):
    features1 = {method: [] for method in for_methods}
    features2 = {method: [] for method in for_methods}

    fe = FeatureExtractor(story.context)
    for method in for_methods:
        if method == "pronoun_contrast":
            features1[method].append(fe.pronoun_contrast(story.ending1))
            features2[method].append(fe.pronoun_contrast(story.ending2))
        elif method == "n_grams_overlap":
            features1[method].append(fe.n_grams_overlap(story.ending1))
            features2[method].append(fe.n_grams_overlap(story.ending2))

    story.features_ending_1 = tf.cast(tf.concat(list(features1.values()), axis=0), dtype=tf.float32)
    story.features_ending_2 = tf.cast(tf.concat(list(features2.values()), axis=0), dtype=tf.float32)
    return story


def compute_sentence_embeddings(story):
    def embed_sentences(*sentences):
        sentences = list(map(lambda s: s.numpy().decode("utf-8"), sentences))
        embeddings = encoder.encode(sentences)
        return tuple(embeddings)

    # Embeds the context and the two endings
    story.context = tf.py_function(
        embed_sentences,
        inp=story.context,
        Tout=[tf.float32 for _ in range(4)]
    )
    story.ending1 = tf.py_function(
        embed_sentences,
        inp=[story.ending1],
        Tout=[tf.float32]
    )
    story.ending2 = tf.py_function(
        embed_sentences,
        inp=[story.ending2],
        Tout=[tf.float32]
    )
    return story.context, story.ending1, story.ending2, story.features_ending_1, story.features_ending_2, story.labels


def endings(sentences):
    return [split_sentences(sentence)[1][0] for sentence in sentences]


def split_sentences(sentences):
    # Split sentences into [context], ending
    return sentences[0:CONTEXT_LENGTH, :], sentences[CONTEXT_LENGTH:, :]
