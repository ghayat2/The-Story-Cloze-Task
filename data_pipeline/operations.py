from functools import reduce

import tensorflow as tf
from data_pipeline.story import Story
from embedding.sentence_embedder import SkipThoughtsEmbedder
from feature_extraction.FeatureExtractor import FeatureExtractor
import numpy as np

CONTEXT_LENGTH = 4



def create_story(*story_tensors,
                 sentence_embeddings=True,
                 random_picker=None,
                 back_picker=None,
                 ratio_random=0,
                 ratio_back=0, 
                 vocabLookup=None,
                 vocab= None):

        
    def create_story_python(*raw_data):
        if len(raw_data) != 5 and len(raw_data) != 7:
            raise AssertionError("Data point should contain at either 5 (train) or 7 (eval) elements.")

        data = list(map(lambda t: t.numpy(), raw_data[:6]))
        if sentence_embeddings:
            data = list(map(lambda t: t.decode("utf-8 "), data))
        else:
             translated_data=np.array(data)
             translated_data = list(map(lambda ids: reduce(lambda words, id2: words + " " + vocabLookup[id2], ids, ""), data))

        # Takes care of adding the label
        if len(raw_data) == 7:
            data.append(raw_data[6].numpy())

        story = Story(context=translated_data[:4], ending1=translated_data[4])
        if len(data) == 5:
            # Story from the training set
            story.set_labels((1, 0))
            story = augment_data(story, random_picker, back_picker, ratio_random, ratio_back)
#            print("shape of ending2", np.array(story.ending2).shape)
#            print("ending2", story.ending2)
#            print("ending2[0]", list(filter(lambda x: x not in '!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', story.ending2.strip().split(" ")))[0])
#            print("data.shape before", np.array(data).shape)
#           
            data.append(encode(list(filter(lambda x: x not in '!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', story.ending2.strip().split(" "))), vocab))
#            print("data.shape after", np.array(data).shape)


        if len(data) == 7:
            # Story from the evaluation set
            story.ending2 = translated_data[5]
            story.set_labels((1, 0) if data[6] == 1 else (0, 1))
        # return story
        story = get_features(story)
        if sentence_embeddings:
            story = compute_sentence_embeddings(story)
        
#        print("soty context[0]", story.context[0])
#        print("soty ending1", len(story.ending1))
#        print("soty feature", len(story.features_ending_1))
#        print("soty labels", len(story.labels))
        return data[0:4], data[4], data[5], story.features_ending_1, story.features_ending_2, story.labels

    if sentence_embeddings:
        output_types = [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.int32]
    else:
        output_types = [tf.int32, tf.int32, tf.int32, tf.float32, tf.float32, tf.int32]
    return tf.py_function(create_story_python, inp=story_tensors, Tout=output_types)


def encode(sentence, vocab):
    return np.array(list(map(lambda word: helper(word, vocab), sentence)))

def helper(word, vocab):
    try :
        return vocab[word]
    except: return 0 

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
    features1 = []
    features2 = []

    fe = FeatureExtractor(story.context)
    for method in for_methods:
        if method == "pronoun_contrast":
            features1.append(fe.pronoun_contrast(story.ending1))
            features2.append(fe.pronoun_contrast(story.ending2))
        elif method == "n_grams_overlap":
            features1.append(fe.n_grams_overlap(story.ending1))
            features2.append(fe.n_grams_overlap(story.ending2))

    story.features_ending_1 = tf.cast(tf.concat(features1, axis=0), dtype=tf.float32)
    story.features_ending_2 = tf.cast(tf.concat(features2, axis=0), dtype=tf.float32)
    return story


def compute_sentence_embeddings(story):
    
    encoder = SkipThoughtsEmbedder()
    def embed_sentences(*sentences):
        sentences = list(map(lambda s: s.numpy().decode("utf-8"), sentences))
        _embeddings = encoder.encode(sentences)
        return tuple(_embeddings)

    # Embeds the context and the two endings
    embeddings = tf.py_function(
        embed_sentences,
        inp=story.context + [story.ending1, story.ending2],
        Tout=[tf.float32 for _ in range(len(story.context) + 2)]
    )
    story.context = embeddings[:len(story.context)]
    story.ending1 = [embeddings[len(story.context)]]
    story.ending2 = [embeddings[-1]]
    return story


def endings(sentences):
    return [split_sentences(sentence)[1][0] for sentence in sentences]


def split_sentences(sentences):
    # Split sentences into [context], ending
    return sentences[0:CONTEXT_LENGTH, :], sentences[CONTEXT_LENGTH:, :]
