# Data utilities
from gensim import models
import gensim.downloader as api 
import tensorflow as tf
import numpy as np

CONTEXT_LENGTH = 4
FLAGS = tf.flags.FLAGS

def makeSymbolStory(array, vocabLookup):
    return [makeSymbols(s, vocabLookup) for s in array.tolist()]

def makeSymbols(array, vocabLookup):
    """
    Convert array of integers into a sentence based on the dic argument
    """
    return list(vocabLookup[x] for x in array)

def endings(sentences):
    return [split_sentences(sentence)[1][0] for sentence in sentences]


def split_sentences(sentences):
    # Split sentences into [context], ending
    return sentences[0:CONTEXT_LENGTH, :], sentences[CONTEXT_LENGTH:, :]




def load_embedding(session, vocab, emb, path, dim_embedding, vocab_size):
    '''
          session        Tensorflow session object
          vocab          A dictionary mapping token strings to vocabulary IDs
          emb            Embedding tensor of shape vocabulary_size x dim_embedding
          path           Path to embedding file
          dim_embedding  Dimensionality of the external embedding.
        '''

    print("Loading external embeddings from %s" % path)

    model = models.KeyedVectors.load_word2vec_format(path, binary=False)
#     model = api.load("glove-twitter-25")  # download the model and return as object ready for use
    external_embedding = np.zeros(shape=(vocab_size, dim_embedding))
    matches = 0

    for tok, idx in vocab.item().items():
        if tok in model.vocab:
            external_embedding[idx] = model[tok]
            matches += 1
        else:
            print("%s not in embedding file" % tok)
            external_embedding[idx] = np.random.uniform(low=-0.25, high=0.25, size=dim_embedding)

    print("%d words out of %d could be loaded" % (matches, vocab_size))

    pretrained_embeddings = tf.placeholder(tf.float32, [None, None])
    assign_op = emb.assign(pretrained_embeddings)
    session.run(assign_op, {pretrained_embeddings: external_embedding})  # here, embeddings are actually set

def sentences_to_sparse_tensor(sentences):
    separated_sentences = sentences.split(".")
    dim1 = len(separated_sentences)
    dim2 = 0
    indices = []
    values = []
    for i in range(len(separated_sentences)):
        words = separated_sentences[i].split()
        dim2 = max(dim2, len(words))
        for j in range(len(words)):
            indices.append((i,j))
            values.append(words[j])
    return tf.SparseTensor(values=values, indices=indices, dense_shape=(dim1, dim2))

def randomize_labels(sentences):
    # The index-4'th sentence is the correct one
    classes = FLAGS.classes
    labels = tf.one_hot(0, depth = classes, dtype=tf.int32)
    indexes = tf.range(classes, dtype = tf.int32)
    shuffled = tf.random.shuffle(indexes)
    print("Randomized, sentences", sentences)
    return tf.gather(sentences, shuffled),\
           tf.cast(tf.argmax(tf.gather(labels, shuffled)), dtype=tf.int32)