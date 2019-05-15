# Data utilities
import tensorflow as tf

def endings(sentences):
    return [split_sentences(sentence)[1] for sentence in sentences]


def split_sentences(sentences):
    # Split sentences into [context], ending
    return sentences[0:CONTEXT_LENGTH], [sentences[CONTEXT_LENGTH]]


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
