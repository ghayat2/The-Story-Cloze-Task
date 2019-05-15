# Data utilities


def endings(sentences):
    return [split_sentences(sentence)[1] for sentence in sentences]


def split_sentences(sentences):
    # Split sentences into [context], ending
    return sentences[0:CONTEXT_LENGTH], [sentences[CONTEXT_LENGTH]]