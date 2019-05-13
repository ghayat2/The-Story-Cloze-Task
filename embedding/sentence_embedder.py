import pandas

import embedding.skipthoughts as skipthoughts
from definitions import ROOT_DIR
import numpy as np

class SentenceEmbedder():

    def __init__(self, *args, **kwargs):
        super(SentenceEmbedder, self).__init__(*args, **kwargs)
        model = skipthoughts.load_model()
        self.encoder = skipthoughts.Encoder(model)

    def encode(self, data_to_encode, batch_size=1):
        return self.encoder.encode(data_to_encode, batch_size=batch_size)


    """
    Example to encode the 10 first stories of the evaluation set.
    Don't use this approach to encode the full dataset, as it will take too long to process at once.
    """
    def get_evaluation_set_embeddings(self):
        eval_set = pandas.io.parsers.read_csv(ROOT_DIR + '/data/cloze_eval.csv').values
        embeddings = list()
        for i in range(eval_set.shape[0]):
            embeddings.append(self.encode(eval_set[i][1:-1]))
            if (i == 10):
                break
        return embeddings

SentenceEmbedder().get_evaluation_set_embeddings()
