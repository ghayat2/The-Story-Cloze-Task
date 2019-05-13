import pandas

import embedding.skipthoughts as skipthoughts
from definitions import ROOT_DIR
from numpy import genfromtxt

class SentenceEmbedder():

    def encode(self, data_to_encode, output_filename):
        model = skipthoughts.load_model()
        encoder = skipthoughts.Encoder(model)
        vectors = encoder.encode(data_to_encode)
        vectors.save(ROOT_DIR + "/data/embeddings/skip_thoughts/embedded/" + output_filename)


sEmb = SentenceEmbedder()
eval_set = pandas.io.parsers.read_csv(ROOT_DIR + '/data/cloze_eval.csv').values
sEmb.encode(eval_set, "cloze_eval_embedded")
