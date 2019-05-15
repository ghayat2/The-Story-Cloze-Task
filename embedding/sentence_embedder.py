import pandas

import embedding.skipthoughts as skipthoughts
from definitions import ROOT_DIR

import numpy as np

# https://github.com/ryankiros/skip-thoughts
class SentenceEmbedder():

    def __init__(self, *args, **kwargs):
        super(SentenceEmbedder, self).__init__(*args, **kwargs)
        model = skipthoughts.load_model()
        self.encoder = skipthoughts.Encoder(model)

    def encode(self, data_to_encode, batch_size=1):
        return self.encoder.encode(data_to_encode, batch_size=batch_size)

    # def decode(self, sentence_embedding):
    #     dec = tools.load_model()
    #     text = tools.run_sampler(dec, sentence_embedding, beam_width=1, stochastic=False, use_unk=False)
    #     print(text)

    def similarity(self, vec1, vec2):
        """Cosine similarity."""
        vec1 = vec1.reshape([4800])
        vec2 = vec2.reshape([4800])
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


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

embedder = SentenceEmbedder()
s1 = embedder.encode(["My name is not what you think"])
s2 = embedder.encode(["My username is different than what you think"])
s4 = embedder.encode(["Beach or horses, give or take, life is full of extremes."])
s3 = embedder.encode(["That is a totally unrelated sentence"])
print("Similarity between s1 and s2: {}".format(embedder.similarity(s1, s2)))
print("Similarity between s1 and s3: {}".format(embedder.similarity(s1, s3)))
print("Similarity between s1 and s4: {}".format(embedder.similarity(s1, s4)))