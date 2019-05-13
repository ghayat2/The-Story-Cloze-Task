import embedding.skipthoughts as skipthoughts
from definitions import ROOT_DIR

class SentenceEmbedder():

    def encode(self, data_to_encode, output_filename):
        model = skipthoughts.load_model()
        encoder = skipthoughts.Encoder(model)
        vectors = encoder.encode(data_to_encode)
        vectors.save(ROOT_DIR + "/data/embeddings/skip_thoughts/embedded/" + output_filename)