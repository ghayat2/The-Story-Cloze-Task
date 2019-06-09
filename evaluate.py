import functools
import sys
import tensorflow as tf
import numpy as np
from definitions import ROOT_DIR
import pandas as pd
from data_pipeline.generate_combined import create_story

from models.bidirectional_lstm import BiDirectional_LSTM

""" Selecting the adequate experiment and checkpoint file to evaluate based on arguments fed to the program """

LSTM_SIZE = 1000
CONTEXT_LENGTH = 4
NB_ENDINGS = 2
NB_SENTENCES = CONTEXT_LENGTH + NB_ENDINGS

if len(sys.argv) < 2:
    raise AssertionError("Please specify the checkpoint folder name")
CHECKPOINT_FILE = sys.argv[1]

"""Flags representing constants of our project """

# Data loading parameters
tf.flags.DEFINE_string("group_number", "19", "Our group number")
tf.flags.DEFINE_string("data_sentences_vocab_path", f"{ROOT_DIR}/data/processed/train_stories.csv_vocab.npy",
                       "Path to vocabulary file.")
# Test parameters
tf.flags.DEFINE_bool("predict", True, "If predicting labels for the test-stories.csv file or assessing the performance"
                                      "instead, using test_for_report-stories_labels.csv")
tf.flags.DEFINE_integer("batch_size", 1, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", f"./runs/{CHECKPOINT_FILE}/checkpoints/",
                       "Checkpoint directory from training run")

# Tensorflow Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# Model Parameters
tf.flags.DEFINE_integer("rnn_cell_size", LSTM_SIZE, "LSTM Size (default: 1000)")
tf.flags.DEFINE_string("rnn_cell", "LSTM", "Type of rnn cell")
tf.flags.DEFINE_boolean("enable_dropout", False, "Enable the dropout layer")
tf.flags.DEFINE_boolean("use_skip_thoughts", True, "True if skip thoughts embeddings should be used")
tf.flags.DEFINE_integer("sentence_len", 30, "Length of sentence")
tf.flags.DEFINE_integer("vocab_size", 20000, "Size of the vocabulary")
tf.flags.DEFINE_string("attention", None,
                       'Attention type (add ~ Bahdanau, mult ~ Luong, None). Only for Roemmele ''models.')
tf.flags.DEFINE_integer("attention_size", 1000, "Attention size.")
tf.flags.DEFINE_bool("used_features", True, "If features were used during training.")
tf.flags.DEFINE_bool("use_pronoun_contrast", True, "Whether the pronoun contrast feature vector should be added to the"
                                                    " networks' input.")
tf.flags.DEFINE_bool("use_n_grams_overlap", True, "Whether the n grams overlap feature vector should be added to the "
                                                  "network's input.")
tf.flags.DEFINE_bool("use_sentiment_analysis", True, "Whether to use the sentiment intensity analysis (4 dimensional "
                                                     "vectors)")
tf.flags.DEFINE_integer("num_sentences_train", 5, "Number of sentences in training set (default: 5)")
tf.flags.DEFINE_integer("sentence_length", 30, "Sentence length (default: 30)")
tf.flags.DEFINE_integer("word_embedding_dimension", 100, "Word embedding dimension size (default: 100)")
tf.flags.DEFINE_integer("num_context_sentences", 4, "Number of context sentences")
tf.flags.DEFINE_integer("classes", 2, "Number of output classes")
tf.flags.DEFINE_integer("num_eval_sentences", 2, "Number of eval sentences")

tf.flags.DEFINE_integer("sentence_embedding_length", 4800, "Length of the sentence embeddings")

tf.flags.DEFINE_integer("num_neg_random", 3, "Number of negative random endings")
tf.flags.DEFINE_integer("num_neg_back", 2, "Number of negative back endings")
tf.flags.DEFINE_integer("ratio_neg_random", 4, "Ratio of negative random endings")
tf.flags.DEFINE_integer("ratio_neg_back", 2, "Ratio of negative back endings")

tf.flags.DEFINE_float("dropout_rate", 0.7, "Dropout rate")

tf.flags.DEFINE_string("path_embeddings", "data/wordembeddings-dim100.word2vec", "Path to the word2vec embeddings")
tf.flags.DEFINE_string("embeddings", "w2v", "embedding types. Options are: w2v, w2v_google, glove")

tf.flags.DEFINE_integer("hidden_layer_size", 100, "Size of hidden layer")
tf.flags.DEFINE_integer("rnn_num", 2, "Number of RNNs")
tf.flags.DEFINE_integer("feature_integration_layer_output_size", 100, "Number of outputs from the dense layer after the"
                                                                      " RNN cell that includes the features")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value.value))
print("")

""" Processing of testing data"""

# Load data
if FLAGS.predict:
    labels = None
    filepath = f"{ROOT_DIR}/data/test-stories.csv"
else:
    filepath = f"{ROOT_DIR}/data/test_for_report-stories_labels.csv"
    labels = pd.read_csv(filepath_or_buffer=filepath, sep=',', usecols=["AnswerRightEnding"])

EMBEDDING_SIZE = 4800 if FLAGS.use_skip_thoughts else 100
FEATURES_SIZE = 22 if FLAGS.used_features else 0

print("Loading and preprocessing test dataset \n")
# x_test = np.load(filepath).astype(np.str)
x_test = pd.read_csv(filepath_or_buffer=filepath, sep=',',
                     usecols=["InputSentence1", "InputSentence2", "InputSentence3", "InputSentence4",
                              "RandomFifthSentenceQuiz1", "RandomFifthSentenceQuiz2"])
vocab = np.load(FLAGS.data_sentences_vocab_path, allow_pickle=True)  # vocab contains [symbol: id]
vocabLookup = dict((v, k) for k, v in vocab.item().items())  # flip our vocab dict so we can easy lookup [id: symbol]

""" Evaluating model"""

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)

    with sess.as_default():
        # Placeholders for stories and labels
        if FLAGS.use_skip_thoughts:
            output_types = [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]
        else:
            output_types = [tf.int32, tf.int32, tf.int32, tf.float32, tf.float32]
        shapes = (
            [FLAGS.batch_size, 1, EMBEDDING_SIZE], [FLAGS.batch_size, 1, EMBEDDING_SIZE],
            [FLAGS.batch_size, 1, EMBEDDING_SIZE], [FLAGS.batch_size, 1, FEATURES_SIZE],
            [FLAGS.batch_size, 1, FEATURES_SIZE]
        )
        next_story = list(map(lambda i: tf.placeholder(tf.string, shape=[FLAGS.batch_size, 1]), range(len(output_types))))
        next_label = tf.placeholder(tf.int32, shape=[FLAGS.batch_size, 1])
        # Creates the dataset
        dataset = tf.data.Dataset.from_tensor_slices(next_story)\
            .map(functools.partial(create_story, **{
                    "use_skip_thoughts": bool(FLAGS.use_skip_thoughts),
                    "vocabLookup": vocabLookup,
                    "vocab": vocab
            }))\
            .batch(FLAGS.batch_size)
        # create the iterator
        iterator = dataset.make_initializable_iterator()  # create the iterator
        next_batch = iterator.get_next()
        for i in range(len(shapes)):
            tf.reshape(next_batch[i], shape=shapes[i])
            next_batch[i].set_shape(shapes[i])
        network_input = {
            "context": next_batch[0],
            "ending1": next_batch[1],
            "ending2": next_batch[2],
            "features1": next_batch[3],
            "features2": next_batch[4]
        }
        # Creating the model
        network = BiDirectional_LSTM(sess, vocab, network_input, FEATURES_SIZE, FLAGS.attention,
                                     FLAGS.attention_size)
        eval_predictions, _, _ = network.build_model()

        # Restore the variables without loading the meta graph!
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_file)

        # Generate batches for one epoch
        def get_batch():
            for batch_num in range(FLAGS.batch_size):
                start_index = batch_num * FLAGS.batch_size
                end_index = min((batch_num + 1) * FLAGS.batch_size, len(x_test))
                story = list(map(lambda sentence: tf.Variable(sentence, dtype=tf.string), x_test[start_index:end_index]))
                # dummy constant if we're predicting since we don't have the labels
                y = tf.Variable((1, 0) if labels is None else labels[i], dtype=tf.int32)
                yield story, y


        batches = get_batch()

        # Collect the predictions here
        predictions = []
        accuracies = []

        accuracy = tf.reduce_mean(
            tf.cast(
                tf.equal(eval_predictions, next_label), dtype=tf.float32
            )
        )

        i = 0
        for story in batches:
            label = labels[i]
            i += 1
            feed_dict = {
                # handle: test_handle,
                network.dropout_rate: 0.0,
                next_story: story,
                next_label: label
            }
            if FLAGS.predict:
                fetches = [eval_predictions]
            else:
                fetches = [accuracy]
            accuracy, preds = sess.run(fetches, feed_dict)
            predictions.append(preds)
            predictions.append(accuracy)

if FLAGS.predict:
    with open(f"group{FLAGS.group_number}_accuracy_{CHECKPOINT_FILE}", 'w') as f:
        for i in range(len(predictions)):
            f.write(str(predictions[i]) + "\n")
else:
    # Only printing out the average accuracy
    avg = np.average(accuracies)
    print(f"Avg accuracy: {avg}")
