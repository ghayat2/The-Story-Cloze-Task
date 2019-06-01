from pathlib import Path

import data_utils as d
from embedding.sentence_embedder import SkipThoughtsEmbedder
from models.bidirectional_lstm import BiDirectional_LSTM
import generate_combined
import losses
import data_utils

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import matplotlib.pyplot as plt

import functools
import sys

# PARAMETERS #
# Data loading parameters
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data used for validation (default: 10%)")
tf.flags.DEFINE_string("data_sentences_path", "./data/processed/train_stories.csv.npy", "Path to sentences file")
tf.flags.DEFINE_string("data_sentences_vocab_path", "./data/processed/train_stories.csv_vocab.npy", "Path to sentences vocab file")
tf.flags.DEFINE_string("data_sentences_eval_path", "./data/processed/eval_stories.csv.npy", "Path to eval sentences file")
tf.flags.DEFINE_string("data_sentences_eval_labels_path", "./data/processed/eval_stories.csv_labels.npy", "Path to eval sentences file")
tf.flags.DEFINE_bool("use_train_set", True, "Whether to use train set, use eval set for training otherwise")

tf.flags.DEFINE_integer("random_seed", 42, "Random seed")

tf.flags.DEFINE_string("unprocessed_training_dataset_path", "./data/train_stories.csv", "Path to the training dataset")
tf.flags.DEFINE_string("skip_thoughts_train_embeddings_path", "./data/processed/train_stories_skip_thoughts.tfrecords",
                       "Path to skip thoughts train sentence embeddings")
tf.flags.DEFINE_string("unprocessed_eval_dataset_path", "./data/eval_stories.csv", "Path to the evaluation dataset")
tf.flags.DEFINE_string("skip_thoughts_eval_embeddings_path", "./data/processed/eval_stories_skip_thoughts.tfrecords",
                       "Path to skip thoughts evaluation sentence embeddings")


# Model parameters
tf.flags.DEFINE_integer("num_sentences_train", 5, "Number of sentences in training set (default: 5)")
tf.flags.DEFINE_integer("sentence_length", 30, "Sentence length (default: 30)")
tf.flags.DEFINE_integer("word_embedding_dimension", 100, "Word embedding dimension size (default: 100)")
tf.flags.DEFINE_integer("num_context_sentences", 4, "Number of context sentences")
tf.flags.DEFINE_integer("classes", 2, "Number of output classes")
tf.flags.DEFINE_integer("num_eval_sentences", 2, "Number of eval sentences")

tf.flags.DEFINE_integer("sentence_embedding_length", 4800, "Length of the sentence embeddings")

tf.flags.DEFINE_integer("ratio_neg_random", 4, "Ratio of negative random endings")
tf.flags.DEFINE_integer("ratio_neg_back", 2, "Ratio of negative back endings")

tf.flags.DEFINE_float("dropout_rate", 0.7, "Dropout rate")


tf.flags.DEFINE_integer("vocab_size", 20000, "Size of the vocabulary")
tf.flags.DEFINE_string("path_embeddings", "data/wordembeddings-dim100.word2vec", "Path to the word2vec embeddings")
tf.flags.DEFINE_bool("use_skip_thoughts", True, "Whether we use skip thoughts for sentences embedding")

tf.flags.DEFINE_string('attention', None, 'Attention type (add ~ Bahdanau, mult ~ Luong, None). Only for Roemmele ''models.')
tf.flags.DEFINE_integer('attention_size', 1000, 'Attention size.')



tf.flags.DEFINE_integer("hidden_layer_size", 100, "Size of hidden layer")
tf.flags.DEFINE_integer("rnn_num", 2, "Number of RNNs")
tf.flags.DEFINE_string("rnn_cell", "LSTM", "Cell type.")
tf.flags.DEFINE_integer("rnn_cell_size", 1000, "RNN cell size")




# Training parameters
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate (default: 0.001)")
tf.flags.DEFINE_integer("repeat_train_dataset", 5000, "Number of times to repeat the dataset")
tf.flags.DEFINE_integer("repeat_eval_dataset", 500, "Number of times to repeat the dataset")
tf.flags.DEFINE_integer("shuffle_buffer_size", 5, "Buffer size for shuffling")
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 3, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_float("grad_clip", 10, "Gradient clip")

tf.flags.DEFINE_string("loss_function", "SOFTMAX", "Loss function to use. Options: SIGMOID, SOFTMAX")
tf.flags.DEFINE_string("optimizer", "ADAM", "Optimizer to use. Options: ADAM, RMS")

tf.flags.DEFINE_string("job_name", None, "Custom job name")

# Tensorflow Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# for running on EULER, adapt this
tf.flags.DEFINE_integer("inter_op_parallelism_threads", 2,
                        "TF nodes that perform blocking operations are enqueued on a pool of "
                        "inter_op_parallelism_threads available in each process (default 0).")
tf.flags.DEFINE_integer("intra_op_parallelism_threads", 2,
                        "The execution of an individual op (for some op types) can be parallelized on a pool of "
                        "intra_op_parallelism_threads (default: 0).")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)

if FLAGS.random_seed is not None:
    np.random.seed(FLAGS.random_seed)
    print(f"Using random seed: {FLAGS.random_seed}")

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value.value))
print("")

if FLAGS.use_skip_thoughts:
    # Saves skip thoughts sentence embeddings if they don't already exist
    skip_thoughts_train_embeddings_file = Path(FLAGS.skip_thoughts_train_embeddings_path)
    skip_thoughts_eval_embeddings_file = Path(FLAGS.skip_thoughts_eval_embeddings_path)
    # If the embeddings haven't been saved in a file yet, compute and save them
    if (not skip_thoughts_train_embeddings_file.is_file()) or (not skip_thoughts_eval_embeddings_file.is_file()):
        embedder = SkipThoughtsEmbedder()
        # Creates the embeddings for the training dataset
        print("Generating skip thoughts training sentence embeddings...")
        if not skip_thoughts_train_embeddings_file.is_file():
            embedder.generate_embedded_training_set(
                FLAGS.unprocessed_training_dataset_path,
                FLAGS.skip_thoughts_train_embeddings_path
            )
        # Creates the embeddings for the testing dataset
        print("Training dataset with skip thoughts embeddings successfully created !")
        print("Generating skip thoughts evaluation sentence embeddings")
        if not skip_thoughts_eval_embeddings_file.is_file():
            embedder.generate_embedded_eval_set(
                FLAGS.unprocessed_eval_dataset_path,
                FLAGS.skip_thoughts_eval_embeddings_path
            )
        print("Evaluation dataset with skip thoughts embeddings successfully created !")

# Load sentences from numpy file, with ids but not embedded
sentences = np.load(FLAGS.data_sentences_path).astype(dtype=np.int32) # [88k, sentence_length (5), vocab_size (30)]
padding_sentences = np.zeros((sentences.shape[0], FLAGS.classes -1, sentences.shape[2]), dtype=np.int32)
sentences = np.concatenate([sentences, padding_sentences], axis=1)

# print(sentences[0])

print(sentences.shape)
# sentences = sentences[:10, :, :]

vocab = np.load(FLAGS.data_sentences_vocab_path, allow_pickle=True)  # vocab contains [symbol: id]
vocabLookup = dict((v,k) for k,v in vocab.item().items()) # flip our vocab dict so we can easy lookup [id: symbol]
vocabLookup[0] = '<pad>'


def eval_shape():
    print("eval sentences shape", np.shape(eval_sentences))


# eval sentences
# six sentences, plus label
eval_sentences = np.load(FLAGS.data_sentences_eval_path).astype(dtype=np.int32)
eval_shape()
if FLAGS.classes > 2:
    padding_sentences = np.zeros((eval_sentences.shape[0], FLAGS.classes -2, eval_sentences.shape[2]), dtype=np.int32)
    eval_sentences = np.concatenate([eval_sentences, padding_sentences], axis=1)
eval_shape()

eval_labels = np.load(FLAGS.data_sentences_eval_labels_path).astype(dtype=np.int32)
eval_labels -= 1

assert FLAGS.classes == 2, "Classes must be 2!"
# print(eval_sentences[0:1], eval_labels[0:1])

# sentence embeddings
# sentences is the vector of size 5 with the vector of size 30 with word numbers, [batch_size, sentence_len, vocab_size]
# [
#   [ 1, 3, 15, 151, .. , ],
#   [ 1, 2 ], ... ]
# ]

# allSentences = sentences.squeeze(axis=1) # make continuous array

# Create sesions
# MODEL AND TRAINING PROCEDURE DEFINITION #
with tf.Graph().as_default():
    pickers = []
    if FLAGS.use_skip_thoughts:
        # Custom pickers for skip thoughts since the standard ones aren't usable
        randomPicker = generate_combined.EmbeddedRandomPicker(SkipThoughtsEmbedder.get_train_tf_dataset().repeat(FLAGS.repeat_train_dataset))
        backPicker = generate_combined.EmbeddedBackPicker()
    else:
        allSentences = tf.constant(np.squeeze(d.endings(sentences), axis=1))
        randomPicker = generate_combined.RandomPicker(allSentences, len(sentences))
        backPicker = generate_combined.BackPicker()

    # Placeholder tensor for input, which is just the sentences with ids
    input_x = tf.placeholder(tf.int32, [None, FLAGS.num_context_sentences + FLAGS.classes, FLAGS.sentence_length]) # [batch_size, sentence_length]
    input_y = tf.placeholder(tf.int32, [None])

    """Iterator stuff"""
    # Initialize model
    handle = tf.placeholder(tf.string, shape=[])

    train_augment_config = {
        'randomPicker': randomPicker,
        'backPicker': backPicker,
        'ratio_random': float(FLAGS.ratio_neg_random),
        'ratio_back': float(FLAGS.ratio_neg_back)
    }
    train_augment_fn = functools.partial(generate_combined.augment_data, **train_augment_config)

    if FLAGS.use_train_set:
        if FLAGS.use_skip_thoughts:
            train_dataset = generate_combined.get_skip_thoughts_data_iterator(
                augment_fn=train_augment_fn,
                batch_size=FLAGS.batch_size,
                repeat_train_dataset=FLAGS.repeat_train_dataset
            )
        else:
            train_dataset = generate_combined.get_data_iterator(
                input_x,
                augment_fn=train_augment_fn,
                batch_size=FLAGS.batch_size,
                repeat_train_dataset=FLAGS.repeat_train_dataset
            )
    else:
        if FLAGS.use_skip_thoughts:
            train_dataset = generate_combined.get_skip_thoughts_eval_iterator(
                labels=input_y,
                batch_size=FLAGS.batch_size,
                repeat_eval_dataset=FLAGS.repeat_train_dataset
            )
        else:
            train_dataset = generate_combined.get_eval_iterator(
                input_x,
                input_y,
                batch_size=FLAGS.batch_size,
                repeat_eval_dataset=FLAGS.repeat_train_dataset
            )

    if FLAGS.use_skip_thoughts:
        test_dataset = generate_combined.get_skip_thoughts_eval_iterator(
            input_y, batch_size=FLAGS.batch_size, repeat_eval_dataset=FLAGS.repeat_eval_dataset
        )
    else:
        test_dataset = generate_combined.get_eval_iterator(input_x,
                                                         input_y,
                                                 batch_size=FLAGS.batch_size,
                                                 repeat_eval_dataset=FLAGS.repeat_eval_dataset)
    
    print("Test output types", test_dataset.output_types)

    # Iterators on the training and validation dataset
    train_iterator = train_dataset.make_initializable_iterator()
    test_iterator = test_dataset.make_initializable_iterator()

    iter = tf.data.Iterator.from_string_handle(
        handle, train_dataset.output_types, train_dataset.output_shapes)

    next_batch_context_x, next_batch_endings_y = iter.get_next()
    print("--------new_batch_x------------", next_batch_context_x[0])
    print("--------new_batch_ending_y------------", next_batch_endings_y[0])
    num_sentences_total = FLAGS.num_context_sentences + FLAGS.classes
    sentence_length = FLAGS.sentence_embedding_length if FLAGS.use_skip_thoughts else FLAGS.sentence_length
    next_batch_context_x.set_shape([FLAGS.batch_size, num_sentences_total, sentence_length])

    train_init_op = iter.make_initializer(train_dataset, name='train_dataset')
    test_init_op = iter.make_initializer(test_dataset, name='test_dataset')

    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement,
        inter_op_parallelism_threads=FLAGS.inter_op_parallelism_threads,
        intra_op_parallelism_threads=FLAGS.intra_op_parallelism_threads)
    sess = tf.Session(config=session_conf)
    with sess.as_default():

        if FLAGS.random_seed is not None:
            tf.set_random_seed(FLAGS.random_seed)

        # Build execution graph
        network = BiDirectional_LSTM(sess, vocab, next_batch_context_x, FLAGS.attention, FLAGS.attention_size)

        # train_logits: [batch_size]
        # eval_predictions: [batch_size] (index of prediction
        eval_predictions, endings, train_logits = network.build_model()

        # Compare with next_batch_endings_y
        if FLAGS.loss_function == "SIGMOID":
            loss = losses.sigmoid(next_batch_endings_y, train_logits)
        elif FLAGS.loss_function == "SOFTMAX":
            loss = losses.sparse_softmax(next_batch_endings_y, endings)
        else:
            raise RuntimeError(f"Loss function {FLAGS.loss_function} not supported.")

        # correct_position = tf.cast(tf.argmax(next_batch_endings_y, axis=1), dtype=tf.int32)
        accuracy = tf.reduce_mean(
            tf.cast(
                tf.equal(eval_predictions, next_batch_endings_y), dtype=tf.float32
            )
        )

        """Initialize iterators"""
        train_handle = sess.run(train_iterator.string_handle())
        test_handle = sess.run(test_iterator.string_handle())

        if FLAGS.use_train_set:
            sess.run(train_iterator.initializer, feed_dict={input_x: sentences})
        else:
            sess.run(train_iterator.initializer, feed_dict={input_x: eval_sentences, input_y: eval_labels})
            
        sess.run(test_iterator.initializer, feed_dict={input_x: eval_sentences, input_y: eval_labels})

        # Iterator test for debugging to compare inputs
        # iterTestSentences, iterTestLabels = sess.run([next_batch_context_x, next_batch_endings_y], {handle: train_handle})
        # print("Story 1", data_utils.makeSymbolStory(iterTestSentences[0], vocabLookup))
        # print("Label 1", iterTestLabels[0])


        # Define training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        # TODO: Define an optimizer, e.g. AdamOptimizer
        if FLAGS.optimizer == "ADAM":
            optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        elif FLAGS.optimizer == "RMS":
            optimizer = tf.train.RMSPropOptimizer(learning_rate=FLAGS.learning_rate)
        else:
            raise RuntimeError(f"Optimizer {FLAGS.optimizer} not supported!")
        # TODO: Define a training operation, including the global_step
        # train_op = optimizer.minimize(loss, global_step=global_step)

        gradients = optimizer.compute_gradients(loss)
        clipped_gradients = [(tf.clip_by_norm(gradient, FLAGS.grad_clip), var) for gradient, var in gradients]
        train_op = optimizer.apply_gradients(clipped_gradients, global_step=global_step)

        # Output directory for models and summaries
        if FLAGS.job_name is not None:
            job_name = FLAGS.job_name
        else:
            job_name = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", job_name))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", loss)
        acc_summary = tf.summary.scalar("accuracy", accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory (TensorFlow assumes this directory already exists so we need to create it)
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Plot dir
        plot_dir = os.path.join(out_dir, "plots")
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        # Initialize all variables
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)
        sess.graph.finalize()

        # Print variables [DEBUG]
        tvars = tf.trainable_variables()
        tvars_vals = sess.run(tvars)

        for var, val in zip(tvars, tvars_vals):
            print(var.name)  # Prints the name of the variable alongside its value.

        # Define training and dev steps (batch)
        def train_step(loss, accuracy, current_step):
            """
            A single training step
            """
            feed_dict = {
                handle: train_handle,
                network.dropout_rate: FLAGS.dropout_rate
            }
            fetches = [train_op, global_step, train_summary_op, loss, accuracy, next_batch_endings_y, eval_predictions, next_batch_context_x, network.train_predictions]
            _, step, summaries, loss, accuracy, by, eval, context, sanity = sess.run(fetches, feed_dict)
            print(f"{sanity}")
            print("shape context", context.shape)
            # print(f"{tl}")
            if not FLAGS.use_skip_thoughts:
                print("--------next_batch_x -----------", d.makeSymbolStory(context[0], vocabLookup))
            print(f"labels {by}")
            print(f"predictions {eval}")
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(
                time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(loss, accuracy, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                handle: test_handle,
                network.dropout_rate: 0.0
            }
            fetches = [global_step, dev_summary_op, loss, accuracy, next_batch_endings_y, eval_predictions, next_batch_context_x]
            step, summaries, loss, accuracy, by, eval, context = sess.run(fetches, feed_dict)
            time_str = datetime.datetime.now().isoformat()
            if not FLAGS.use_skip_thoughts:
                print("--------next_batch_x -----------", d.makeSymbolStory(context[0], vocabLookup))
            print(f"labels {by}")
            print(f"predictions {eval}")
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        """ Training loop - default option is that the model trains until an OutOfRange exception """
        current_step = 0
        while True:
            try:
                train_step(loss, accuracy, current_step)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(loss, accuracy, writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
            except tf.errors.OutOfRangeError:
                print("Iterator of range! Terminating")
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
                break
