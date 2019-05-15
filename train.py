import data_utils as d
from models.bidirectional_lstm import BiDirectional_LSTM
import augment
from genreate_random import RandomPicker

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
tf.flags.DEFINE_string("data_sentences_path", "PATH", "Path to sentences file")


# Model parameters
tf.flags.DEFINE_integer("num_sentences_train", 5, "Number of sentences in training set (default: 5)")
tf.flags.DEFINE_integer("sentence_length", 30, "Sentence length (default: 30)")
tf.flags.DEFINE_integer("word_embedding_dimension", 100, "Word embedding dimension size (default: 100)")
tf.flags.DEFINE_integer("num_context_sentences", 4, "Number of context sentences")
tf.flags.DEFINE_integer("classes", 2, "Number of output classes")


tf.flags.DEFINE_integer("hidden_layer_size", 100, "Size of hidden layer")
tf.flags.DEFINE_integer("rnn_num", 2, "Number of RNNs")
tf.flags.DEFINE_string("rnn_cell", "LSTM", "Cell type.")
tf.flags.DEFINE_integer("rnn_cell_size", 2, "RNN cell size")


# Augmenting parameters


# Training parameters
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate (default: 0.001)")
tf.flags.DEFINE_integer("repeat_train_dataset", 5000, "Number of times to repeat the dataset")
tf.flags.DEFINE_integer("shuffle_buffer_size", 5, "Buffer size for shuffling")
tf.flags.DEFINE_integer("batch_size", 4, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

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

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value.value))
print("")

# Load sentences from numpy file, with ids but not embedded
sentences = np.load(FLAGS.data_sentences_path) # [88k, sentence_length (5), vocab_size (30)]

# sentence embeddings
# sentences is the vector of size 5 with the vector of size 30 with word numbers, [batch_size, sentence_len, vocab_size]
# [
#   [ 1, 3, 15, 151, .. , ],
#   [ 1, 2 ], ... ]
# ]

allSentences = sentences.squeeze(axis=1) # make continuous array
randomPicker = RandomPicker(d.endings(allSentences))


# Create sesions
# MODEL AND TRAINING PROCEDURE DEFINITION #
with tf.Graph().as_default():

    # Placeholder tensor for input, which is just the sentences with ids
    input_x = tf.placeholder(tf.int32, [None, FLAGS.sentence_length]) # [batch_size, sentence_length]

    """Iterator stuff"""
    # Initialize model
    handle = tf.placeholder(tf.string, shape=[])

    train_augment_config = {
        'randomPicker': randomPicker,
    }
    train_augment_fn = functools.partial(augment.augment_data, **train_augment_config)

    validation_augment_config = {
        'randomPicker': randomPicker,
    }
    validation_augment_fn = functools.partial(augment.augment_data, **validation_augment_config)

    train_dataset = augment.get_data_iterator(input_x,
                                                 augment_fn=train_augment_fn,
                                                 batch_size=FLAGS.batch_size,
                                                 repeat_train_dataset=FLAGS.repeat_train_dataset) \
        .shuffle(buffer_size=FLAGS.shuffle_buffer_size)

    test_dataset = augment.get_data_iterator(input_x,
                                             augment_fn=validation_augment_fn,
                                             batch_size=FLAGS.batch_size,
                                             repeat_train_dataset=FLAGS.repeat_train_dataset)

    # Iterators on the training and validation dataset
    train_iterator = train_dataset.make_initializable_iterator()
    test_iterator = test_dataset.make_initializable_iterator()

    iter = tf.data.Iterator.from_string_handle(
        handle, train_dataset.output_types, train_dataset.output_shapes)

    next_batch_context_x, next_batch_endings_y = iter.get_next()

    next_batch_context_x.set_shape([FLAGS.batch_size, FLAGS.sentence_length, FLAGS.word_embedding_dimension])

    train_init_op = iter.make_initializer(train_dataset, name='train_dataset')
    test_init_op = iter.make_initializer(test_dataset, name='test_dataset')

    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement,
        inter_op_parallelism_threads=FLAGS.inter_op_parallelism_threads,
        intra_op_parallelism_threads=FLAGS.intra_op_parallelism_threads)
    sess = tf.Session(config=session_conf)
    with sess.as_default():

        # Build execution graph
        network = BiDirectional_LSTM(next_batch_context_x)

        output = network.build_model()

        # Compare with next_batch_endings_y
        loss = tf.reduce_mean(
            # loss something
            tf.losses.sparse_softmax_cross_entropy(next_batch_endings_y, output)
        )

        predictions = tf.argmax(
            tf.sparse_softmax(output), axis=1
        )
        accuracy = tf.equal(output, next_batch_endings_y)

        """Initialize iterators"""
        train_handle = sess.run(train_iterator.string_handle())
        test_handle = sess.run(test_iterator.string_handle())

        # Define training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        # TODO: Define an optimizer, e.g. AdamOptimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        # TODO: Define a training operation, including the global_step
        train_op = optimizer.minimize(loss, global_step=global_step)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
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

        # Define training and dev steps (batch)
        def train_step(loss, accuracy, current_step):
            """
            A single training step
            """
            feed_dict = {handle: train_handle}
            fetches = [train_op, global_step, train_summary_op, loss, accuracy]
            _, step, summaries, loss, accuracy = sess.run(fetches, feed_dict)

            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(
                time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(loss, accuracy, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                handle: test_handle
            }
            fetches = [global_step, dev_summary_op, loss, accuracy]
            step, summaries, loss, accuracy = sess.run(fetches, feed_dict)
            time_str = datetime.datetime.now().isoformat()
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
