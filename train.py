import data_utils
from models.bidirectional_lstm import BiDirectional_LSTM

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

# Model parameters

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


# Load data into iterator

# Create sesions
# MODEL AND TRAINING PROCEDURE DEFINITION #
with tf.Graph().as_default():

    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement,
        inter_op_parallelism_threads=FLAGS.inter_op_parallelism_threads,
        intra_op_parallelism_threads=FLAGS.intra_op_parallelism_threads)
    sess = tf.Session(config=session_conf)
    with sess.as_default():

        # Build execution graph
        network = BiDirectional_LSTM

        loss = tf.reduce_mean(
            # loss something
        )

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
        f1_summary = tf.summary.scalar("f1", f1)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, f1_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary, f1_summary])
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

