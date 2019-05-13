import tensorflow as tf


FLAGS = tf.flags.FLAGS

class BiDirectional_LSTM:

    def __init__(self, context):
        print("Super awesome model")

        with tf.variable_scope("embedding"):
            # Use embedding stuff here

        with tf.variable_scope("discriminator"):
            rnn_cell = tf.nn.rnn_cell.BasicLSTMCell
            outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                cells_fw=[rnn_cell(FLAGS.hidden_layer_size) for _ in range(FLAGS.rnn_num)],
                cells_bw=[rnn_cell(FLAGS.hidden_layer_size) for _ in range(FLAGS.rnn_num)],
                inputs=context, dtype=tf.float32)
            summed_output = tf.reduce_sum(outputs, 1)
            # Gives an output between 0 and 1
            # representing the probability of the last sentence being correct
            self.output = tf.contrib.layers.fully_connected(summed_output, 1, activation_fn=tf.sigmoid)