import tensorflow as tf
from typing import Tuple


FLAGS = tf.flags.FLAGS

class BiDirectional_LSTM:

    def __init__(self, context):
        print("Super awesome model")
        self.input = context

    def _word_embeddings(self):
        # embed the words here
        return self.input


    def _create_cell(self, rnn_cell_dim, name=None) -> tf.nn.rnn_cell.RNNCell:
        if FLAGS.rnn_cell == "LSTM":
            return tf.nn.rnn_cell.LSTMCell(rnn_cell_dim, name=name)
        elif FLAGS.rnn_cell == "GRU":
            return tf.nn.rnn_cell.GRUCell(rnn_cell_dim, name=name)
        elif FLAGS.rnn_cell == "VAN":
            return tf.nn.rnn_cell.BasicRNNCell(rnn_cell_dim, name=name)
        else:
            raise ValueError(f"Unknown rnn_cell {FLAGS.rnn_cell}.")

    def _story_embeddings(self, sentence_wordword_states: tf.Tensor) -> Tuple[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        effective_cell_dim = FLAGS.rnn_cell_size * 2
        if FLAGS.rnn_cell == "LSTM":
            effective_cell_dim *= 2

            # TODO: look into this when we run
        states_unflat = tf.reshape(sentence_wordword_states,
                                   (-1, FLAGS.num_context_sentences + FLAGS.classes, effective_cell_dim))
        print("states_unflat", states_unflat.get_shape())
        states_sentences = states_unflat[:, :FLAGS.num_context_sentences, :]
        states_endings = states_unflat[:, -FLAGS.classes:, :]
        print("states separate", states_sentences.get_shape(), states_endings.get_shape())

        sentence_state = tf.layers.flatten(states_sentences[:, -1, :])  # last sentence
        print("sentence_state", sentence_state.get_shape())
        ending1_state = tf.layers.flatten(states_endings[:, 0, :])
        print("ending1_state", ending1_state.get_shape())
        ending2_state = tf.layers.flatten(states_endings[:, 1, :])
        print("ending2_state", ending1_state.get_shape())
        return sentence_state, (ending1_state, ending2_state)

    def _sentence_rnn(self, inputs: tf.Tensor) -> tf.Tensor:

        rnn_cell_words = self._create_cell(FLAGS.rnn_cell_size)
        _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_bw=rnn_cell_words,
                cell_fw=rnn_cell_words,
                inputs=inputs,
                sequence_length=FLAGS.num_context_sentences + FLAGS.classes,
                dtype=tf.float32)
        if FLAGS.rnn_cell == "LSTM":
            state_fw = tf.concat(state_fw, axis=-1)
            state_bw = tf.concat(state_bw, axis=-1)
        sentence_wordword_states = tf.concat([state_fw, state_bw], axis=1)
        return sentence_wordword_states

    def _fully_connected_layer(self, sentence_state: tf.Tensor, ending_states: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        with tf.variable_scope("sentences_fc"):
            sentences_fc = tf.layers.dense(sentence_state, 1024, activation=tf.nn.leaky_relu)
            sentences_fc = tf.layers.dense(sentences_fc, 512, activation=tf.nn.leaky_relu)
            print("sentences_fc", sentences_fc.get_shape())
        with tf.variable_scope("endings_fc"):
            ending_fc = tf.layers.dense(ending_states[0], 1024, activation=tf.nn.leaky_relu, name="fc1")
            ending1_output = tf.layers.dense(ending_fc, 512, activation=tf.nn.leaky_relu, name="fc2")
            print("ending1_output", ending1_output.get_shape())
        with tf.variable_scope("endings_fc", reuse=tf.AUTO_REUSE):
            ending_fc = tf.layers.dense(ending_states[1], 1024, activation=tf.nn.leaky_relu, name="fc1")
            ending2_output = tf.layers.dense(ending_fc, 512, activation=tf.nn.leaky_relu, name="fc2")
            print("ending2_output", ending2_output.get_shape())
        with tf.variable_scope("common_fc"):
            flatten = tf.concat([sentences_fc, ending1_output, ending2_output], axis=1)
            fc = tf.layers.dense(flatten, 1024, activation=tf.nn.leaky_relu, name="fc1")
            fc = tf.layers.dense(fc, 512, activation=tf.nn.leaky_relu, name="fc2")
            output = tf.layers.dense(fc, FLAGS.classes, activation=None, name="output")
            print("output", output.get_shape())
        return output

    def build_model(self):
        with tf.name_scope("word_embeddings"):
            sentence_word_embeddings = self._word_embeddings()

        with tf.name_scope("sentence_embeddings"):
            sentence_word_states = self._sentence_rnn(sentence_word_embeddings)

        with tf.name_scope("story_embeddings"):
            states_sentences, states_endings = self._story_embeddings(sentence_word_states)

        with tf.name_scope("fc"):
            output_layer = self._fully_connected_layer(states_sentences, states_endings)

        return output_layer