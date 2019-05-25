import tensorflow as tf
from typing import Tuple
import data_utils
from embedding.sentence_embedder import SkipThoughtsEmbedder

FLAGS = tf.flags.FLAGS

class BiDirectional_LSTM:

    ACTIVATION_NODES = 1

    def __init__(self, session, vocab, context):
        print("Super awesome model")
        self.input = context
        self.session = session
        self.vocab = vocab

    def _word_embeddings(self):
        # embed the words here
        self.embedding_matrix = tf.get_variable("embedding_matrix",
                                                initializer=tf.random_uniform([FLAGS.vocab_size, FLAGS.word_embedding_dimension],
                                                                              -1.0, 1.0),
                                                dtype=tf.float32,
                                                trainable=True)

        data_utils.load_embedding(self.session, self.vocab, self.embedding_matrix, FLAGS.path_embeddings, FLAGS.word_embedding_dimension,
                                  FLAGS.vocab_size)

        embedded_words = tf.nn.embedding_lookup(self.embedding_matrix,
                                                     self.input)  # DIM [batch_size, sentence_len, embedding_dim]

        # concatenated_words = tf.reshape(embedded_words, [-1,
        #                                                 FLAGS.num_context_sentences + FLAGS.classes,
        #                                                 FLAGS.sentence_length * FLAGS.word_embedding_dimension
        #                                                 ])
        # concatenated_words = tf.concat(embedded_words, axis=3)

        # concatenated_words = tf.reduce_mean(embedded_words, axis=2) # Average, like roemmele
        # print("concatenated_words", concatenated_words)
        return embedded_words

    def _sentence_states(self) -> tf.Tensor:
        with tf.name_scope("word_embeddings"):
            sentence_word_embeddings = self._word_embeddings()

            with tf.variable_scope("word_rnn"):
                # per_sentence_states = self._word_rnn(sentence_word_embeddings)
                per_sentence_states = tf.reduce_sum(sentence_word_embeddings, axis=2)

            return per_sentence_states

    def _create_cell(self, rnn_cell_dim, name=None) -> tf.nn.rnn_cell.RNNCell:
        reuse = tf.AUTO_REUSE
        if FLAGS.rnn_cell == "LSTM":
            return tf.nn.rnn_cell.LSTMCell(rnn_cell_dim, name=name, reuse=reuse)
        elif FLAGS.rnn_cell == "GRU":
            return tf.nn.rnn_cell.GRUCell(rnn_cell_dim, name=name, reuse=reuse)
        elif FLAGS.rnn_cell == "VAN":
            return tf.nn.rnn_cell.BasicRNNCell(rnn_cell_dim, name=name, reuse=reuse)
        else:
            raise ValueError(f"Unknown rnn_cell {FLAGS.rnn_cell}.")

    def _sentence_rnn(self, per_sentence_states: tf.Tensor) -> tf.Tensor:
        assert len(per_sentence_states.get_shape()) == 3
        assert per_sentence_states.get_shape()[1] == FLAGS.num_context_sentences + 1
        # Create the cell
        rnn_cell_sentences = self._create_cell(FLAGS.rnn_cell_size, name='sentence_cell')

        inputs = tf.unstack(per_sentence_states, axis=1)
        outputs, state = tf.nn.static_rnn(cell=rnn_cell_sentences, inputs=inputs, dtype=tf.float32)
        if FLAGS.rnn_cell == "LSTM":
            state = state[0]  # c_state

        print("outputs[0]", outputs[0].get_shape())
        outputs_lst = [tf.expand_dims(x, axis=1) for x in outputs]
        outputs_tensor = tf.concat(outputs_lst, axis=1)
        print("outputs_tensor", outputs_tensor.get_shape())

        sentence_states = [state]

        res = tf.concat(sentence_states, axis=1)
        print("sentence_states", res.get_shape())
        return res

    def _output_fc(self, state: tf.Tensor) -> tf.Tensor:
        output = tf.layers.dense(state, self.ACTIVATION_NODES, activation=None, name="output")
        print("output", output.get_shape())
        return output

    def _dropout_layer(self, state: tf.Tensor) -> tf.Tensor:
        if FLAGS.dropout_rate > 0:
            return tf.nn.dropout(state, rate=FLAGS.dropout_rate)
        return state
            

    def build_model(self):

#        with tf.name_scope("split_endings"):
#            per_sentence_states = self._sentence_states()
#            sentence_states = per_sentence_states[:, :FLAGS.num_context_sentences, :]
#            print("sentence_states", sentence_states.get_shape())
#            ending1_states = per_sentence_states[:, FLAGS.num_context_sentences + 0, :]
#            ending1_states = tf.expand_dims(ending1_states, axis=1)
#            print("ending1_states", ending1_states.get_shape())
#            ending2_states = per_sentence_states[:, FLAGS.num_context_sentences + 1, :]
#            ending2_states = tf.expand_dims(ending2_states, axis=1)
#            print("ending2_states", ending2_states.get_shape())
#            ending1_states = tf.concat([sentence_states, ending1_states], axis=1)
#            ending2_states = tf.concat([sentence_states, ending2_states], axis=1)
            
        with tf.name_scope("split_endings"):
            per_sentence_states = self.input if FLAGS.use_skip_thoughts else self._sentence_states()
            sentence_states = per_sentence_states[:, :FLAGS.num_context_sentences, :]

            print("sentence_states", sentence_states.get_shape())
            ending_states = per_sentence_states[:, FLAGS.num_context_sentences:, :]
            print("-------------ENDING_STATES-----------", ending_states)
            # sentence_states = tf.reshape(sentence_states, [FLAGS.num_context_sentences,FLAGS.batch_size,FLAGS.word_embedding_dimension])
            # ending_states = tf.reshape(ending_states, [FLAGS.classes,FLAGS.batch_size,FLAGS.word_embedding_dimension])
            # stories = tf.map_fn(lambda x: tf.concat([sentence_states, tf.expand_dims(x, axis = 0)], axis = 0), ending_states)
            # stories = tf.reshape(stories, [FLAGS.classes, FLAGS.batch_size, FLAGS.num_context_sentences + 1, FLAGS.word_embedding_dimension])
            # print("-------------ENDING_STATES-----------", stories)




        with tf.variable_scope("ending", reuse=tf.AUTO_REUSE) as ending_scope:
            with tf.name_scope("sentence_rnn"):
                per_story_states = []
                for i in range(FLAGS.classes):
                    story = tf.concat([sentence_states, ending_states[:, i:i+1, :]], axis=1)
                    print(f"Story {i}", story)
                    res = self._sentence_rnn(story)

                    # dropout
                    res = self._dropout_layer(res)

                    print("RES", res)
                    # per_story_states = tf.map_fn(self._sentence_rnn, stories)
                    with tf.name_scope("fc"):
                        ending_outputs = self._output_fc(res)
                        per_story_states.append(ending_outputs)

                ending_outputs = tf.stack(per_story_states, axis=1)

        print("-------------ENDING_OUTPUTS-----------", ending_outputs)

        with tf.name_scope("eval_predictions"):
            endings = tf.squeeze(ending_outputs, axis=2)
            self.sanity_endings = endings
            print("-------------ENDINGS-----------", endings)
            eval_predictions = tf.to_int32(tf.argmax(endings, axis=1))
            print("-------------EVAL_PREDICTION-----------", eval_predictions)

        with tf.name_scope("train_predictions"):
            self.train_logits = tf.squeeze(ending_outputs[:, 0], axis=[1])
            self.train_probs = tf.sigmoid(self.train_logits)
            self.train_predictions = tf.to_int32(tf.round(self.train_probs))

        return eval_predictions, endings, self.train_logits
