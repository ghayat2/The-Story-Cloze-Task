import tensorflow as tf
from typing import Tuple
from data_pipeline import data_utils

FLAGS = tf.flags.FLAGS


class BiDirectional_LSTM:
    ACTIVATION_NODES = 1

    def __init__(self, session, vocab, context, attention: str = None, attention_size: int = 1):
        print("Super awesome model")
        self.input = context
        self.session = session
        self.vocab = vocab
        self.attention = attention
        self.attention_size = attention_size

        self.dropout_rate = tf.placeholder_with_default(0.0, shape=())

    def _word_embeddings(self):
        # embed the words here
        self.embedding_matrix = tf.get_variable("embedding_matrix",
                                                initializer=tf.random_uniform(
                                                    [FLAGS.vocab_size, FLAGS.word_embedding_dimension],
                                                    -1.0, 1.0),
                                                dtype=tf.float32,
                                                trainable=False)

        data_utils.load_embedding(self.session, self.vocab, self.embedding_matrix, FLAGS.path_embeddings,
                                  FLAGS.word_embedding_dimension,
                                  FLAGS.vocab_size)

        embedded_context = tf.nn.embedding_lookup(self.embedding_matrix, self.input["context"])  # DIM [batch_size, sentence_len, embedding_dim]
        embedded_ending1 = tf.nn.embedding_lookup(self.embedding_matrix, self.input["ending1"])
        embedded_ending2 = tf.nn.embedding_lookup(self.embedding_matrix, self.input["ending2"])

        return embedded_context, embedded_ending1, embedded_ending2

    def _sentence_states(self) -> tf.Tensor:
        if FLAGS.use_skip_thoughts:
            return self.input

        with tf.name_scope("word_embeddings"):
            (context_embedding, ending1_embedding, ending2_embedding) = self._word_embeddings()
            alternative_1 = tf.concat(values=[context_embedding, ending1_embedding], axis=2)
            alternative_2 = tf.concat(values=[context_embedding, ending2_embedding], axis=2)
            per_sentence_states = []
            with tf.variable_scope("word_rnn"):
                # per_sentence_states = self._word_rnn(sentence_word_embeddings)
                per_sentence_states.append(tf.reduce_mean(alternative_1, axis=2))
                per_sentence_states.append(tf.reduce_mean(alternative_2, axis=2))
            return tf.stack(values=per_sentence_states, name="per_sentence_states")

    def _create_cell(self, rnn_cell_dim, name=None) -> tf.nn.rnn_cell.RNNCell:
        if FLAGS.rnn_cell == "LSTM":
            return tf.nn.rnn_cell.LSTMCell(rnn_cell_dim, name=name)
        elif FLAGS.rnn_cell == "GRU":
            return tf.nn.rnn_cell.GRUCell(rnn_cell_dim, name=name)
        elif FLAGS.rnn_cell == "VAN":
            return tf.nn.rnn_cell.BasicRNNCell(rnn_cell_dim, name=name)
        else:
            raise ValueError(f"Unknown rnn_cell {FLAGS.rnn_cell}.")

    def _sentence_rnn(self, per_sentence_states: tf.Tensor) -> tf.Tensor:
        assert len(per_sentence_states.get_shape()) == 3
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

        if self.attention is not None:  # with attention
            with tf.variable_scope("attention"):
                context = self._add_attention(outputs_tensor, cell_output=state, prefix="attention")
                print("context", context.get_shape())
            sentence_states.append(context)

        res = tf.concat(sentence_states, axis=1)
        print("sentence_states", res.get_shape())
        return res

    def _output_fc(self, state: tf.Tensor) -> tf.Tensor:
        output = tf.layers.dense(state, self.ACTIVATION_NODES, activation=None, name="output")
        print("output", output.get_shape())
        return output

    def _dropout_layer(self, state: tf.Tensor) -> tf.Tensor:
        return tf.nn.dropout(state, rate=self.dropout_rate)

    def _integrate_features(self, state: tf.Tensor, features) -> tf.Tensor:
        state = tf.concat(values=(state, tf.squeeze(features)), axis=1)
        return tf.layers.dense(state, FLAGS.feature_integration_layer_output_size,
                               activation=tf.nn.relu, name="features_dense")

    def build_model(self):

        with tf.name_scope("split_endings"):
            per_sentence_states = self._sentence_states()
            alternative1 = tf.concat(values=(per_sentence_states["context"], per_sentence_states["ending1"]), axis=2)
            alternative2 = tf.concat(values=(per_sentence_states["context"], per_sentence_states["ending2"]), axis=2)

        with tf.variable_scope("ending") as ending_scope:
            with tf.name_scope("sentence_rnn"):
                print(f"Story {1}", alternative1)
                res = self._sentence_rnn(alternative1)
                res = self._dropout_layer(res)

                print("RES2", res)

            with tf.name_scope("features_dense"):
                res = self._integrate_features(res, per_sentence_states["features1"])

            with tf.name_scope("fc"):
                ending_outputs1 = self._output_fc(res)

        with tf.variable_scope(ending_scope, reuse=True):
            with tf.name_scope("sentence_rnn"):
                print(f"Story {2}", alternative2)
                res = self._sentence_rnn(alternative2)
                res = self._dropout_layer(res)

                print("RES2", res)

            with tf.name_scope("features_dense"):
                res = self._integrate_features(res, per_sentence_states["features2"])

            with tf.name_scope("fc"):
                ending_outputs2 = self._output_fc(res)

        ending_outputs = tf.stack([ending_outputs1, ending_outputs2], axis=1)

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

    def _add_attention(self, outputs: tf.Tensor, cell_output: tf.Tensor, prefix="") -> tf.Tensor:
        attention_mechanism = self._create_attention(outputs)
        context, alignments, next_attention_state = self._compute_attention(
            attention_mechanism,
            cell_output,
            attention_state=attention_mechanism.initial_state(FLAGS.batch_size, dtype=tf.float32))
        #        with tf.name_scope("summaries"):
        #            self._attention_summaries(alignments, prefix=prefix)
        return context

    def _compute_attention(self, mechanism: tf.contrib.seq2seq.AttentionMechanism, cell_output: tf.Tensor,
                           attention_state: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/contrib/seq2seq
        # /python/ops/attention_wrapper.py
        # Alignments shape is [batch_size, 1, memory_time].
        alignments, next_attention_state = mechanism(cell_output, attention_state)
        # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
        expanded_alignments = tf.expand_dims(alignments, axis=1)
        # Context is the inner product of alignments and values along the memory time dimension.
        # The attention_mechanism.values shape is [batch_size, memory_time, memory_size].
        # The matmul is over memory_time, so the output shape is [batch_size, 1, memory_size].
        context = tf.matmul(expanded_alignments, mechanism.values)
        # We then squeeze out the singleton dim.
        context = tf.squeeze(context, axis=[1])
        return context, alignments, next_attention_state

    def _create_attention(self, encoder_outputs: tf.Tensor) -> tf.contrib.seq2seq.AttentionMechanism:
        if self.attention == "add":
            attention = tf.contrib.seq2seq.BahdanauAttention
        elif self.attention == "mult":
            attention = tf.contrib.seq2seq.LuongAttention
        else:
            raise ValueError(f"Unknown attention {self.attention}. Possible values: add, mult.")
        return attention(num_units=self.attention_size, memory=encoder_outputs, dtype=tf.float32)

    def _attention_summaries(self, alignments: tf.Tensor, prefix="") -> None:
        with self.summary_writer.as_default():
            with tf.contrib.summary.record_summaries_every_n_global_steps(50):
                img = self._attention_images_summary(alignments, prefix=f"{prefix}/train")
                self.summaries["train"].append(img)

    def _attention_images_summary(self, alignments: tf.Tensor, prefix: str = "") -> tf.Operation:
        # https://github.com/tensorflow/nmt/blob/master/nmt/attention_model.py
        """
        Create attention image and attention summary.
        """
        # Reshape to (batch, tgt_seq_len, src_seq_len, 1)
        print("alignments", alignments.get_shape())
        attention_images = tf.expand_dims(tf.expand_dims(alignments, axis=-1), axis=-1)
        attention_images = tf.transpose(attention_images, perm=(0, 2, 1, 3))  # make img horizontal
        # Scale to range [0, 255]
        attention_images *= 255
        attention_summary = tf.contrib.summary.image(f"{prefix}/attention_images", attention_images)
        return attention_summary
