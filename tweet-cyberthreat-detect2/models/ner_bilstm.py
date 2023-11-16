import tensorflow as tf
import numpy as np
from .tf_model import TFModel
tf.compat.v1.disable_eager_execution()
import tensorflow_addons as tfa
tf.compat.v1.disable_v2_behavior()
# sess = tf.compat.v1.Session()
# tf.compat.v1.initialize_all_variables().run(session=sess)


class RNN(TFModel):
    def input_layer(self):
        '''
            Data and Hyperparameters
        '''
        with tf.compat.v1.variable_scope("input_layer"):
            # Tensor containing word ids
            # shape = (batch size, max length of sentence in batch)
            self.word_ids = tf.compat.v1.placeholder(tf.int32, shape=[None, None],
                                           name="word_ids")
            # Tensor containing the real length of each sentence
            # shape = (batch size)
            self.sentence_lengths = tf.compat.v1.placeholder(tf.int32, shape=[None],
                                                   name="sentence_lengths")
            # Tensor containing char ids
            # shape = (batch size, max length of sentence, max length of word)
            self.char_ids = tf.compat.v1.placeholder(tf.int32, shape=[None, None, None],
                                           name="char_ids")

            # shape = (batch_size, max_length of sentence)
            self.word_lengths = tf.compat.v1.placeholder(tf.int32, shape=[None, None],
                                               name="word_lengths")
            # Tensor containing the real length of each word
            # shape = (batch size, max length of sentence in batch)
            self.labels = tf.compat.v1.placeholder(tf.int32, shape=[None, None],
                                         name="labels")

            # Dropout tensors
            self.char_drop_input = tf.compat.v1.placeholder_with_default(
                input=1.0, shape=(), name="char_drop_input")
            self.char_drop_state = tf.compat.v1.placeholder_with_default(
                input=1.0, shape=(), name="char_drop_state")
            self.char_drop_output = tf.compat.v1.placeholder_with_default(
                input=1.0, shape=(), name="char_drop_output")
            self.word_drop_input = tf.compat.v1.placeholder_with_default(
                input=1.0, shape=(), name="word_drop_input")
            self.word_drop_state = tf.compat.v1.placeholder_with_default(
                input=1.0, shape=(), name="word_drop_state")
            self.word_drop_output = tf.compat.v1.placeholder_with_default(
                input=1.0, shape=(), name="word_drop_output")

            # Training variables
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            # Using a decaying learning rate
            self.lr = tf.compat.v1.train.exponential_decay(
                learning_rate=self.config.learning["rate"],
                global_step=self.global_step,
                decay_steps=self.config.learning["decay_steps"],
                decay_rate=self.config.learning["decay"],
                staircase=self.config.learning["staircase"])
            # Create the optimizer/trainer
            # I initialize it here for multi-gpu training
            self.optimizer = tf.compat.v1.train.AdamOptimizer(self.lr)

    def embedding_layer(self):
        '''
            Embedding matrices
        '''
        with tf.compat.v1.variable_scope("embedding_layer"):
            if self.config.pretrained is None:
                # Using randomly initialized vectors
                # Word embedding matrix
                word_embedding = tf.compat.v1.get_variable(
                    name="word_embedding",
                    dtype=tf.float32,
                    initializer=tf.compat.v1.random_uniform(
                        shape=[self.config.n_words, self.config.dim_word],
                        minval=-0.25, maxval=0.25))
            else:
                word_embedding = tf.compat.v1.get_variable(
                    name="word_embedding",
                    initializer=np.asarray(self.config.wordvec_matrix, dtype=np.float32),
                    dtype=tf.float32,
                    trainable=self.config.non_static)

            if self.config.use_chars:
                # Char embedding matrix
                char_embedding = tf.compat.v1.get_variable(
                    name="char_embedding",
                    dtype=tf.float32,
                    initializer=tf.compat.v1.random_uniform(
                        shape=[self.config.n_chars, self.config.dim_char],
                        minval=-0.25, maxval=0.25))

            self.word_vectors = tf.nn.embedding_lookup(
                word_embedding, self.word_ids, name="word_matrix")

            if self.config.use_chars:
                self.char_vectors = tf.nn.embedding_lookup(
                    char_embedding, self.char_ids, name="char_matrix")

            '''
                word_embedding = (batch size, max length of sentence in batch, self.config.dim_word)
                char_embedding = (batch size, max length of sentence in batch, max length of word, self.config.dim_char)
            '''

    def RNN_layer(self):
        '''
            Recurrent Layer
        '''

        def Cells(num_units, char_cell=False):
            '''
                Function to build cells
            '''
            # TODO: Wrappers

            if self.config.cells == "rnn":
                self.cell_fw = tf.compat.v1.nn.rnn_cell.BasicRNNCell(num_units=num_units)
                if self.config.bidirectional:
                    self.cell_bw = tf.compat.v1.nn.rnn_cell.BasicRNNCell(num_units=num_units)

            elif self.config.cells == "lstm":
                self.cell_fw = tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=num_units)
                if self.config.bidirectional:
                    self.cell_bw = tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=num_units)

            else:
                self.cell_fw = tf.compat.v1.nn.rnn_cell.GRUCell(num_units=num_units)
                if self.config.bidirectional:
                    self.cell_bw = tf.compat.v1.nn.rnn_cell.GRUCell(num_units=num_units)

            if char_cell:
                self.cell_fw = tf.compat.v1.nn.rnn_cell.DropoutWrapper(
                    cell=self.cell_fw, input_keep_prob=self.char_drop_input, output_keep_prob=self.char_drop_output, state_keep_prob=self.char_drop_state)
                if self.config.bidirectional:
                    self.cell_bw = tf.compat.v1.nn.rnn_cell.DropoutWrapper(
                        cell=self.cell_bw, input_keep_prob=self.char_drop_input, output_keep_prob=self.char_drop_output, state_keep_prob=self.char_drop_state)
            else:
                self.cell_fw = tf.compat.v1.nn.rnn_cell.DropoutWrapper(
                    cell=self.cell_fw, input_keep_prob=self.word_drop_input, output_keep_prob=self.word_drop_output, state_keep_prob=self.word_drop_state)
                if self.config.bidirectional:
                    self.cell_bw = tf.compat.v1.nn.rnn_cell.DropoutWrapper(
                        cell=self.cell_bw, input_keep_prob=self.word_drop_input, output_keep_prob=self.word_drop_output, state_keep_prob=self.word_drop_state)

        # Word Level Network
        if self.config.use_chars:
            with tf.compat.v1.variable_scope("word_layer"):
                # Put the word length in the axis 1 (time dimension)
                s = tf.shape(self.char_vectors)
                # new shape = [batch*sentence_length,word_length,char_dim]
                self.char_vectors = tf.reshape(self.char_vectors,
                                               shape=[s[0] * s[1], s[-2], self.config.dim_char])
                word_lengths = tf.reshape(self.word_lengths, shape=[s[0] * s[1]])

                # CELLS
                Cells(self.config.cell_char)

                # Bidirectional
                if self.config.bidirectional:
                    _, (output_state_fw, output_state_bw) = tf.compat.v1.nn.bidirectional_dynamic_rnn(
                        cell_fw=self.cell_fw, cell_bw=self.cell_bw, inputs=self.char_vectors,
                        sequence_length=word_lengths, dtype=tf.float32)

                    if self.config.cells == "lstm":
                        output_state_fw, output_state_bw = output_state_fw[1], output_state_bw[1]

                    self.char_output = tf.concat([output_state_fw, output_state_bw], axis=-1)

                # Unidirectional
                else:
                    _, output_state_fw = tf.compat.v1.nn.dynamic_rnn(
                        cell=self.cell_fw, inputs=self.char_vectors,
                        sequence_length=word_lengths, dtype=tf.float32)

                    if self.config.model == "lstm":
                        output_state_fw = output_state_fw[1]

                    self.char_output = output_state_fw

                # shape = (batch size, max sentence length, char hidden size)
                # self.h = self.char_output.shape[1].value
                self.h = self.char_output.shape[1]
                self.char_output = tf.reshape(self.char_output, shape=[s[0], s[1], self.h])
                self.word_vectors = tf.concat([self.word_vectors, self.char_output], axis=-1)

        # Sentence Level Network
        with tf.compat.v1.variable_scope("sentence_layer"):

            # Create Cells
            Cells(self.config.cell_word)

            # Bidirectional
            if self.config.bidirectional:
                (output_fw, output_bw), _ = tf.compat.v1.nn.bidirectional_dynamic_rnn(
                    cell_fw=self.cell_fw, cell_bw=self.cell_bw, inputs=self.word_vectors,
                    sequence_length=self.sentence_lengths, dtype=tf.float32)

                self.lstm_output = tf.concat([output_fw, output_bw], axis=-1)
            # Unidirectional
            else:
                output_state_fw, _ = tf.compat.v1.nn.dynamic_rnn(
                    cell=self.cell_fw, inputs=self.word_vectors,
                    sequence_length=self.sentence_lengths, dtype=tf.float32)

                self.lstm_output = output_state_fw

            # tf.shape() gets us the dynamic shape of a tensor
            # Save the max sentence length
            self.nsteps = tf.shape(self.lstm_output)[1]
            # .shape on the other hand provides the static shape of a tensor
            # Save the hidden length
            # self.h = self.lstm_output.shape[2].value
            self.h = self.lstm_output.shape[2]
            # current shape = [batch,max sentence length, hidden size]
            # after shape = [batch * max sentence , hidden size]
            self.layer_output = tf.reshape(self.lstm_output, [-1, self.h])

    def output_layer(self):
        with tf.compat.v1.variable_scope("output_layer"):
            layer = {
                'weights': tf.compat.v1.get_variable(name="W", initializer=tf.compat.v1.truncated_normal([self.h, self.config.n_tags])),
                'biases': tf.compat.v1.get_variable(name="b", initializer=tf.compat.v1.truncated_normal([self.config.n_tags]))
            }

            self.pred = tf.compat.v1.nn.xw_plus_b(
                self.layer_output, layer["weights"], layer["biases"], name="preds")
            self.logits = tf.reshape(
                self.pred, [-1, self.nsteps, self.config.n_tags], name="logits")

    def loss_function(self):
        with tf.compat.v1.variable_scope("loss_layer"):
            if self.config.use_crf:
                log_likelihood, trans_params = tfa.text.crf.crf_log_likelihood(
                    self.logits, self.labels, self.sentence_lengths)
                self.trans_params = tf.compat.v1.Variable(trans_params, name="trans_params",
                                                          trainable=None,
                                                          collections=None,
                                                          validate_shape=True,
                                                          caching_device=None,
                                                          variable_def=None,
                                                          dtype=None,
                                                          expected_shape=None,
                                                          import_scope=None,
                                                          constraint=None,
                                                          use_resource=None,
                                                          synchronization=tf.VariableSynchronization.AUTO,
                                                          aggregation=tf.compat.v1.VariableAggregation.NONE,
                                                          shape=None)
                self.loss = tf.reduce_mean(-log_likelihood)
            else:
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.logits, labels=self.labels)
                mask = tf.sequence_mask(self.sentence_lengths)
                losses = tf.boolean_mask(losses, mask)
                self.loss = tf.reduce_mean(losses)

    def train_op(self):
        with tf.compat.v1.variable_scope("train_step"):
            self.gradient = self.optimizer.compute_gradients(loss=self.loss)
            self.train_op = self.optimizer.apply_gradients(grads_and_vars=self.gradient,
                                                           global_step=self.global_step)

    def build(self):
        self.input_layer()
        self.embedding_layer()
        self.RNN_layer()
        self.output_layer()
        self.loss_function()
        # Generic functions that add training op and initialize session
        self.train_op()
        self.initialize_session()  # now self.sess is defined and vars are init

    def load_model(self, dir):
        self.initialize_session()
        self.saver = tf.compat.v1.train.import_meta_graph("{}.meta".format(dir))
        self.saver.restore(self.sess, dir)

        # Get the operations easily
        graph = tf.compat.v1.get_default_graph()
        # INPUT_LAYER
        self.word_ids = graph.get_operation_by_name("input_layer/word_ids").outputs[0]
        self.sentence_lengths = graph.get_operation_by_name(
            "input_layer/sentence_lengths").outputs[0]
        self.char_ids = graph.get_operation_by_name("input_layer/char_ids").outputs[0]
        self.word_lengths = graph.get_operation_by_name("input_layer/word_lengths").outputs[0]
        self.labels = graph.get_operation_by_name("input_layer/labels").outputs[0]
        # OUTPUT_LAYER
        self.logits = graph.get_operation_by_name("output_layer/logits").outputs[0]
        # CRF
        if self.config.use_crf:
            self.trans_params = graph.get_operation_by_name("loss_layer/trans_params").outputs[0]

    def __init__(self, config):
        super(RNN, self).__init__(config)

    def predict_batch(self, feed):
        # Batch Prediction
        # CRF Prediction
        if self.config.use_crf:
            # get tag scores and transition params of CRF
            viterbi_sequences = []
            logits, trans_params = self.sess.run(
                [self.logits, self.trans_params], feed_dict=feed)

            # iterate over the sentences because no batching in vitervi_decode
            for logit, sentence_length in zip(logits, feed[self.sentence_lengths]):
                logit = logit[:sentence_length]  # keep only the valid steps
                viterbi_seq, _ = tfa.text.viterbi_decode(
                    logit, trans_params)
                viterbi_sequences.append(viterbi_seq)

            return viterbi_sequences
        # Softmax Prediction
        else:
            labels_pred = self.sess.run(self.logits, feed_dict=feed)
            # labels_pred = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)
            labels_pred = np.argmax(labels_pred, axis=-1)
            return labels_pred
