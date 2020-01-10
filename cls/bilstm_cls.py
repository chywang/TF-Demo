import tensorflow as tf


class BiLSTM_Cls():
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, hidden_dim, keep_prob):
        # input, output
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_size])

        def lstm_cell():
            return tf.contrib.rnn.LSTMCell(hidden_dim, state_is_tuple=True)

        def dropout():
            cell = lstm_cell()
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)

        # embedding layer
        with tf.name_scope("embedding"):
            W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_size]), trainable=True, name="W")
            embedding_init = W.assign(self.embedding_placeholder)
            self.raw_embedded = tf.nn.embedding_lookup(embedding_init, self.input_x)

        # lstm layers
        with tf.name_scope("bilstm"):
            lstm_cell_fw = dropout()
            lstm_cell_bw = dropout()
            _outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell_fw, cell_bw=lstm_cell_bw,
                                                          inputs=self.raw_embedded,
                                                          dtype=tf.float32)
            # [batch_size, max_time, depth]
            last = tf.concat(_outputs, 2)
            # [batch_size, depth]
            last = tf.reduce_mean(last, axis=1)

        # output layer
        with tf.name_scope("output"):
            dense_out = tf.layers.dense(last, hidden_dim)
            dense_out = tf.contrib.layers.dropout(dense_out, keep_prob)
            dense_out = tf.nn.relu(dense_out)

            self.logit = tf.layers.dense(dense_out, num_classes)
            self.prob = tf.nn.sigmoid(self.logit)
            self.predictions = tf.argmax(self.logit, 1, name="predictions")

        with tf.name_scope("loss"):
            self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logit, labels=self.input_y)
            self.loss = tf.reduce_mean(self.loss)
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
