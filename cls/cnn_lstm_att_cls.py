import numpy as np
import tensorflow as tf


class CNN_LSTM_Att_Cls():
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters, num_layers,
                 hidden_dim, attention_size):
        # input, output
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

        self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_size])
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        def attention(inputs, attention_size, time_major=False, return_alphas=False):
            if isinstance(inputs, tuple):
                # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
                inputs = tf.concat(inputs, 2)
            if time_major:
                # (T,B,D) => (B,T,D)
                inputs = tf.array_ops.transpose(inputs, [1, 0, 2])
            hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer
            # Trainable parameters
            w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
            b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
            u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
            with tf.name_scope('v'):
                # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
                #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
                v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)
            # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
            vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
            alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape
            # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
            output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)
            if not return_alphas:
                return output
            else:
                return output, alphas

        # embedding layer
        with tf.name_scope("embedding"):
            W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_size]), trainable=True, name="W")
            embedding_init = W.assign(self.embedding_placeholder)
            self.raw_embedded = tf.nn.embedding_lookup(embedding_init, self.input_x)
            self.raw_embedded_expanded = tf.expand_dims(self.raw_embedded, -1)

        # cnn and pooling layers
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("cnn-filter-%s" % filter_size):
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.raw_embedded_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID")
                h = tf.nn.relu(tf.nn.bias_add(conv, b))
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID')
                pooled_outputs.append(pooled)

        # cnn concat layer
        with tf.name_scope("cnn-concat-dropout"):
            self.feature_size = num_filters * len(filter_sizes)
            self.cnn_pool = tf.concat(pooled_outputs, axis=3)
            self.cnn_pool_flat = tf.reshape(self.cnn_pool, [-1, self.feature_size])
            self.cnn_pool_drop = tf.nn.dropout(self.cnn_pool_flat, self.dropout_keep_prob)

        # lstm layers
        with tf.name_scope("lstm-att"):
            cells = [tf.contrib.rnn.BasicLSTMCell(hidden_dim, state_is_tuple=True) for _ in range(num_layers)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
            self.lstm_outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=self.raw_embedded,
                                                     dtype=tf.float32)
            self.attention_output = attention(self.lstm_outputs, attention_size)

        # output layer
        with tf.name_scope("output"):
            joint_pool = tf.concat([self.cnn_pool_drop, self.attention_output], axis=1)
            dense_out = tf.layers.dense(joint_pool, hidden_dim)
            dense_out = tf.nn.relu(dense_out)
            self.logit = tf.layers.dense(dense_out, num_classes)
            self.prob = tf.nn.sigmoid(self.logit)
            self.predictions = tf.argmax(self.logit, 1, name="predictions")

        with tf.name_scope("loss"):
            self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logit, labels=self.input_y)
            self.loss = tf.reduce_mean(self.loss)
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
