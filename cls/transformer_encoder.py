import tensorflow as tf
import numpy as np


class TransformerEncoder():
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, num_heads):

        self.embedding_size = embedding_size
        self.num_heads = num_heads

        # input, output
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_size])

        # embedding layer
        with tf.name_scope("embedding"):
            W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_size]), trainable=True, name="W")
            embedding_init = W.assign(self.embedding_placeholder)
            self.raw_embedded = tf.nn.embedding_lookup(embedding_init, self.input_x)
            self.positional_embedded = self.sinusoidal_positional_encoding(self.input_x, embedding_size)
            self.x_embedded = self.raw_embedded + self.positional_embedded
            # self.x_embedded = tf.reduce_mean(self.x_embedded, axis=1)

        # attention layer
        with tf.name_scope("attention"):
            self.x_attn = self.multihead_attention(queries=self.x_embedded, keys=self.x_embedded)
            self.output = self.feed_forward(self.x_attn, [self.embedding_size, self.embedding_size])
            self.output = tf.reshape(self.output, [-1, sequence_length * self.embedding_size])

        # output layer
        with tf.name_scope("output"):
            self.logit = tf.layers.dense(self.output, num_classes)
            self.prob = tf.nn.sigmoid(self.logit)
            self.predictions = tf.argmax(self.logit, 1, name="predictions")

        with tf.name_scope("loss"):
            self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logit, labels=self.input_y)
            self.loss = tf.reduce_mean(self.loss)
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def layer_normalization(self, inputs,
                            epsilon=1e-8,
                            scope="ln",
                            reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]
            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.Variable(tf.zeros(params_shape))
            gamma = tf.Variable(tf.ones(params_shape))
            normalized = (inputs - mean) / ((variance + epsilon) ** .5)
            outputs = gamma * normalized + beta
        return outputs

    def multihead_attention(self, queries,
                            keys,
                            num_units=None,
                            dropout_rate=0,
                            causality=False,
                            scope="multihead_attention",
                            reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            if num_units is None:  # set default size for attention size C
                num_units = queries.get_shape().as_list()[-1]
            # Linear Projections
            Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # [N, T_q, C]
            K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # [N, T_k, C]
            V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # [N, T_k, C]
            # Split and concat
            Q_ = tf.concat(tf.split(Q, self.num_heads, axis=-1), axis=0)  # [num_heads * N, T_q, C/num_heads]
            K_ = tf.concat(tf.split(K, self.num_heads, axis=-1), axis=0)  # [num_heads * N, T_k, C/num_heads]
            V_ = tf.concat(tf.split(V, self.num_heads, axis=-1), axis=0)  # [num_heads * N, T_k, C/num_heads]
            # Attention
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (num_heads * N, T_q, T_k)
            # Scale : outputs = outputs / sqrt( d_k)
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
            # Key Masking
            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
            key_masks = tf.tile(key_masks, [self.num_heads, 1])  # (h*N, T_k)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)
            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)  # -infinity
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

            # Causality = Future blinding
            if causality:
                diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
                tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense()  # (T_q, T_k)
                masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)
                paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
                outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

            # Activation: outputs is a weight matrix
            outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)
            # Query Masking
            query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
            query_masks = tf.tile(query_masks, [self.num_heads, 1])  # (h*N, T_q)
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
            outputs *= query_masks  # broadcasting. (N, T_q, C)
            # dropouts
            outputs = tf.layers.dropout(outputs, rate=dropout_rate)
            # weighted sum
            outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)
            # reshape
            outputs = tf.concat(tf.split(outputs, self.num_heads, axis=0), axis=2)  # (N, T_q, C)
            # residual connection
            outputs += queries
            # layer normaliztion
            outputs = self.layer_normalization(outputs)
            return outputs

    def feed_forward(self, inputs,
                     num_units=[2048, 512],
                     scope="multihead_attention",
                     reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            # Inner layer
            params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                      "activation": tf.nn.relu, "use_bias": True}
            outputs = tf.layers.conv1d(**params)
            # Readout layer
            params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                      "activation": None, "use_bias": True}
            outputs = tf.layers.conv1d(**params)
            print("Conv ret:", outputs.shape)
            # Residual connection
            outputs += inputs
            # Normalize
            outputs = self.layer_normalization(outputs)
        return outputs

    def sinusoidal_positional_encoding(self, inputs, zero_pad=False, scale=False):
        T = inputs.get_shape().as_list()[1]
        position_idx = tf.tile(tf.expand_dims(tf.range(T), 0), [tf.shape(inputs)[0], 1])
        position_enc = np.array(
            [[pos / np.power(10000, 2. * i / self.embedding_size) for i in range(self.embedding_size)] for pos in
             range(T)])
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])
        lookup_table = tf.convert_to_tensor(position_enc, tf.float32)
        if zero_pad:
            lookup_table = tf.concat([tf.zeros([1, self.embedding_size]), lookup_table[1:, :]], axis=0)
        outputs = tf.nn.embedding_lookup(lookup_table, position_idx)
        if scale:
            outputs = outputs * self.embedding_size ** 0.5
        return outputs
