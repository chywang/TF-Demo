import numpy as np
import tensorflow as tf


class CNN_Cls():
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters,
                 l2_reg_lambda):
        # input, output
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

        self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_size])
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

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

        # output layer
        with tf.name_scope("output"):
            softmax_weights = tf.Variable(tf.truncated_normal([self.feature_size, num_classes],
                                                              stddev=0.1))
            softmax_biases = tf.Variable(tf.zeros([num_classes], dtype=tf.float32))
            self.logit = tf.matmul(self.cnn_pool_drop, softmax_weights) + softmax_biases
            self.prob = tf.nn.sigmoid(self.logit)
            self.predictions = tf.argmax(self.logit, 1, name="predictions")

        with tf.name_scope("loss"):
            self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logit, labels=self.input_y)
            self.loss = tf.reduce_mean(self.loss) + l2_reg_lambda * (
                tf.nn.l2_loss(softmax_weights) + tf.nn.l2_loss(softmax_biases))
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
