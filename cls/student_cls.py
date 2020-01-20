import tensorflow as tf


class Student_Cls():
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, hidden_dim, temperature,
                 lam):
        # input, output
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.teacher_y = tf.placeholder(tf.float32, [None, num_classes], name="teacher_y")
        self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_size])

        # embedding layer
        with tf.name_scope("embedding"):
            W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_size]), trainable=True, name="W")
            embedding_init = W.assign(self.embedding_placeholder)
            self.raw_embedded = tf.nn.embedding_lookup(embedding_init, self.input_x)
            self.raw_embedded = tf.reduce_mean(self.raw_embedded, axis=1)

        # output layer
        with tf.name_scope("output"):
            dense_out = tf.layers.dense(self.raw_embedded, hidden_dim)
            dense_out = tf.nn.relu(dense_out)

            self.logit = tf.layers.dense(dense_out, num_classes)
            self.prob = tf.nn.sigmoid(self.logit)
            self.predictions = tf.argmax(self.logit, 1, name="predictions")
            learned_correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.learned_accuracy = tf.reduce_mean(tf.cast(learned_correct_predictions, "float"),
                                                   name="learned_accuracy")

        with tf.name_scope("loss"):
            self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logit, labels=self.input_y)
            self.loss = tf.reduce_mean(self.loss)
            # correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            # self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

            self.teacher_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logit / temperature,
                                                                        labels=tf.nn.sigmoid(
                                                                            self.teacher_y / temperature))
            self.teacher_loss = tf.reduce_mean(self.teacher_loss) * lam + self.loss * (1 - lam)
