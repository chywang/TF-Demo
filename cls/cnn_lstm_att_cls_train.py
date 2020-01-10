from cls.data_utils import *
from cls.cnn_lstm_att_cls import CNN_LSTM_Att_Cls
import tensorflow as tf
import numpy as np
from tensorflow.contrib import learn

# paths
tf.flags.DEFINE_string("positive_data_file", "rt-polaritydata/rt-polarity.pos", "")
tf.flags.DEFINE_string("negative_data_file", "rt-polaritydata/rt-polarity.neg", "")
tf.flags.DEFINE_string("embedding_data_file", "../glove.6B.300d.txt", "")

# parameters
tf.flags.DEFINE_float("dev_sample_percentage", .1, "")
tf.flags.DEFINE_float("dropout_keep_prob", .5, "")
tf.flags.DEFINE_integer("embedding_dim", 300, "")
tf.flags.DEFINE_integer("hidden_dim", 300, "")
tf.flags.DEFINE_integer("attention_size", 300, "")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "")
tf.flags.DEFINE_integer("num_filters", 100, "")
tf.flags.DEFINE_integer("batch_size", 128, "")
tf.flags.DEFINE_integer("num_epochs", 1000, "")
tf.flags.DEFINE_integer("num_layers", 2, "")


FLAGS = tf.flags.FLAGS


def train_for_one_batch(x_batch, y_batch, train_step, curr_epoch):
    """
    A single training step
    """
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.embedding_placeholder: embedding,
        model.dropout_keep_prob: FLAGS.dropout_keep_prob
    }
    _, loss, accuracy = sess.run(
        [train_step, model.loss, model.accuracy],
        feed_dict)
    if curr_epoch % 10 == 0:
        print("epoch {0} train acc {1}".format(curr_epoch, accuracy))


def test(x_batch, y_batch, curr_epoch):
    """
    Evaluates model on a dev set
    """
    if curr_epoch % 10 != 0:
        return

    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.embedding_placeholder: embedding,
        model.dropout_keep_prob: FLAGS.dropout_keep_prob
    }
    loss, accuracy = sess.run(
        [model.loss, model.accuracy],
        feed_dict)
    print("epoch {0} test acc {1}".format(curr_epoch, accuracy))


x_text, y = load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
print("loading training and testing data ok...")

max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))

vocab_dict = vocab_processor.vocabulary_._mapping
embedding = load_GloVe(FLAGS.embedding_data_file, vocab_dict)
print("loading glove ok...")

shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))  # Vocabulary Size: 18758
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))  # Train/Dev split: 9596/1066

with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        model = CNN_LSTM_Att_Cls(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            num_layers=FLAGS.num_layers,
            hidden_dim=FLAGS.hidden_dim,
            attention_size=FLAGS.attention_size
        )

        # Define Training procedure
        optimizer = tf.train.AdamOptimizer()
        train_step = optimizer.minimize(model.loss)
        sess.run(tf.global_variables_initializer())

        for i in range(0, FLAGS.num_epochs):
            x_batch_train, y_batch_train = create_one_batch(x_train, y_train, FLAGS.batch_size, curr_epoch=i,
                                                            shuffle=True)
            train_for_one_batch(x_batch_train, y_batch_train, train_step, curr_epoch=i)
            test(x_dev, y_dev, curr_epoch=i)
