from cls.data_utils import *
from cls.lstm_cls import LSTM_Cls
from cls.student_cls import Student_Cls
import tensorflow as tf
import numpy as np
from tensorflow.contrib import learn


def teacher_predict_logits(model, x_batch):
    feed_dict = {
        model.input_x: x_batch,
        model.embedding_placeholder: embedding
    }
    return sess.run(model.logit, feed_dict)


def test_on_dev_set(model, x_batch, y_batch):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.embedding_placeholder: embedding
    }
    loss, accuracy = sess.run(
        [model.loss, model.accuracy],
        feed_dict)
    print("test acc {0}".format(accuracy))


def train_for_one_batch(model, sess, x_batch, y_batch, train_step, curr_epoch):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.embedding_placeholder: embedding
    }
    _, loss, accuracy = sess.run([train_step, model.loss, model.accuracy], feed_dict)
    if curr_epoch % 10 == 0:
        print("epoch {0} train acc {1}".format(curr_epoch, accuracy))


def train_teacher_network(model, train_step):
    for i in range(0, FLAGS.teacher_num_epochs):
        x_batch_train, y_batch_train = create_one_batch(x_train, y_train, FLAGS.batch_size, curr_epoch=i,
                                                        shuffle=True)
        train_for_one_batch(model, sess, x_batch_train, y_batch_train, train_step, curr_epoch=i)


def train_student_network(teacher_network, student_network, student_train_step):
    for i in range(0, FLAGS.student_num_epochs):
        x_batch_train, y_batch_train = create_one_batch(x_train, y_train, FLAGS.batch_size, curr_epoch=i,
                                                        shuffle=True)
        train_for_one_batch_for_student(teacher_network, student_network, x_batch_train, y_batch_train,
                                        student_train_step,
                                        curr_epoch=i)


def train_for_one_batch_for_student(teacher_network, student_network, x_batch, y_batch, train_step, curr_epoch):
    teacher_logits = teacher_predict_logits(teacher_network, x_batch)
    feed_dict = {
        student_network.input_x: x_batch,
        student_network.input_y: y_batch,
        student_network.teacher_y: teacher_logits,
        student_network.embedding_placeholder: embedding
    }
    _, loss, accuracy = sess.run(
        [train_step, student_network.teacher_loss, student_network.learned_accuracy],
        feed_dict)
    if curr_epoch % 10 == 0:
        print("epoch {0} train acc {1}".format(curr_epoch, accuracy))


def test_on_dev_set_for_student(model, x_batch, y_batch):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.embedding_placeholder: embedding
    }
    accuracy = sess.run(
        model.learned_accuracy,
        feed_dict)
    print("test acc {0}".format(accuracy))


# paths
tf.flags.DEFINE_string("positive_data_file", "rt-polaritydata/rt-polarity.pos", "")
tf.flags.DEFINE_string("negative_data_file", "rt-polaritydata/rt-polarity.neg", "")
tf.flags.DEFINE_string("embedding_data_file", "../glove.6B.300d.txt", "")

# parameters
tf.flags.DEFINE_float("dev_sample_percentage", .1, "")
tf.flags.DEFINE_integer("embedding_dim", 300, "")
tf.flags.DEFINE_integer("num_heads", 6, "")
tf.flags.DEFINE_integer("hidden_dim", 300, "")
tf.flags.DEFINE_integer("num_layers", 2, "")
tf.flags.DEFINE_integer("batch_size", 128, "")
tf.flags.DEFINE_integer("teacher_num_epochs", 200, "")
tf.flags.DEFINE_integer("student_num_epochs", 200, "")
tf.flags.DEFINE_integer("temperature", 1, "")
tf.flags.DEFINE_float("lam", .2, "")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "")

FLAGS = tf.flags.FLAGS

# load datasets

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

print("Building networks")
# train teacher network
teacher_network = LSTM_Cls(
    sequence_length=x_train.shape[1],
    num_classes=y_train.shape[1],
    vocab_size=len(vocab_processor.vocabulary_),
    embedding_size=FLAGS.embedding_dim,
    hidden_dim=FLAGS.hidden_dim,
    keep_prob=FLAGS.dropout_keep_prob,
    num_layers=FLAGS.num_layers
)
student_network = Student_Cls(
    sequence_length=x_train.shape[1],
    num_classes=y_train.shape[1],
    vocab_size=len(vocab_processor.vocabulary_),
    embedding_size=FLAGS.embedding_dim,
    hidden_dim=FLAGS.hidden_dim,
    num_layers=FLAGS.num_layers,
    temperature=FLAGS.temperature,
    lam=FLAGS.lam
)

optimizer = tf.train.AdamOptimizer()
teacher_train_step = optimizer.minimize(teacher_network.loss)
student_train_step = optimizer.minimize(student_network.teacher_loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

print("Learn Teacher Network")
train_teacher_network(teacher_network, teacher_train_step)
test_on_dev_set(teacher_network, x_dev, y_dev)

print("Learn Student Network")
train_student_network(teacher_network, student_network, student_train_step)
test_on_dev_set_for_student(student_network, x_dev, y_dev)
