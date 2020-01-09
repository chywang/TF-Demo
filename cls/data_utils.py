import numpy as np
import tensorflow as tf
import re


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def create_one_batch(x_data, y_data, batch_size, curr_epoch, shuffle=True):
    """
    Generates a batch for a dataset.
    """
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(len(x_data)))
        shuffled_x_data = x_data[shuffle_indices]
        shuffled_y_data = y_data[shuffle_indices]
    else:
        shuffled_x_data = x_data
        shuffled_y_data = y_data
    start_index = (batch_size * curr_epoch) % len(x_data)
    end_index = start_index + batch_size
    return shuffled_x_data[start_index:end_index], shuffled_y_data[start_index:end_index]


def load_GloVe(filename, vocab_dict):
    # vocab_dict: word-id mappings
    # generate glove mappings
    glove_dict={}
    file = open(filename, 'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        glove_dict[row[0]]=row[1:]
    file.close()
    glove_dict['<UNK>'] = [0] * tf.flags.FLAGS.embedding_dim

    #create embedding matrix
    embeddings=np.zeros(shape=(len(vocab_dict.keys()), tf.flags.FLAGS.embedding_dim))
    for word, id in vocab_dict.items():
        if word in glove_dict.keys():
            embeddings[id]=glove_dict[word]
        else:
            embeddings[id] = np.random.rand(tf.flags.FLAGS.embedding_dim)
    return embeddings