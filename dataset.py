import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import os
import xml.etree.ElementTree as ET
import random

from params import *


files = ['data/'+i+'/'+j for i in os.listdir('data')
         for j in os.listdir('data/'+i) if 'parsed' in j]


def read_file(file):
    data = []
    root = ET.parse(file).getroot()
    for review in root.findall('review'):
        data.append(review.text.replace('\t', '').replace('\n', ''))
    if 'positive' in file:
        labels = [1 for i in data]
    else:
        labels = [0 for i in data]
    return data, labels


class Dataset:
    def __init__(self, class_num=0):
        if 0 <= class_num < 4:
            file = files[2*class_num]
            print(file)
            pos_data, pos_labels = read_file(file)
            neg_data, neg_labels = read_file(file.replace('positive', 'negative'))

            data = pos_data + neg_data
            labels = pos_labels + neg_labels

        self.int_vectorize_layer = tf.keras.layers.TextVectorization(
            max_tokens=VOCAB_SIZE,
            output_mode='int',
            output_sequence_length=MAX_SEQUENCE_LENGTH)
        self.int_vectorize_layer.adapt(data)
        self.data = tf.data.Dataset.from_tensor_slices(
            data).map(tf.strings.lower).map(self.int_vectorize_layer)
        self.labels = tf.data.Dataset.from_tensor_slices(labels)
        self.batch_size = BATCH_SIZE

    def get_data(self):
        self.dataset = tf.data.Dataset.zip((self.data, self.labels)).shuffle(
            1000, reshuffle_each_iteration=True)
        return self.dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)


if __name__ == '__main__':
    data = Dataset(2)
    for elem in data.get_data():
        print(elem[0].numpy(), elem[1].numpy())
