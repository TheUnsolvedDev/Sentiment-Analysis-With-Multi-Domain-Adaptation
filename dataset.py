import nltk
import os
import numpy as np
import tensorflow as tf

objects = [i for i in os.listdir('./text_image')]
base_path = './text_image/'
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 2000


def _parse_function(example_proto):
    # Parse the input `tf.train.Example` proto using the dictionary above.
    feature_desc = {
        'image': tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0.0, allow_missing=True),
        'labels': tf.io.FixedLenSequenceFeature([], tf.int64, default_value=0, allow_missing=True)
    }
    return tf.io.parse_single_example(example_proto, feature_desc)


class Datagenerator(tf.keras.utils.Sequence):
    def __init__(self, data, labels, shuffle=True, batch_size=500):
        self.data = data
        self.labels = labels

        self.indices = np.arange(len(self.data))
        self.batch_size = batch_size
        if shuffle:
            self.shuffle_on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.data)/self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[(
            index)*self.batch_size:(index+1)*self.batch_size]
        return self.data[indices], self.labels[indices]

    def shuffle_on_epoch_end(self):
        np.random.shuffle(self.indices)


class Data:
    def __init__(self, data=objects):
        self.objects = data
        self.base_path = base_path
        self.image_size = IMAGE_SIZE
        self.batch_size = BATCH_SIZE

    def task(self, index):
        if 0 <= index <= len(self.objects):
            index -= 1
        else:
            return "RTFM"

        data = []
        labels = []
        files = [self.base_path+self.objects[index]+'/' +
                 i for i in os.listdir(self.base_path + self.objects[index])]

        for file in files:
            raw = tf.data.TFRecordDataset(file).map(_parse_function)
            for parsed_record in raw.take(5):
                data.append(
                    np.array(parsed_record['image']).reshape(2000, 64, 64))
                labels.append(
                    np.array(parsed_record['labels']).reshape(2000, 1))

        data = np.array(data).reshape(len(data)*2000, 64, 64,1)
        labels = np.array(labels).reshape(len(labels)*2000, 1)

        return Datagenerator(data, labels)


if __name__ == '__main__':
    # file = ['./text_image/books/music_to_array_999.tfrecord']
    # raw = tf.data.TFRecordDataset(file).map(_parse_function)
    # for parsed_record in raw.take(5):
    #     print(parsed_record)

    # print(objects)
    obj = Data()
    for i in range(1, 5):
        oj = obj.task(i)
        print('New task')
        for dat in range(oj.__len__()):
            a, b = oj.__getitem__(dat)
            print(a.shape, b.shape)
