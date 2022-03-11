import nltk
import os
import sys
import numpy as np
import tensorflow as tf
import re
from tqdm import tqdm
import sys
import os

store_base_path = './text_data/'
stemmer = nltk.stem.porter.PorterStemmer()
lemmatizer = nltk.stem.WordNetLemmatizer()


def _read_file(base_folder):
    positive = ''
    negative = ''

    full_files = ['./processed_acl/kitchen/positive.review',
                  './processed_acl/kitchen/negative.review',
                  './processed_acl/kitchen/unlabeled.review',
                  './processed_acl/dvd/positive.review',
                  './processed_acl/dvd/negative.review',
                  './processed_acl/dvd/unlabeled.review',
                  './processed_acl/electronics/positive.review',
                  './processed_acl/electronics/negative.review',
                  './processed_acl/electronics/unlabeled.review',
                  './processed_acl/books/positive.review',
                  './processed_acl/books/negative.review',
                  './processed_acl/books/unlabeled.review']

    for ind, file in enumerate(full_files):
        print("Processing file:", file)
        file = open(file, 'r')
        for line in tqdm(file.readlines()):
            line = re.sub('[^a-zA-Z]', ' ', line).replace('  ', ' ')
            label = line[-9:]
            line = line[:len(line)-16]
            review = line.lower()
            review = review.split()
            review = [lemmatizer.lemmatize(
                word) for word in review if word not in nltk.corpus.stopwords.words('english')]
            review = ' '.join(review)

            obj = full_files[ind].split('/')[2]

            if 'pos' in label:
                try:
                    f = open(obj + '_positive.txt', 'a')
                    f.write(review)
                    f.write(' ')

                except Exception as e:
                    print(e)
                finally:
                    f.close()
            if 'neg' in label:
                try:
                    f1 = open(obj + '_negative.txt', 'a')
                    f1.write(review)
                    f1.write(' ')
                except Exception as e:
                    print(e)
                finally:
                    f1.close()

        file.close()
    os.system('mv *.txt {}'.format(store_base_path))
    print('Done.')


def tokenize_file(file):
    f = open(file, 'r')
    data = f.readline()
    f.close()

    data = nltk.word_tokenize(data)
    tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='<OOV>')
    tokenizer.fit_on_texts(data)

    word_index = tokenizer.word_index
    sequence = tokenizer.texts_to_sequences(data)

    return sequence


def text_to_image(pos_data, neg_data, image_size=64):
    max_len = 0
    if len(pos_data) > len(neg_data):
        max_len = len(neg_data)
    else:
        max_len = len(pos_data)

    pos_data = np.array(pos_data[:max_len]).reshape(-1,)
    neg_data = np.array(neg_data[:max_len]).reshape(-1,)

    embedding = tf.keras.layers.Embedding(20000, image_size)
    batches = int(np.floor(max_len/image_size))

    data = []
    label = []

    for img_batch in tqdm(range(batches)):
        img_pos = embedding(tf.constant(
            pos_data[img_batch*(image_size):(img_batch+1)*image_size]))
        img_neg = embedding(tf.constant(
            neg_data[img_batch*(image_size):(img_batch+1)*image_size]))
        data.append(img_pos)
        data.append(img_neg)
        label.append(1)
        label.append(0)

        if (img_batch+1) % 1000 == 0:
            img_list = tf.train.FloatList(value=np.array(data).reshape(-1,))
            label_list = tf.train.Int64List(
                value=np.array(label))

            image = tf.train.Feature(float_list=img_list)
            labels = tf.train.Feature(int64_list=label_list)

            fully = {
                'image': image,
                'labels': labels
            }
            full = tf.train.Features(feature=fully)
            example = tf.train.Example(features=full)

            with tf.io.TFRecordWriter('music_to_array_'+str(img_batch)+'.tfrecord') as writer:
                writer.write(example.SerializeToString())

            data = []
            label = []
    # return np.array(data)


if __name__ == '__main__':
    _read_file('./processed_acl/')

    for category in ['books', 'dvd', 'electronics', 'kitchen']:
        print('Processing category:', category)
        pos = tokenize_file('./text_data/'+category+'_positive.txt')
        neg = tokenize_file('./text_data/'+category+'_negative.txt')
        text_to_image(pos,neg)
        os.system('mkdir -p ./text_image/'+category)
        os.system('mv *.tfrecord ./text_image/'+category)

