import tensorflow as tf
import os
import sys
import re
import nltk
import tqdm

base_folder = 'processed_acl'

data = []
labels = []

# for sub_folder in os.listdir(base_folder):
#     sub_fold = (base_folder+'/'+sub_folder)
#     for files in os.listdir(sub_fold):
#         files = sub_fold+'/'+files
#         file = open(files, 'r')
#         for line in file.readlines():
#             line = re.sub('[^a-zA-Z]', ' ', line).replace('  ', ' ')
#             line = line[:len(line)-16]
#             data.append(line)
#             labels.append(1 if line[-9:] == 'positive' else 0)
#         file.close()

sentence_list = tf.train.BytesList(value=[b'sentence1', b'sentence2'])
token_list = tf.train.FloatList(value=[1.0, 2.0])

sentences = tf.train.Feature(bytes_list=sentence_list)
tokens = tf.train.Feature(float_list=token_list)

sentence_dict = {
    'sentence': sentences,
    'Token': tokens
}
feature_sentence = tf.train.Features(feature=sentence_dict)

example = tf.train.Example(features=feature_sentence)

with tf.io.TFRecordWriter('sentences.tfrecord') as writer:
    writer.write(example.SerializeToString())
