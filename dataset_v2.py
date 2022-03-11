import numpy as np
import xml.etree.ElementTree as tree
import nltk
import tensorflow as tf
import re
import os


def read_file(folder, destination):
    for fold in os.listdir(folder):
        os.system("mkdir -p "+destination+'/'+fold)
        for file in os.listdir(folder+'/'+fold):
            with open(folder+'/'+fold+'/'+file, 'r') as f:
                xml = f.readlines()
            print(xml)            

    # with open(filename, 'r') as f:
    #     xml = f.readlines()
    # print(xml)


if __name__ == '__main__':
    read_file('sorted_data_acl', 'text_xml')
