import tensorflow as tf
import numpy as np
from torch import le
from tqdm import tqdm

from params import *


def classification_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GlobalAveragePooling1D(),
        # tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])
    # model.compile(loss='binary_crossentropy',
    #               optimizer='adam', metrics=['accuracy'])
    return model


def ewc_fisher_matrix(datas, model, samples=150):
    fisher = [tf.zeros_like(tensor) for tensor in model.trainable_weights]
    length = BATCH_SIZE

    for data, labels in tqdm(datas):
        for sample in range(samples):
            # data = data.numpy()
            # labels = labels.numpy()
            num = np.random.randint(data.shape[0])
            with tf.GradientTape() as tape:
                probs = (model(tf.expand_dims(data[num], axis=0)))
                log_likelyhood = tf.math.log(probs)
                # log_likelyhood = model.loss(label[num],probs)

            derv = tape.gradient(log_likelyhood, model.weights)
            fisher = [(fis + tf.convert_to_tensor(dv)**2) for fis, dv in zip(fisher, derv)]

    fisher = [fish/((samples)*length) for fish in fisher]
    return fisher


if __name__ == '__main__':
    m = classification_model()
    m.summary()
