import numpy as np
import tensorflow as tf
from tqdm import tqdm

IMAGE_SIZE = (64, 64, 1)


def Lenet5(input_shape=IMAGE_SIZE):
    inputs = tf.keras.Input(shape=input_shape)
    conv1 = tf.keras.layers.Conv2D(
        filters=32, kernel_size=5, activation='tanh', padding='same')(inputs)
    maxpool2 = tf.keras.layers.MaxPool2D(pool_size=(
        2, 2), strides=None, padding='valid')(conv1)
    droupout1 = tf.keras.layers.Dropout(0.25)(maxpool2)
    conv3 = tf.keras.layers.Conv2D(
        filters=64, kernel_size=5, activation='tanh', padding='same')(maxpool2)
    maxpool3 = tf.keras.layers.MaxPool2D(pool_size=(
        2, 2), strides=None, padding='valid')(conv3)
    dropout2 = tf.keras.layers.Dropout(0.25)(maxpool3)

    flat = tf.keras.layers.Flatten()(dropout2)
    fc1 = tf.keras.layers.Dense(units=240, activation='tanh')(flat)
    fc2 = tf.keras.layers.Dense(units=128, activation='tanh')(fc1)
    final = tf.keras.layers.Dense(units=1, activation='sigmoid')(fc2)

    model = tf.keras.Model(inputs=inputs, outputs=final)
    return model

def DNN(input_shape=IMAGE_SIZE):
    inputs = tf.keras.Input(shape=input_shape)
    flat = tf.keras.layers.Flatten()(inputs)
    fc1 = tf.keras.layers.Dense(units=1024, activation='tanh')(flat)
    fc2 = tf.keras.layers.Dense(units=1024, activation='tanh')(fc1)
    final = tf.keras.layers.Dense(units=1, activation='sigmoid')(fc2)

    model = tf.keras.Model(inputs=inputs, outputs=final)
    return model


def ewc_fisher_matrix(gens, model, samples=400):
    fisher = [tf.zeros_like(tensor) for tensor in model.trainable_weights]
    length = len(gens)

    datas = []
    labelss = []
    for gen in gens:
        entry1 = []
        entry2 = []
        batches_total = gen.__len__()

        for i in range(samples):
            choice = np.random.randint(0, batches_total)
            data, labels = gen.__getitem__(choice)
            choice_2 = np.random.randint(0, len(data))
            entry1.append(data[choice_2])
            entry2.append(labels[choice_2])

        dt = np.array(entry1)
        lb = np.array(entry2)
        datas.append(dt)
        labelss.append(lb)

    for label, data in zip(labelss, datas):
        for sample in tqdm(range(samples)):
            num = np.random.randint(data.shape[0])
            with tf.GradientTape() as tape:
                probs = (model(tf.expand_dims(data[num], axis=0)))
                log_likelyhood = tf.math.log(probs)

            derv = tape.gradient(log_likelyhood, model.weights)
            fisher = [(fis + dv**2) for fis, dv in zip(fisher, derv)]

    fisher = [fish/(samples*length) for fish in fisher]
    return fisher


if __name__ == '__main__':
    model = Lenet5()
    model.summary()
