import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import os
from tqdm import tqdm

from dataset import Dataset
from model import classification_model, ewc_fisher_matrix
from params import *

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=7),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, write_graph=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0),
]


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, name1, name2) -> None:
        super(CustomCallback).__init__()
        self.task_1_accuracy = []
        self.task_2_accuracy = []
        self.name1 = name1
        self.name2 = name2

    def on_epoch_end(self, epoch, logs=None):
        self.task_1_accuracy.append(logs['accuracy'])
        self.task_2_accuracy.append(logs['val_accuracy'])

    def on_train_end(self, logs=None):
        plt.plot(self.task_1_accuracy, label=self.name1)
        plt.plot(self.task_2_accuracy, label=self.name2)
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()


def plot_result(history, item):
    plt.plot(history.history[item], label=item)
    plt.plot(history.history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()


def ewc_loss_fn(y_true, y_pred, lam=0.5):
    total_loss = tf.keras.losses.binary_crossentropy(
        y_true, y_pred)
    for j in range(len(theta_star)):
        for i in range(len(theta_star[j])):
            diff = tf.reduce_sum(I[j][i]*(tf.square(theta[i]) - 2*tf.multiply(
                theta_star[j][i], theta[i]) + tf.square(theta_star[j][i])))
            total_loss += (lam/2)*diff

    return total_loss


if __name__ == '__main__':
    datas = [Dataset(i).get_data() for i in range(4)]
    star_model = classification_model()
    star_model.summary()
    star_model.compile(loss='binary_crossentropy',
                       optimizer='adam', metrics=['accuracy'])
    # star_model.load_weights('modelA*.h5')
    history = star_model.fit(datas[0], epochs=EPOCHS,
                             validation_data=datas[0], callbacks=callbacks)
    # plot_result(history, 'accuracy')

    theta = star_model.weights
    theta_star = [[tf.constant(i) for i in star_model.get_weights()]]
    I = [ewc_fisher_matrix(datas[0], star_model)]
    star_model.compile(loss=ewc_loss_fn, optimizer='adam',
                       metrics=['accuracy'])
    history = star_model.fit(datas[1], epochs=EPOCHS,
                             validation_data=datas[0], callbacks=callbacks + [tf.keras.callbacks.ModelCheckpoint(
                                 filepath='modelA*.h5', save_weights_only=True, monitor='loss', save_best_only=True), CustomCallback('B', 'A')])
    # plot_result(history, 'accuracy')

    theta = star_model.weights
    theta_star = [[tf.constant(i)
                   for i in star_model.get_weights()]] + theta_star
    I = [ewc_fisher_matrix(datas[1], star_model)] + I
    star_model.compile(loss=ewc_loss_fn, optimizer='adam',
                       metrics=['accuracy'])
    history = star_model.fit(datas[2], epochs=EPOCHS,
                             validation_data=datas[1], callbacks=callbacks + [tf.keras.callbacks.ModelCheckpoint(
                                 filepath='modelA*.h5', save_weights_only=True, monitor='loss', save_best_only=True), CustomCallback('C', 'B')])

    theta = star_model.weights
    theta_star = [[tf.constant(i)
                   for i in star_model.get_weights()]] + theta_star
    I = [ewc_fisher_matrix(datas[2], star_model)] + I
    star_model.compile(loss=ewc_loss_fn, optimizer='adam',
                       metrics=['accuracy'])
    history = star_model.fit(datas[3], epochs=EPOCHS,
                             validation_data=datas[2], callbacks=callbacks + [tf.keras.callbacks.ModelCheckpoint(
                                 filepath='modelA*.h5', save_weights_only=True, monitor='loss', save_best_only=True), CustomCallback('D', 'C')])

    star_model.summary()
    accA = star_model.evaluate(datas[0])[1]
    accB = star_model.evaluate(datas[1])[1]
    accC = star_model.evaluate(datas[2])[1]
    accD = star_model.evaluate(datas[3])[1]

    plt.plot([accA, accB, accC, accD])
    plt.xlabel('Task Number')
    plt.ylabel('Accuracy')
    plt.show()
