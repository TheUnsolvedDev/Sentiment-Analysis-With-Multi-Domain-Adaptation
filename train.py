import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import Data
from model import Lenet5, DNN, ewc_fisher_matrix


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


callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=7),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='model.h5', save_weights_only=True, monitor='accuracy', save_best_only=True),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, write_graph=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='accuracy', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0),
]

ewc_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='modelB.h5', save_weights_only=True, monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, write_graph=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0),
]


def train(train_gen, epochs=100, model=None):
    loss = tf.keras.losses.BinaryCrossentropy()
    model.compile(loss=loss, optimizer='Adam', metrics=['accuracy'])
    history = model.fit(train_gen, epochs=epochs, callbacks=callbacks)
    return model


def ewc_loss_fn(y_true, y_pred, lam=25):
    total_loss = tf.keras.losses.binary_crossentropy(
        y_true, y_pred)
    for j in range(len(theta_star)):
        for i in range(len(theta_star[j])):
            diff = tf.reduce_sum(I[j][i]*(tf.square(theta[i]) - 2*tf.multiply(
                theta_star[j][i], theta[i]) + tf.square(theta_star[j][i])))
            total_loss += (lam/2)*diff

    return total_loss


def train_ewc(train_gen, val_gen, epochs=100, model=None):
    model.compile(loss=ewc_loss_fn, optimizer='Adam', metrics=['accuracy'])
    history = model.fit(train_gen, validation_data=(
        val_gen), epochs=epochs, callbacks=ewc_callbacks + [CustomCallback('A', 'B')])
    return model


if __name__ == '__main__':
    tasks = Data()
    task_1_gen = tasks.task(1)
    task_2_gen = tasks.task(2)
    task_3_gen = tasks.task(3)
    task_4_gen = tasks.task(4)

    star = DNN()
    star = train(task_1_gen, model=star, epochs=100)
    # star.load_weights('model.h5')

    I = [ewc_fisher_matrix([task_1_gen], star)]
    theta = star.weights
    theta_star = [[tf.constant(i) for i in star.get_weights()]]
    star = train_ewc(task_2_gen, task_1_gen, model=star)

    I = [ewc_fisher_matrix([task_1_gen], star)] + I
    theta = star.weights
    theta_star = [[tf.constant(i) for i in star.get_weights()]] + theta_star
    star = train_ewc(task_3_gen, task_2_gen, model=star)

    star.evaluate(task_1_gen)
    star.evaluate(task_2_gen)
    star.evaluate(task_3_gen)