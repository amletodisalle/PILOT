import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from numpy import savetxt
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import collections
from sklearn.model_selection import StratifiedShuffleSplit
import os
import time

#os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

start_time = time.time()

BASE_DIR = './'
EPOCHS = 8
BATCH_SIZE = 64

def plot_history(loss, accuracy, val_loss, val_accuracy, path):
    x_plot = list(range(1, EPOCHS * 10 + 1))

    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(x_plot, loss)
    plt.plot(x_plot, val_loss)
    plt.legend(['Training', 'Validation'])

    plt.savefig(path + 'FFNN-loss.png')

    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(x_plot, accuracy)
    plt.plot(x_plot, val_accuracy)
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.savefig(path + 'FFNN-accuracy.png')
    plt.show()


def predict_labels(x_test, model):
    predictions = model.predict(x_test)
    predictions_labels = np.argmax(predictions, axis=1)
    """predictions_labels = (predictions > 0.5).astype(np.int8)
    print(np.equal(predictions_labels, np.round(predictions)).all())"""
    return predictions_labels


def plot_cm(predictions, y_test, path):
    print(classification_report(y_test, predictions))

    cm = confusion_matrix(y_test, predictions)
    # plt.figure(figsize=(10,10), dpi = 70)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap=plt.cm.Blues, cbar=False)
    classes = np.unique(y_test)
    ax.set(xlabel="Prediction", ylabel="Real", xticklabels=classes, yticklabels=classes, title="Confusion matrix")

    """ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')"""

    plt.savefig(path + 'FFNN-cm.png')


# load training, validation and testing labels from CSV files
def load_data_and_labels(path):
    raw_data = open(path + '/training_labels.csv', 'rt')
    tr_l = np.loadtxt(raw_data, delimiter=",").astype(int)
    print("Train: ", collections.Counter(tr_l))

    raw_data = open(path + '/validation_labels.csv', 'rt')
    val_l = np.loadtxt(raw_data, delimiter=",").astype(int)
    print("Validation: ", collections.Counter(val_l))

    raw_data = open(path + '/testing_labels.csv', 'rt')
    te_l = np.loadtxt(raw_data, delimiter=",").astype(int)
    print("Test: ", collections.Counter(te_l))

    raw_data = open(path + '/training_data.csv', 'rt')
    tr_d = np.loadtxt(raw_data, delimiter=",")
    print("Train data: ", tr_d.shape)

    raw_data = open(path + '/validation_data.csv', 'rt')
    val_d = np.loadtxt(raw_data, delimiter=",")
    print("Validation data: ", val_d.shape)

    raw_data = open(path + '/testing_data.csv', 'rt')
    te_d = np.loadtxt(raw_data, delimiter=",")
    print("Test data: ", te_d.shape)

    return (tr_d, tr_l, val_d, val_l, te_d, te_l)


def model_fit(train_data, train_labels, validation_data, validation_labels):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(10, activation="relu", input_dim=5073))
    model.add(tf.keras.layers.Dense(5, activation='softmax'))

    opt = tf.keras.optimizers.Adam(learning_rate=0.005)
    model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    model.summary()

    # plot_model(model, to_file=base_dir + 'multiclass/FFNN-results/FFNN-model.png', show_shapes=True, show_layer_names=True)

    history = model.fit(train_data, train_labels,
                        epochs=EPOCHS,
                        validation_data=(validation_data, validation_labels),
                        batch_size=BATCH_SIZE,
                        verbose=2)

    return (model, history)


def run(path):
    print("Load data and labels")
    x_train, y_train, x_val, y_val, x_test, y_test = load_data_and_labels(path)
    # train_sequences, validation_sequences, test_sequences, emb_matrix = load_sequences_and_matrix(path)

    history = []
    predictions = []
    model, history = model_fit(x_train, y_train, x_val, y_val)
    predictions = predict_labels(x_test, model)

    return (history, predictions, y_test)


def runExperiment(root):
    loss = []
    accuracy = []
    val_loss = []
    val_accuracy = []
    predictions = []
    y_test = []

    for i in range(10):
        path = root + 'Round' + str(i + 1)
        print(path)

        history, round_predictions, round_y_test = run(path)

        print(history.history['loss'])

        loss = np.append(loss, history.history['loss'])
        accuracy = np.append(accuracy, history.history['accuracy'])
        val_loss = np.append(val_loss, history.history['val_loss'])
        val_accuracy = np.append(val_accuracy, history.history['val_accuracy'])
        predictions = np.append(predictions, round_predictions)
        y_test = np.append(y_test, round_y_test)

        print("Seconds: ", (time.time() - start_time))
        print("\n")

    return (loss, accuracy, val_loss, val_accuracy, predictions, y_test)


loss, accuracy, val_loss, val_accuracy, predictions, y_test = runExperiment(BASE_DIR + "multiclass/MaldonadoDataset/")

np.savetxt(BASE_DIR + "multiclass/FFNN-results/FFNN-Prediction.csv", predictions.T.astype(int), delimiter=",", fmt="%i")
np.savetxt(BASE_DIR + "multiclass/FFNN-results/FFNN-Truth.csv", y_test.T.astype(int), delimiter=",", fmt="%i")
np.savetxt(BASE_DIR + "multiclass/FFNN-results/FFNN-Loss.csv", loss.T.astype(float), delimiter=",", fmt="%f")
np.savetxt(BASE_DIR + "multiclass/FFNN-results/FFNN-Val_Loss.csv", val_loss.T.astype(float), delimiter=",", fmt="%f")
np.savetxt(BASE_DIR + "multiclass/FFNN-results/FFNN-Acc.csv", accuracy.T.astype(float), delimiter=",", fmt="%f")
np.savetxt(BASE_DIR + "multiclass/FFNN-results/FFNN-Val_Acc.csv", val_accuracy.T.astype(float), delimiter=",", fmt="%f")

print(loss.shape)
print(accuracy.shape)
print(val_loss.shape)
print(val_accuracy.shape)
print(predictions.shape)
print(y_test.shape)

plot_cm(predictions, y_test, BASE_DIR + "multiclass/FFNN-results/")
plot_history(loss, accuracy, val_loss, val_accuracy, BASE_DIR + "multiclass/FFNN-results/")

print("Total time: ", (time.time() - start_time))
