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

# os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

start_time = time.time()

BASE_DIR = './'
EPOCHS = 5
BATCH_SIZE = 16


def plot_history(loss, accuracy, val_loss, val_accuracy, path):
    x_plot = list(range(1, EPOCHS * 10 + 1))

    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(x_plot, loss)
    plt.plot(x_plot, val_loss)
    plt.legend(['Training', 'Validation'])

    plt.savefig(path + 'RNN-loss.png')

    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(x_plot, accuracy)
    plt.plot(x_plot, val_accuracy)
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.savefig(path + 'RNN-accuracy.png')
    plt.show()


def predict_labels(x_test, model):
    predictions = model.predict(x_test)
    predictions_labels = np.argmax(predictions, axis=1)
    return predictions_labels


def plot_cm(predictions, y_test, path):
    print(classification_report(y_test, predictions))

    cm = confusion_matrix(y_test, predictions)
    # plt.figure(figsize=(10,10), dpi = 70)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap=plt.cm.Blues, cbar=False)
    classes = np.unique(y_test)
    ax.set(xlabel="Prediction", ylabel="Real", xticklabels=classes, yticklabels=classes, title="Confusion matrix")

    plt.savefig(path + 'RNN-cm.png')


# load training, validation and testing labels from CSV files
def load_labels(path):
    """==============read training labels=============="""
    raw_data = open(path + '/training_labels.csv', 'rt')
    tr_l = np.loadtxt(raw_data, delimiter=",").astype(int)
    print("Train: ", collections.Counter(tr_l))

    """==============read validation data=============="""
    raw_data = open(path + '/validation_labels.csv', 'rt')
    val_l = np.loadtxt(raw_data, delimiter=",").astype(int)
    print("Validation: ", collections.Counter(val_l))

    """==============read testing data=============="""
    raw_data = open(path + '/testing_labels.csv', 'rt')
    te_l = np.loadtxt(raw_data, delimiter=",").astype(int)
    print("Test: ", collections.Counter(te_l))

    return (tr_l, val_l, te_l)


# load training, validation and testing sequences and emb_matrix (word2vec) from CSV files
def load_sequences_and_matrix(path):
    """==============read training sequences=============="""
    raw_data = open(path + '/train_sequences.csv', 'rt')
    tr_s = np.loadtxt(raw_data, delimiter=",")
    print("Train sequences: ", tr_s.shape)

    """==============read validation sequences=============="""
    raw_data = open(path + '/validation_sequences.csv', 'rt')
    val_s = np.loadtxt(raw_data, delimiter=",")
    print("Validation sequences: ", val_s.shape)

    """==============read testing sequences=============="""
    raw_data = open(path + '/test_sequences.csv', 'rt')
    te_s = np.loadtxt(raw_data, delimiter=",")
    print("Test sequences: ", te_s.shape)

    """==============read emb_matrix=============="""
    raw_data = open(path + '/emb_matrix.csv', 'rt')
    emb_matrix = np.loadtxt(raw_data, delimiter=",")
    print("Emb_matrix: ", emb_matrix.shape)

    return (tr_s, val_s, te_s, emb_matrix)


def create_model(train_sequences, emb_matrix):
    print(emb_matrix.shape[1])

    # result-1: 8 epochs, 128 batch_size
    input_ = tf.keras.layers.Input(shape=train_sequences[0, :].shape, )
    x = tf.keras.layers.Embedding(5097, emb_matrix.shape[1], weights=[emb_matrix], trainable=False)(input_)
    # x = layers.Bidirectional(layers.LSTM(64))(x) # LSTM layer
    x = tf.keras.layers.LSTM(64, dropout=0.5)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)

    # x = layers.Dense(64, activation='relu')(x)
    output = tf.keras.layers.Dense(5, activation='softmax')(x)
    model = tf.keras.models.Model(input_, output)
    opt = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9)
    model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    model.summary()

    # plot_model(model, to_file=base_dir, show_shapes=True, show_layer_names=True)
    # plot_model(model, to_file=BASE_DIR + 'multiclass/RNN-results/RNN-model.png', show_shapes=True, show_layer_names=True)

    return model


def model_fit(train_sequences, train_labels, validation_sequences, validation_labels, model):
    history = model.fit(train_sequences, train_labels,
                        epochs=EPOCHS,
                        validation_data=(validation_sequences, validation_labels),
                        batch_size=BATCH_SIZE,
                        verbose=2)
    return history


def change_dataset(x_tr, y_tr, x_v, y_v, x_te, y_te):
    print(type(x_tr))
    for i in range(y_tr.shape[0]):
        if (y_tr[i] == 1):
            seq = x_tr[i, :]
            x_te = np.append(x_te, [seq], axis=0)
            y_te = np.append(y_te, 1)
            break
    y_tr = np.delete(y_tr, i)
    x_tr = np.delete(x_tr, i, axis=0)

    for i in range(y_te.shape[0]):
        if (y_te[i] == 2):
            seq = x_te[i, :]
            x_v = np.append(x_v, [seq], axis=0)
            y_v = np.append(y_v, 2)
            break
    y_te = np.delete(y_te, i)
    x_te = np.delete(x_te, i, axis=0)
    for i in range(y_te.shape[0]):
        if (y_te[i] == 2):
            seq = x_te[i, :]
            x_v = np.append(x_v, [seq], axis=0)
            y_v = np.append(y_v, 2)
            break
    y_te = np.delete(y_te, i)
    x_te = np.delete(x_te, i, axis=0)
    for i in range(y_te.shape[0]):
        if (y_te[i] == 2):
            seq = x_te[i, :]
            x_tr = np.append(x_tr, [seq], axis=0)
            y_tr = np.append(y_tr, 2)
            break
    y_te = np.delete(y_te, i)
    x_te = np.delete(x_te, i, axis=0)

    for i in range(y_te.shape[0]):
        if (y_te[i] == 3):
            seq = x_te[i, :]
            x_tr = np.append(x_tr, [seq], axis=0)
            y_tr = np.append(y_tr, 3)
            break
    y_te = np.delete(y_te, i)
    x_te = np.delete(x_te, i, axis=0)

    for i in range(y_tr.shape[0]):
        if (y_tr[i] == 4):
            seq = x_tr[i, :]
            x_te = np.append(x_te, [seq], axis=0)
            y_te = np.append(y_te, 4)
            break
    y_tr = np.delete(y_tr, i)
    x_tr = np.delete(x_tr, i, axis=0)
    for i in range(y_v.shape[0]):
        if (y_v[i] == 4):
            seq = x_v[i, :]
            x_te = np.append(x_te, [seq], axis=0)
            y_te = np.append(y_te, 4)
            break
    y_v = np.delete(y_v, i)
    x_v = np.delete(x_v, i, axis=0)

    print("POST")
    print("Train sequences: ", x_tr.shape)
    print("Train: ", collections.Counter(y_tr))
    print("Val sequences: ", x_v.shape)
    print("Val: ", collections.Counter(y_v))
    print("Test sequences: ", x_te.shape)
    print("Test: ", collections.Counter(y_te))

    return (x_tr, y_tr, x_v, y_v, x_te, y_te)


def run(path):
    print("Load labels")
    y_train, y_val, y_test = load_labels(path)
    print("Load sequences")
    train_sequences, validation_sequences, test_sequences, emb_matrix = load_sequences_and_matrix(path)

    print("SHAPE", train_sequences.shape)
    x_train, y_train, x_val, y_val, x_test, y_test = change_dataset(train_sequences, y_train, validation_sequences,
                                                                    y_val, test_sequences, y_test)
    # change_dataset(train_sequences, y_train, validation_sequences, y_val, test_sequences, y_test)

    history = []
    predictions = []
    model = create_model(x_train, emb_matrix)  # number of features
    history = model_fit(x_train, y_train, x_val, y_val, model)
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

print(loss.shape)
print(accuracy.shape)
print(val_loss.shape)
print(val_accuracy.shape)
print(predictions.shape)
print(y_test.shape)

plot_cm(predictions, y_test, BASE_DIR + "multiclass/RNN-results/")
plot_history(loss, accuracy, val_loss, val_accuracy, BASE_DIR + "multiclass/RNN-results/")

np.savetxt(BASE_DIR + "multiclass/RNN-results/RNN-Prediction.csv", predictions.T.astype(int), delimiter=",", fmt="%i")
np.savetxt(BASE_DIR + "multiclass/RNN-results/RNN-Truth.csv", y_test.T.astype(int), delimiter=",", fmt="%i")
np.savetxt(BASE_DIR + "multiclass/RNN-results/RNN-Loss.csv", loss.T.astype(float), delimiter=",", fmt="%f")
np.savetxt(BASE_DIR + "multiclass/RNN-results/RNN-Val_Loss.csv", val_loss.T.astype(float), delimiter=",", fmt="%f")
np.savetxt(BASE_DIR + "multiclass/RNN-results/RNN-Acc.csv", accuracy.T.astype(float), delimiter=",", fmt="%f")
np.savetxt(BASE_DIR + "multiclass/RNN-results/RNN-Val_Acc.csv", val_accuracy.T.astype(float), delimiter=",", fmt="%f")

print("Total time: ", (time.time() - start_time))
