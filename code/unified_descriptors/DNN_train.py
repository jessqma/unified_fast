import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn import preprocessing




#directory setting

data_dir = '../../data'
processed_data_dir = os.path.join(data_dir, 'processed_data_training')


# data loading
feature_all = np.load(os.path.join(processed_data_dir, 'binary_under_all7.npy'))
CH = np.load(os.path.join(processed_data_dir, 'CH_descriptor_7.npy'))
GE = np.load(os.path.join(processed_data_dir, 'GE_descriptor_7.npy'))
GT = np.load(os.path.join(processed_data_dir, 'GT_descriptor_7.npy'))

GE_GT = np.load(os.path.join(processed_data_dir, 'GE_GT_descriptor_7.npy'))
CH_GE = np.load(os.path.join(processed_data_dir, 'GE_CH_descriptor_7.npy'))
CH_GT = np.load(os.path.join(processed_data_dir, 'GT_CH_descriptor_7.npy'))


# train and test data set
X_all = feature_all[:,:64]
y_all = feature_all[:,64]
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_all, y_all, test_size=0.2, random_state=10)

X_CH = CH[:,:34]
y_CH = CH[:,34]
X_train_CH, X_test_CH, y_train_CH, y_test_CH = train_test_split(X_CH, y_CH, test_size=0.2, random_state=10)

X_GE = GE[:,:14]
y_GE = GE[:,14]
X_train_GE, X_test_GE, y_train_GE, y_test_GE = train_test_split(X_GE, y_GE, test_size=0.2, random_state=10)

X_GT = GT[:,:16]
y_GT = GT[:,16]
X_train_GT, X_test_GT, y_train_GT, y_test_GT = train_test_split(X_GT, y_GT, test_size=0.2, random_state=10)

X_GE_GT = GE_GT[:,:30]
y_GE_GT = GE_GT[:,30]
X_train_GE_GT, X_test_GE_GT, y_train_GE_GT, y_test_GE_GT = train_test_split(X_GE_GT, y_GE_GT, test_size=0.2, random_state=10)

X_CH_GE = CH_GE[:,:48]
y_CH_GE = CH_GE[:,48]
X_train_CH_GE, X_test_CH_GE, y_train_CH_GE, y_test_CH_GE = train_test_split(X_CH_GE, y_CH_GE, test_size=0.2, random_state=10)

X_CH_GT = CH_GT[:,:50]
y_CH_GT = CH_GT[:,50]
X_train_CH_GT, X_test_CH_GT, y_train_CH_GT, y_test_CH_GT = train_test_split(X_CH_GT, y_CH_GT, test_size=0.2, random_state=10)


def DNN_roc_auc_cross_val(X_, y, k_fold, filepath):
    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X_)

    #     cv = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=0)
    cv = KFold(n_splits=k_fold, shuffle=True, random_state=0)

    fp = []
    tp = []
    th = []
    roc_auc = []
    acc = []
    f1 = []
    HISTORY = []
    re_model = []

    for ii, (train, test) in tqdm(enumerate(cv.split(X, y))):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]

        model = keras.Sequential([
            layers.Input(X_train.shape[1:]),
            layers.Flatten(),
            layers.BatchNormalization(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(2, activation='softmax'),
        ])
        optimizer = tf.keras.optimizers.Adam()
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=optimizer,
                      metrics=['accuracy'])
        filepath_ = filepath + '%d.h5' % ii
        checkpoint = ModelCheckpoint(filepath_, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=128, epochs=1000,
                            shuffle=True, verbose=1, callbacks=callbacks_list)
        reconstructed_model = keras.models.load_model("best_model")
        y_pred = reconstructed_model.predict_classes(X_test)
        y_score0 = reconstructed_model.predict_proba(X_test)[:, 0]
        fp_rate, tp_rate, threshold = roc_curve(y_test, y_score0, pos_label=0)
        roc_auc_score_rate = 1 - roc_auc_score(y_test, y_score0)
        accuracy = (X_test.shape[0] - (y_test != y_pred).sum()) / X_test.shape[0]
        # print(X_test.shape[0])
        # print(accuracy)

        fp.append(fp_rate)
        tp.append(tp_rate)
        th.append(threshold)
        roc_auc.append(roc_auc_score_rate)
        acc.append(accuracy)
        f1.append(f1_score(y_test, y_pred))
        HISTORY.append(history)
        re_model.append(reconstructed_model)

    return fp, tp, th, roc_auc, acc, f1, HISTORY, re_model


def fpr_interp(fp, tp):
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    for ii in range(len(fp)):
        interp_tpr = np.interp(mean_fpr, fp[ii], tp[ii])
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
    mean_tpr = np.mean(tprs, axis=0)
    #     mean_tpr = np.max(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    return mean_fpr, mean_tpr, std_tpr, tprs_upper, tprs_lower
