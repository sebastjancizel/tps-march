import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics, preprocessing
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
from tensorflow.keras import util


def create_model(data, catcols, contcols):
    inputs = []
    outputs = []

    for c in catcols:
        num_unique_values = int(data[c].nunique())
        embed_dim = int(min(np.ceil(num_unique_values / 2), 50))

        inp = layers.Input(shape=(1,))
        out = layers.Embedding(num_unique_values + 10, embed_dim, name=c)(inp)  # huh?

        out = layers.SpatialDropout1D(0.3)(out)
        inputs.append(inp)
        outputs.append(out)

    numerical_input = layers.Input(shape=(len(contcols),), name="numeric_input")
    inputs.append(numerical_input)

    numerical = layers.Dense(512, activation="relu")(numerical_input)
    numerical = layers.Dropout(0.2)(numerical)
    numerical = layers.Dense(512, activation="relu")(numerical)
    numerical = layers.Dropout(0.2)(numerical)
    numerical = layers.Dense(512, activation="relu")(numerical)
    numerical = layers.Reshape((1, -1))(numerical)
    outputs.append(numerical)

    x = layers.Concatenate()(outputs)

    x = layers.BatchNormalization()(x)

    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)

    y = layers.Dense(2, activation="softmax")(x)
    y = layers.Reshape((-1,))(y)
    model = Model(inputs=inputs, outputs=y)

    model.compile(loss="binary_crossentropy", optimizer="adam")

    return model


def run(fold):
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    model = create_model(df, catcols, contcols)

    features = catcols + contcols

    xtrain = [df_train[features].values[:, k] for k in range(len(catcols))]
    xvalid = [df_valid[features].values[:, k] for k in range(len(catcols))]

    xtrain.append(df_train[contcols].values)
    xvalid.append(df_valid[contcols].values)

    ytrain = df_train.target.values.reshape((-1, 1))
    yvalid = df_valid.target.values.reshape((-1, 1))

    ytrain_cat = utils.to_categorical(ytrain)
    yvalid_cat = utils.to_categorical(yvalid)

    model.fit(
        xtrain,
        ytrain_cat,
        validation_data=(xvalid, yvalid_cat),
        verbose=1,
        batch_size=512,
        epochs=10,
    )
    valid_preds = model.predict(xvalid)[:, 1]

    print(metrics.roc_auc_score(yvalid, valid_preds))
    K.clear_session()
