import os
import sys

from pathlib import Path


import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import mixed_precision
from tensorflow.keras.layers import (Layer, Input, Conv1D, ReLU, MaxPooling1D, GlobalAveragePooling1D, Conv1DTranspose, Concatenate, Dense, MultiHeadAttention, LayerNormalization)

from tensorflow.keras.models import Model

import numpy as np

from tensorflow import keras
from tensorflow.data import Dataset
from tensorflow.keras import layers

# Get the directory of your current script
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent.parent
sys.path.append(str(parent_dir))

import gravyflow as gf
from layers import PositionalEncoding, build_denoising_autoencoder, AttentionBlock, build_feature_extractor

def build_crosswave(input_shape, config):
    inputs = Input(shape=input_shape)

    denoising_livingston = build_denoising_autoencoder()
    denoising_hanford = build_denoising_autoencoder()
    feature_extraction = build_feature_extractor()


    x_livingston = denoising_livingston(inputs[:, :, 0:1])
    features_livingston = feature_extraction(x_livingston)
    embedded_livingston = Conv1D(filters=128, kernel_size=1, padding='valid', activation='relu')(features_livingston)
    embedded_livingston = PositionalEncoding()(embedded_livingston)

    x_hanford = denoising_hanford(inputs[:, :, 1:2])
    features_hanford = feature_extraction(x_hanford)
    embedded_hanford = Conv1D(filters=128, kernel_size=1, padding='valid', activation='relu')(features_hanford)
    embedded_hanford = PositionalEncoding()(embedded_hanford)

    x_livingston = embedded_livingston
    x_hanford = embedded_hanford

    num_attention_blocks = 3
    head_size = 128
    num_heads = 8
    dropout = 0.5
    num_dense_neurons = 128

    for _ in range(num_attention_blocks - 1):
        x_livingston = AttentionBlock(head_size, num_heads, num_dense_neurons, dropout=dropout)(x_livingston)
        x_hanford = AttentionBlock(head_size, num_heads, num_dense_neurons, dropout=dropout)(x_hanford)

    x_cross_attention = AttentionBlock(head_size, num_heads, num_dense_neurons, dropout=dropout)([x_livingston, x_hanford])
    
    x = GlobalAveragePooling1D()(x_cross_attention)
    x = Dense(512, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    output = Dense(2, activation='softmax')(x)

    return Model(inputs=inputs, outputs=output)

def getInput(element):
    return (tf.transpose(tf.stack([element['H1_strain'], element['L1_strain']], axis=1), perm=[0, 2, 1]), tf.cast(element['overlap_present'], tf.float16))

def getInputRegression(element):
    return (element['H1_strain'], (tf.math.minimum(element['H1_time_signal_a'], element['H1_time_signal_b']) - 50.0, tf.math.maximum(element['H1_time_signal_a'], element['H1_time_signal_b']) - 50.0))

def getInputRegressionOneHot(element):
    return (element['H1_strain'], timesToOneHot((element['H1_time_signal_a'], element['H1_time_signal_b']), 16_384, 16))

@tf.function
def __regresionFilter(dataset):
    return dataset['overlap_present'] == 1

if __name__ == "__main__":

    training_config = dict(
        learning_rate=1e-4,
        patience=10,
        epochs=200,
        batch_size=32
    )
    
    # Load Dataset:
    with gf.env():

        model = None
        extract = None
        loss = None
        metrics = None

        model_path = "./models/chapter_07_crosswave_detection/self_attention"

        model = build_crosswave

        extract = getInput
        loss = "sparse_categorical_crossentropy"
        metrics = ["sparse_categorical_accuracy"]

        model_config = {}

        #train_dataset      = dataset['train'].batch(batch_size=training_config["batch_size"])
        #test_dataset       = dataset['test'].batch(batch_size=training_config["batch_size"])
        #validation_dataset = dataset['validate'].batch(batch_size=1)

        # Get Signal Element Shape:
        input_shape = (32768//2, 2)

        model = model(
            input_shape,
            model_config
        )

        model.compile(
            loss=loss,
            optimizer=keras.optimizers.Adam(
                learning_rate=training_config["learning_rate"]),
            metrics=metrics,
        )
        model.summary()

        quit()

        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=training_config["patience"],
                restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(
                model_path,
                monitor="val_loss",
                save_best_only=True,
                save_freq="epoch", 
            )
        ]

        history = model.fit(
            train_dataset.map(extract, num_parallel_calls=tf.data.AUTOTUNE),
            validation_data=test_dataset.map(extract, num_parallel_calls=tf.data.AUTOTUNE),
            epochs=training_config["epochs"],
            batch_size=training_config["batch_size"],
            # verbose=2,
            callbacks=callbacks
        )

        model.save(model_path)

        plt.figure()
        plt.plot(history.history[metrics[0]])
        plt.plot(history.history[f'val_{metrics[0]}'])
        plt.title(f'model {metrics[0]}')
        plt.ylabel(metrics[0])
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(f"{metrics[0]}_history")

        plt.figure()
        plt.plot(history.history[loss])
        plt.plot(history.history[f'val_{loss}'])
        plt.title(f'{model_path} {loss}')
        plt.ylabel(loss)
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(f"{loss}_history")

        model.evaluate(validation_dataset.map(extract, num_parallel_calls=tf.data.AUTOTUNE), verbose=1)