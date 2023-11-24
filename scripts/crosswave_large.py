from pathlib import Path
import sys

import tensorflow as tf
import matplotlib.pyplot as plt
import os

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
from layers import PositionalEncoding, create_denoising_layers, AttentionBlock

def build_crosswave(input_shape, config):
    inputs = Input(shape=input_shape)

    denoising_livingston = create_denoising_layers()
    denoising_hanford = create_denoising_layers()

    x_livingston = denoising_livingston(inputs[:, :, 0:1])
    x_hanford = denoising_hanford(inputs[:, :, 1:2])
    
    feature_extraction = tf.keras.Sequential([
        Conv1D(filters=64, kernel_size=8, padding='valid', activation='relu'),
        MaxPooling1D(pool_size=8, strides=8),
        Conv1D(filters=32, kernel_size=8, padding='valid', activation='relu'),
        Conv1D(filters=32, kernel_size=8, padding='valid', activation='relu'),
        MaxPooling1D(pool_size=8, strides=8),
        Conv1D(filters=16, kernel_size=8, padding='valid', activation='relu'),
        Conv1D(filters=16, kernel_size=8, padding='valid', activation='relu'),
        Conv1D(filters=128, kernel_size=8, padding='valid', activation='relu')
    ])

    features_livingston = feature_extraction(x_livingston)
    features_hanford = feature_extraction(x_hanford)

    embedded_livingston = PositionalEncoding()(features_livingston)
    embedded_hanford = PositionalEncoding()(features_hanford)
    
    s_livingston = embedded_livingston 
    s_hanford = embedded_hanford

    num_transformer_blocks = 3
    head_size = 128
    num_heads = 8
    dropout = 0.5
    ff_dim = 128

    for _ in range(num_transformer_blocks):
        
        s_livingston = AttentionBlock(head_size, num_heads, ff_dim, dropout=dropout)(s_livingston)
        s_hanford    = AttentionBlock(head_size, num_heads, ff_dim, dropout=dropout)(s_hanford)
        
        x_livingston = AttentionBlock(head_size, num_heads, ff_dim, dropout=dropout)([s_livingston, s_hanford])
        x_hanford    = AttentionBlock(head_size, num_heads, ff_dim, dropout=dropout)([s_hanford, s_livingston])
        
        s_livingston += x_livingston
        s_hanford    += x_hanford
        
    mult = Concatenate(axis=2)([s_livingston, s_hanford])
    
    x = GlobalAveragePooling1D()(mult)
    x = Dense(512, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    output = Dense(2, activation='softmax')(x)

    return Model(inputs=inputs, outputs=output)

def getInput(element):
    return (tf.transpose(tf.stack([element['H1_strain'], element['L1_strain']], axis=1), perm=[0, 2, 1]), tf.cast(element['overlap_present'], tf.float16))

training_config = dict(
    learning_rate=1e-4,
    patience=10,
    epochs=200,
    batch_size=32
)

with gf.env():

    model = None
    extract = None
    loss = None
    metrics = None

    model_path = "crosswave_cross"

    model = build_crosswave

    extract = getInput
    loss = "sparse_categorical_crossentropy"
    metrics = ["sparse_categorical_accuracy"]

    model_config = {}

    #train_dataset      = dataset['train'].batch(batch_size=training_config["batch_size"])
    #test_dataset       = dataset['test'].batch(batch_size=training_config["batch_size"])
    #validation_dataset = dataset['validate'].batch(batch_size=1)

    # Get Signal Element Shape:
    input_shape = (32768//2, 2) #get_element_shape(train_dataset)

    plt.savefig("label.png")
    plt.figure()

    model = model(
        input_shape,
        model_config
    )

    model.compile(
        loss=loss,
        optimizer=
            keras.optimizers.Adadelta(),
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
        verbose=2,
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