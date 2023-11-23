import tensorflow_datasets as tfds
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

def get_element_shape(dataset):
    for element in dataset:
        return element['H1_strain'].shape[1:]

def setup_CUDA(verbose, device_num):
        
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_num)
        
    gpus =  tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy(gpus)

    physical_devices = tf.config.list_physical_devices('GPU')
    
    for device in physical_devices:    

        try:
            tf.config.experimental.set_memory_growth(device, True)
        except:
            # Invalid device or cannot modify virtual devices once initialized.
            pass
    
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    if verbose:
        tf.config.list_physical_devices("GPU")
        
    return strategy

class PositionalEncoding(Layer):
    def __init__(self, max_len=4096, d_model=128, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        self.pe = self.create_positional_encoding()

    def create_positional_encoding(self):
        position = tf.range(self.max_len, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(tf.range(0, self.d_model, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / self.d_model))

        sin_values = tf.sin(position * div_term)
        cos_values = tf.cos(position * div_term)

        sin_values_expanded = tf.expand_dims(sin_values, 1)
        cos_values_expanded = tf.expand_dims(cos_values, 1)

        pe = tf.concat([sin_values_expanded, cos_values_expanded], axis=1)
        pe = tf.reshape(pe, (1, self.max_len, self.d_model))

        return pe

    def call(self, x):
        return x + self.pe[:, :tf.shape(x)[1], :]

def create_denoising_layers():
    inputs = Input(shape=(16384, 1))
    x = Conv1D(8, 3, padding='same')(inputs)
    x = ReLU()(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(16, 3, padding='same')(x)
    x = ReLU()(x)
    x = MaxPooling1D(2)(x)
    x = Conv1DTranspose(8, 3, strides=2, padding='same')(x)
    x = ReLU()(x)
    x = Conv1DTranspose(1, 3, strides=2, padding='same')(x)
    x = ReLU()(x)
    return Model(inputs=inputs, outputs=x)

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    # Conv1D(filters=ff_dim, kernel_size=1, activation="relu")
    x = layers.Dense(ff_dim, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    # Conv1D((filters=inputs.shape[-1], kernel_size=1))
    x = layers.Dense(inputs.shape[-1])(x)
    x = layers.Dropout(dropout)(x)
    return x + res

def build_starscream(input_shape, config):
    inputs = Input(shape=input_shape)

    denoising_livingston = create_denoising_layers()
    denoising_hanford = create_denoising_layers()

    x_livingston = denoising_livingston(inputs[:, :, 0:1])
    x_hanford = denoising_hanford(inputs[:, :, 1:2])

    feature_extraction = tf.keras.Sequential([
        Conv1D(16, 3, padding='same'),
        ReLU(),
        MaxPooling1D(2),
        Conv1D(32, 3, padding='same'),
        ReLU(),
        MaxPooling1D(2)
    ])

    features_livingston = feature_extraction(x_livingston)
    features_hanford = feature_extraction(x_hanford)

    combined_features = Concatenate(axis=2)([features_livingston, features_hanford])
    combined_features = tf.reshape(combined_features, [tf.shape(inputs)[0], -1, 32 * 1024 * 2])
    fused_features = Dense(512)(combined_features)

    embedding_livingston = Dense(128)(fused_features[:, :1024])
    embedding_hanford = Dense(128)(fused_features[:, 1024:])
    positional_encoding = PositionalEncoding()

    embedded_livingston = positional_encoding(embedding_livingston[:, tf.newaxis, :])
    embedded_hanford = positional_encoding(embedding_hanford[:, tf.newaxis, :])

    num_transformer_blocks = 3
    x_livingston = embedded_livingston
    x_hanford = embedded_hanford

    head_size = 128
    num_heads = 8
    dropout = 0.5
    ff_dim = 128
    for _ in range(num_transformer_blocks - 1):
        x_livingston = transformer_encoder(x_livingston, head_size, num_heads, ff_dim, dropout=dropout)
        x_hanford    = transformer_encoder(x_hanford, head_size, num_heads, ff_dim, dropout=dropout)

    x_cross_attention = MultiHeadAttention(num_heads=8, key_dim=128)(x_livingston, x_hanford)
    x_cross_attention = LayerNormalization()(x_cross_attention)
    x_cross_attention = layers.Dense(ff_dim, activation="relu")(x_cross_attention)
    x_cross_attention = layers.Dropout(dropout)(x_cross_attention)
    x_cross_attention = layers.Dense(x_livingston.shape[-1])(x_cross_attention)
    x_cross_attention = layers.Dropout(dropout)(x_cross_attention)

    x = tf.squeeze(x_cross_attention, axis=1)
    x = GlobalAveragePooling1D()(x)
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

strategy = setup_CUDA(True, "1,2,3,5,6")

training_config = dict(
    learning_rate=1e-4,
    patience=10,
    epochs=200,
    batch_size=32
)

regression = False #If true model traineer runs regression model

# Load Dataset:
dataset = tfds.load(
    "mloverlaps_dataset",
    data_dir = "../MLOverlaps_data/mloverlaps_dataset_multidetector_v2"
)

with strategy.scope():

    model = None
    extract = None
    loss = None
    metrics = None

    if (regression == False):
        model_path = "skywarp_prototype"

        model = build_starscream

        extract = getInput
        loss = "sparse_categorical_crossentropy"
        metrics = ["sparse_categorical_accuracy"]

        model_config = {}

    else:
        model_path = "model_regression"
        
        model = build_cnn_regression

        extract = getInputRegression
        loss = 'mean_absolute_error'

        metrics = ["mean_absolute_error"]

        model_config = dict(
            head_size=16,
            num_heads=8,
            ff_dim=8,
            num_transformer_blocks=8,
            mlp_units=[512],
            mlp_dropout=0.1,
            dropout=0.1
        )

        for key in dataset.keys():
            dataset[key] = dataset[key].filter(__regresionFilter)

    train_dataset      = dataset['train'].batch(batch_size=training_config["batch_size"])
    test_dataset       = dataset['test'].batch(batch_size=training_config["batch_size"])
    validation_dataset = dataset['validate'].batch(batch_size=1)

    # Get Signal Element Shape:
    input_shape = (32768//2, 2) #get_element_shape(train_dataset)

    #plt.plot(getInputRegression(train_dataset.take(1).get_single_element())[1].numpy())
    plt.savefig("label.png")
    plt.figure()

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