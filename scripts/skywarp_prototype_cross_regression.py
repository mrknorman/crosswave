import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

from tensorflow.keras.layers import (Layer, Input, Conv1D, ReLU, MaxPooling1D, GlobalAveragePooling1D, Conv1DTranspose, Concatenate, Dense, MultiHeadAttention, LayerNormalization, UpSampling1D)

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

        return tf.cast(pe, dtype=tf.float16)

    def call(self, x):
        return x + self.pe[:, :tf.shape(x)[1], :]

    
def create_denoising_layers():
    # Encoder part
    inputs = Input(shape=(16384, 1))
    x = Conv1D(filters=64, kernel_size=8, padding='same', activation='relu')(inputs)
    x = MaxPooling1D(pool_size=8, strides=8)(x)
    x = Conv1D(filters=32, kernel_size=8, padding='same', activation='relu')(x)
    x = Conv1D(filters=32, kernel_size=8, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=8, strides=8)(x)
    x = Conv1D(filters=16, kernel_size=8, padding='same', activation='relu')(x)
    x = Conv1D(filters=16, kernel_size=8, padding='same', activation='relu')(x)
    encoded = Conv1D(filters=16, kernel_size=8, padding='same', activation='relu')(x)

    # Decoder part
    x = Conv1D(filters=16, kernel_size=8, padding='same', activation='relu')(encoded)
    x = Conv1D(filters=16, kernel_size=8, padding='same', activation='relu')(x)
    x = UpSampling1D(size=8)(x)
    x = Conv1D(filters=32, kernel_size=8, padding='same', activation='relu')(x)
    x = Conv1D(filters=32, kernel_size=8, padding='same', activation='relu')(x)
    x = UpSampling1D(size=8)(x)
    x = Conv1D(filters=64, kernel_size=8, padding='same', activation='relu')(x)
    decoded = Conv1D(filters=1, kernel_size=8, padding='same', activation='relu')(x)

    autoencoder = Model(inputs=inputs, outputs=decoded)
    return autoencoder

def transformer_encoder(input_1, input_2, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(input_1)
    y = layers.LayerNormalization(epsilon=1e-6)(input_2)
    
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, y)
    x = layers.Dropout(dropout)(x)
    res = x + input_1

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    # Conv1D(filters=ff_dim, kernel_size=1, activation="relu")
    x = layers.Dense(ff_dim, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    # Conv1D((filters=inputs.shape[-1], kernel_size=1))
    x = layers.Dense(input_1.shape[-1])(x)
    x = layers.Dropout(dropout)(x)
    return x + res

def build_starscream(input_shape, config):
    inputs = Input(shape=input_shape)

    denoising_livingston = create_denoising_layers()
    denoising_hanford = create_denoising_layers()

    x_livingston = denoising_livingston(inputs[:, :, 0:1])
    x_hanford = denoising_hanford(inputs[:, :, 1:2])
    
    feature_extraction = tf.keras.Sequential([
        Conv1D(filters=64, kernel_size=8, padding='same', activation='relu'),
        MaxPooling1D(pool_size=8, strides=8),
        Conv1D(filters=32, kernel_size=8, padding='same', activation='relu'),
        Conv1D(filters=32, kernel_size=8, padding='same', activation='relu'),
        MaxPooling1D(pool_size=8, strides=8),
        Conv1D(filters=16, kernel_size=8, padding='same', activation='relu'),
        Conv1D(filters=16, kernel_size=8, padding='same', activation='relu'),
        Conv1D(filters=128, kernel_size=8, padding='same', activation='relu')
    ])

    features_livingston = feature_extraction(x_livingston)
    features_hanford = feature_extraction(x_hanford)

    positional_encoding = PositionalEncoding()

    embedded_livingston = positional_encoding(features_livingston)
    embedded_hanford = positional_encoding(features_hanford)

    num_transformer_blocks = 2
    
    s_livingston = embedded_livingston 
    s_hanford = embedded_hanford

    head_size = 128
    num_heads = 8
    dropout = 0.3
    ff_dim = 128
    for _ in range(num_transformer_blocks):
        
        s_livingston = transformer_encoder(s_livingston, s_livingston, head_size, num_heads, ff_dim, dropout=dropout)
        s_hanford    = transformer_encoder(s_hanford, s_hanford, head_size, num_heads, ff_dim, dropout=dropout)
        
        x_livingston = transformer_encoder(s_livingston, s_hanford, head_size, num_heads, ff_dim, dropout=dropout)
        x_hanford    = transformer_encoder(s_hanford, s_livingston, head_size, num_heads, ff_dim, dropout=dropout)
        
        s_livingston += x_livingston
        s_hanford    += x_hanford
        
    concatenated = Concatenate(axis=2)([s_livingston, s_hanford])
    
    x = GlobalAveragePooling1D()(concatenated)
    x = Dense(512, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    output = Dense(28, activation='linear')(x)
    #output = tf.math.minimum(x, tf.ones((32, 29), dtype=np.float16))

    return Model(inputs=inputs, outputs=output)

def getInput(element):
    return (tf.transpose(tf.stack([element['H1_strain'], element['L1_strain']], axis=1), perm=[0, 2, 1]), tf.cast(element['overlap_present'], tf.float16))

def timeNorm(element):    
    return 0.0 + tf.cast((element != 0.0), np.float32)*((element - 60.0) / 4.0)

def massNorm(element):
    return 0.0 + tf.cast((element != 0.0), np.float32)*(element / 100.0)

def distanceNorm(element):
    return 0.0 + tf.cast((element != 0.0), np.float32)*(element / 1000.0)

def spinNorm(element):
    return 0.0 + tf.cast((element != 0.0), np.float32)*((element + 1.0) / 2.0)


def getInputRegression(element):
    return tf.transpose(tf.stack([element['H1_strain'], element['L1_strain']], axis=1), perm=[0, 2, 1]), tf.transpose(tf.stack([
             massNorm(element['mass1_signal_a']),
             massNorm(element['mass2_signal_a']),
             massNorm(element['mass1_signal_b']),
             massNorm(element['mass2_signal_b']),
             spinNorm(element['a1_signal_a']),
             spinNorm(element['spin1x_signal_a']),
             spinNorm(element['spin1y_signal_a']),
             spinNorm(element['spin1z_signal_a']),
             spinNorm(element['a2_signal_a']),
             spinNorm(element['spin2x_signal_a']),
             spinNorm(element['spin2y_signal_a']),
             spinNorm(element['spin2z_signal_a']),
             spinNorm(element['a1_signal_b']),
             spinNorm(element['spin1x_signal_b']), 
             spinNorm(element['spin1y_signal_b']),
             spinNorm(element['spin1z_signal_b']),
             spinNorm(element['a2_signal_b']),
             spinNorm(element['spin2x_signal_b']),
             spinNorm(element['spin2y_signal_b']),
             spinNorm(element['spin2z_signal_b']),
             distanceNorm(element['luminosity_distance_signal_a']),
             distanceNorm(element['luminosity_distance_signal_b']),
             timeNorm(element['H1_time_signal_a']),
             timeNorm(element['H1_time_signal_b']),
             timeNorm(element['L1_time_signal_a']),
             timeNorm(element['L1_time_signal_b']),
             timeNorm(element['geocent_time_signal_a']),
             timeNorm(element['geocent_time_signal_b'])
        ]
    ))

def getInputRegressionWorks(element):
    return tf.transpose(tf.stack([element['H1_strain'], element['L1_strain']], axis=1), perm=[0, 2, 1]), tf.transpose(tf.stack([
             massNorm(element['mass1_signal_a']),
             massNorm(element['mass2_signal_a']),
             massNorm(element['mass1_signal_b']),
             massNorm(element['mass2_signal_b']),
             timeNorm(element['H1_time_signal_a']),
             timeNorm(element['H1_time_signal_b']),
             timeNorm(element['L1_time_signal_a']),
             timeNorm(element['L1_time_signal_b']),
             timeNorm(element['geocent_time_signal_a']),
             timeNorm(element['geocent_time_signal_b'])
        ]
    ))


def getInputRegressionOneHot(element):
    return (element['H1_strain'], timesToOneHot((element['H1_time_signal_a'], element['H1_time_signal_b']), 16_384, 16))

@tf.function
def __regresionFilter(dataset):
    return (dataset['network_SNR_signal_a'] >= 12.0) and (dataset['network_SNR_signal_b'] >= 12.0)
    #return (dataset['network_SNR_signal_a'] >= 12.0) and ((dataset['network_SNR_signal_b'] >= 12.0) or (dataset['network_SNR_signal_b'] == 0.0))

strategy = setup_CUDA(True, "1")

training_config = dict(
    learning_rate=0.0001,
    patience=10,
    epochs=200,
    batch_size=32
)

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

    model_path = "skywarp_prototype_regression"

    model = build_starscream

    extract = getInputRegression
    loss = "mean_squared_error"
    metrics = ["mean_absolute_error"]

    model_config = {}

    for key in dataset.keys():
        dataset[key] = dataset[key].filter(__regresionFilter)

    model_config = dict(
        head_size=16,
        num_heads=8,
        ff_dim=8,
        num_transformer_blocks=8,
        mlp_units=[512],
        mlp_dropout=0.1,
        dropout=0.1
    )

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
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            mode="auto",
            min_delta=0.0001,
            cooldown=4,
            min_lr=0,
        )
    ]
    
    history = model.fit(
        train_dataset.map(extract, num_parallel_calls=tf.data.AUTOTUNE),
        validation_data=test_dataset.map(extract, num_parallel_calls=tf.data.AUTOTUNE),
        epochs=training_config["epochs"],
        batch_size=training_config["batch_size"],
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