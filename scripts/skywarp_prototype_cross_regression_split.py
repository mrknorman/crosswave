import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

from tensorflow.keras.layers import (Layer, Input, Add, Conv1D, ReLU, PReLU, MaxPooling1D, GlobalAveragePooling1D, Conv1DTranspose, Concatenate, Dense, MultiHeadAttention, BatchNormalization, LayerNormalization, Dropout, UpSampling1D)
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.models import Model

import numpy as np

from tensorflow import keras
from tensorflow.data import Dataset
from tensorflow.keras import layers

from tensorflow.keras.callbacks import Callback

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
    
class SpinNormRegularisation(Layer):
    def __init__(self, **kwargs):
        super(SpinNormRegularisation, self).__init__(**kwargs)

    def call(self, pe_output):
        latter_half_params_slice = pe_output.shape[1] // 2

        def process_half(half_params):
            norms = [tf.sqrt(tf.reduce_sum(half_params[:, i:i + 3] ** 2, axis=-1, keepdims=True)) for i in range(4, 10, 3)]
            return tf.concat([half_params] + norms, axis=-1)

        signal_a_params = process_half(pe_output[:, :latter_half_params_slice])
        signal_b_params = process_half(pe_output[:, latter_half_params_slice:])

        regularised_pe_output = tf.concat([signal_a_params, signal_b_params], axis=-1)
        return regularised_pe_output
    
class ChirpMassRegularisation(Layer):
    def __init__(self, **kwargs):
        super(ChirpMassRegularisation, self).__init__(**kwargs)

    def call(self, pe_output):
        latter_half_params_slice = pe_output.shape[1] // 2

        def calculateChirpMass(half_params):
            epsilon = 1e-6
            m1 = tf.maximum(tf.nn.relu(half_params[:, 2]), epsilon)
            m2 = tf.maximum(tf.nn.relu(half_params[:, 3]), epsilon)

            numerator = tf.pow(m1 * m2, 3/5)
            denominator = tf.pow(m1 + m2, 1/5)

            chirp_mass = numerator / denominator

            return tf.concat([half_params, tf.expand_dims(chirp_mass, axis=-1)], axis=-1)

        signal_a_params = calculateChirpMass(pe_output[:, :latter_half_params_slice])
        signal_b_params = calculateChirpMass(pe_output[:, latter_half_params_slice:])

        regularised_pe_output = tf.concat([signal_a_params, signal_b_params], axis=-1)
        return regularised_pe_output

class ConditionalOutputLayer(Layer):
    def __init__(self, **kwargs):
        super(ConditionalOutputLayer, self).__init__(**kwargs)
        
    def call(self, inputs):
        pe_output, detection_output = inputs
        latter_half_params_slice = pe_output.shape[1] // 2
        signal_presence = detection_output[:, 1:2]  # Assuming that the second column represents the probability of presence

        signal_b_params = pe_output[:, latter_half_params_slice:]
        signal_b_params *= signal_presence

        conditioned_pe_output = tf.concat([pe_output[:, :latter_half_params_slice], signal_b_params], axis=-1)
        return conditioned_pe_output

# Helper function to create a convolution block
def conv_block(x, filters, transpose=False, kernel_size=8, padding='same', kernel_initializer=HeNormal(), num_layers = 2):
    # Use Conv1DTranspose for the decoder part, Conv1D for the encoder part
    y = x
    for i in range(num_layers - 1):
        if transpose:
            x = Conv1DTranspose(filters, kernel_size, padding=padding, kernel_initializer=kernel_initializer)(x)
        else:
            x = Conv1D(filters, kernel_size, padding=padding, kernel_initializer=kernel_initializer)(x)
        x = BatchNormalization()(x)
        x = PReLU()(x)  # Apply PReLU activation function
    
    if transpose:
        x = Conv1DTranspose(filters, kernel_size, padding=padding, kernel_initializer=kernel_initializer)(x)
    else:
        x = Conv1D(filters, kernel_size, padding=padding, kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization()(x)
    
    # Add 1x1 convolution to y to match the number of filters in x
    y = Conv1D(filters, 1, padding='same', kernel_initializer=kernel_initializer)(y)
    
    x = Add()([x, y])
    x = PReLU()(x)  # Apply PReLU activation function
    
    return x

# Helper function to build the encoder or decoder part of the autoencoder
def build_encoder_decoder(x, filters_list, num_layers_list, pool_or_upsample_list, transpose=False):
    # Iterate through the filters and corresponding pooling/upsampling operations
    for filters, pool_or_upsample, num_layers in zip(filters_list, pool_or_upsample_list, num_layers_list):
        # Add a convolution block (regular or transpose based on the transpose flag)
        x = conv_block(x, filters, transpose=transpose, num_layers = num_layers)
        
        # Apply the pooling/upsampling operation if needed (True in the pool_or_upsample_list)
        if pool_or_upsample:
            if transpose:
                x = UpSampling1D(size=8)(x)  # Upsampling for the decoder part
            else:
                x = MaxPooling1D(pool_size=8, strides=8)(x)  # MaxPooling for the encoder part
    return x

def create_denoising_layers():
    # Encoder part
    inputs = Input(shape=(16384, 1))
    filters_list = [64, 32, 16]  # List of filter sizes for the encoder
    pool_or_upsample_list = [True, True, False]  # List of pooling operations for the encoder
    num_layers_list = [1, 2, 2]
    encoded = build_encoder_decoder(inputs, filters_list, num_layers_list, pool_or_upsample_list, transpose=False)

    # Decoder part
    reversed_filters_list = filters_list[::-1]  # Reverse the filter sizes list for the decoder
    reversed_upsampling_list = pool_or_upsample_list[::-1]  # Reverse the pooling operations list for the decoder
    reversed_num_layers_list = num_layers_list[::-1]  # Reverse the pooling operations list for the decoder

    decoded = build_encoder_decoder(encoded, reversed_filters_list, reversed_num_layers_list, reversed_upsampling_list,  transpose=True)
    
    # Final convolution layer to match the output shape with the input shape
    decoded = Conv1D(filters=1, kernel_size=8, padding='same', activation='linear')(decoded)

    # Create the autoencoder model
    autoencoder = Model(inputs=inputs, outputs=decoded)
    return autoencoder

def transformer_encoder(input_1, input_2, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = LayerNormalization(epsilon=1e-6)(input_1)
    y = LayerNormalization(epsilon=1e-6)(input_2)
    
    x = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, y)
    x = Dropout(dropout)(x)
    res = x + input_1

    # Feed Forward Part
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, kernel_initializer=HeNormal())(x)
    x = PReLU()(x)
    x = Dropout(dropout)(x)
    x = Dense(input_1.shape[-1], kernel_initializer=HeNormal())(x)
    x = Dropout(dropout)(x)
    return x + res

def build_starscream(input_shape, config):
    inputs = Input(shape=input_shape)

    denoising_livingston = create_denoising_layers()
    denoising_hanford = create_denoising_layers()

    x_livingston = denoising_livingston(inputs[:, :, 0:1])
    x_hanford = denoising_hanford(inputs[:, :, 1:2])
    
    feature_extraction = tf.keras.Sequential([
        Conv1D(filters=64, kernel_size=8, padding='same', kernel_initializer=HeNormal()),
        PReLU(),
        MaxPooling1D(pool_size=8, strides=8),
        Conv1D(filters=32, kernel_size=8, padding='same',  kernel_initializer=HeNormal()),
        PReLU(),
        Conv1D(filters=32, kernel_size=8, padding='same', kernel_initializer=HeNormal()),
        PReLU(),
        MaxPooling1D(pool_size=8, strides=8),
        Conv1D(filters=16, kernel_size=8, padding='same', kernel_initializer=HeNormal()),
        PReLU(),
        Conv1D(filters=16, kernel_size=8, padding='same', kernel_initializer=HeNormal()),
        PReLU(),
        Conv1D(filters=128, kernel_size=8, padding='same', kernel_initializer=HeNormal()),
        PReLU()
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
        
    concatenated = Concatenate(axis=-1)([s_livingston, s_hanford])
    
    x = GlobalAveragePooling1D()(concatenated)
    x = Dense(512)(x)
    x = PReLU()(x)
    x = layers.Dropout(dropout)(x)
    pe_output = Dense(20, activation='linear', kernel_initializer=HeNormal())(x)
    detection_output = Dense(2, activation='softmax', kernel_initializer=HeNormal(), name = 'overlap_detected')(x)
    
    pe_output = SpinNormRegularisation()(pe_output)
    pe_output = ChirpMassRegularisation(name = 'estimated_parameters')(pe_output)
    #pe_output = ConditionalOutputLayer(name = 'estimated_parameters')([pe_output, detection_output])

    return Model(inputs=inputs, outputs=[pe_output, detection_output])

def getInput(element):
    return (tf.transpose(tf.stack([element['H1_strain'], element['L1_strain']], axis=1), perm=[0, 2, 1]), tf.cast(element['overlap_present'], tf.float16))

def timeNorm(element):    
    return 0.0 + tf.cast((element != 0.0), np.float32)*((element - 60.0) / 4.0)

def massNorm(element):
    epsilon=1e-6
    return epsilon + tf.cast((element != 0.0), np.float32)*(element / 70.0)

def distanceNorm(element):
    return 0.0 + tf.cast((element != 0.0), np.float32)*((element - 500.0) / 100.0)

def angleNorm(element):
    return 0.0 + tf.cast((element != 0.0), np.float32)*((element) / 2.0*np.pi)

def getInputRegression(element):
    return tf.transpose(tf.stack([element['H1_strain'], element['L1_strain']], axis=1), perm=[0, 2, 1]), (
             tf.transpose(tf.stack([
                #distanceNorm(element['luminosity_distance_signal_a']),
                timeNorm(element['H1_time_signal_a']),
                timeNorm(element['L1_time_signal_a']),
                massNorm(element['mass1_signal_a']),
                massNorm(element['mass2_signal_a']),
                element['spin1x_signal_a'],
                element['spin1y_signal_a'],
                element['spin1z_signal_a'],
                element['spin2x_signal_a'],
                element['spin2y_signal_a'],
                element['spin2z_signal_a'],
                #angleNorm(element['polarization_signal_a']),
                #angleNorm(element['phase_signal_a']),
                #angleNorm(element['inclination_signal_a']),
                element['a2_signal_a'],
                element['a1_signal_a'],
                massNorm(element['chirp_mass_signal_a']),
                #distanceNorm(element['luminosity_distance_signal_b']),
                timeNorm(element['H1_time_signal_b']),
                timeNorm(element['L1_time_signal_b']),
                massNorm(element['mass1_signal_b']),
                massNorm(element['mass2_signal_b']),
                element['spin1x_signal_b'], 
                element['spin1y_signal_b'],
                element['spin1z_signal_b'],
                element['spin2x_signal_b'],
                element['spin2y_signal_b'],
                element['spin2z_signal_b'],
                #angleNorm(element['polarization_signal_b']),
                #angleNorm(element['phase_signal_b']),
                #angleNorm(element['inclination_signal_b']),
                element['a1_signal_b'],
                element['a2_signal_b'],
                massNorm(element['chirp_mass_signal_b']),
        ])), element['overlap_present']
    )

@tf.function
def __regresionFilter(dataset):
    return (dataset['network_SNR_signal_a'] >= 12.0) and ((dataset['network_SNR_signal_b'] >= 12.0) or (dataset['network_SNR_signal_b'] == 0.0))

strategy = setup_CUDA(True, "1")

training_config = dict(
    learning_rate=0.0001,
    patience=20,
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
    extract = getInputRegression
    loss = None
    metrics = None

    model_path = "skywarp_prototype_regression"

    model = build_starscream
                 
    loss = {
        'estimated_parameters': tf.keras.losses.Huber(delta=0.1),
        'overlap_detected': 'sparse_categorical_crossentropy'
    }
    metrics = {
        'estimated_parameters': ['mse', 'mae'],
        'overlap_detected': ['accuracy']
    }
    loss_weights = {
        'estimated_parameters': 1.0,
        'overlap_detected': 1.0
    }

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
    input_shape = (32768//2, 2) #get_element_shape(train_dataset
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
        loss_weights=loss_weights
    )
    model.summary()
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=training_config["patience"],
            restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            model_path,
            monitor="val_loss",
            save_best_only=True,
            save_freq="epoch", 
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
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