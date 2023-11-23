import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from tensorflow.keras import mixed_precision

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

def residual_block(inputs, kernel_size, num_kernels, num_layers):
    
    x = inputs
    for i in range(num_layers):
        x = layers.Conv1D(num_kernels, kernel_size, padding = 'same')(x) 
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
    inputs = layers.Conv1D(num_kernels, 1)(inputs) 
    
    return x + inputs

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

def positional_enc(seq_len: int, model_dim: int) -> tf.Tensor:
    """
    Computes pre-determined postional encoding as in (Vaswani et al., 2017).
    """
    pos = np.arange(seq_len)[..., None]
    dim = np.arange(model_dim, step=2)

    frequencies = 1.0 / np.power(1000, (dim / model_dim))

    positional_encoding_table = np.zeros((seq_len, model_dim))
    positional_encoding_table[:, 0::2] = np.sin(pos * frequencies)
    positional_encoding_table[:, 1::2] = np.cos(pos * frequencies)

    return tf.cast(positional_encoding_table, tf.float32)

def build_transformer(
    input_shape,
    config
):
    # Unpack dict:

    head_size = config["head_size"]
    num_heads = config["num_heads"]
    ff_dim = config["ff_dim"]
    num_transformer_blocks = config["num_transformer_blocks"]
    mlp_units = config["mlp_units"]
    mlp_dropout = config["mlp_dropout"]
    dropout = config["dropout"]

    inputs = keras.Input(shape=input_shape, name='strain')
    
    model_dim = num_heads * head_size

    # projection to increase the size of the model
    x = layers.Reshape((input_shape[0], 1))(inputs)
    
    x = residual_block(x, 4, int(model_dim/8), 2)
    x = layers.MaxPool1D(4)(x) 
    x = residual_block(x, 8, int(model_dim/4), 2)
    x = layers.MaxPool1D(4)(x) 
    x = residual_block(x, 16, int(model_dim/2), 2)
    x = layers.MaxPool1D(4)(x) 
    x = residual_block(x, 32, int(model_dim), 2)
    x = layers.MaxPool1D(4)(x) 
    
    # positional encoding
    seq_len = x.shape[1]
    positional_encoding = positional_enc(seq_len, model_dim)  # or model_dim=1

    x += positional_encoding[:x.shape[1]]
    x = layers.Dropout(dropout)(x)

    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    
    outputs = layers.Dense(2, activation="relu", name = 'signal_present')(x)
    return keras.Model(inputs, outputs)

def build_cnn(
    input_shape,
    config
):
    
    inputs = keras.Input(shape=input_shape)
    x = layers.Reshape((input_shape[-1], 1))(inputs)
    x = layers.Conv1D(64, 8, activation="relu")(x) 
    x = layers.MaxPool1D(8)(x) 
    x = layers.Conv1D(32, 8, activation="relu")(x) 
    x = layers.Conv1D(32, 16, activation="relu")(x) 
    x = layers.MaxPool1D(6)(x) 
    x = layers.Conv1D(16, 16, activation="relu")(x) 
    x = layers.Conv1D(16, 32, activation="relu")(x) 
    #x = layers.MaxPool1D(4)(x) 
    x = layers.Conv1D(16, 32, activation="relu")(x) 
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x) 
    x = layers.Dropout(0.5)(x) 
    x = layers.Dense(64, activation="relu")(x) 
    x = layers.Dropout(0.5)(x) 
    outputs = layers.Dense(2, activation="softmax")(x) 
        
    return keras.Model(inputs, outputs)

def build_cnn_regression(
    input_shape,
    config
):
    inputs = keras.Input(shape=input_shape)
    x = layers.Reshape((input_shape[-1], 1))(inputs)
    x = layers.MaxPool1D(8)(x) 
    x = layers.Conv1D(64,  8, activation="relu", padding = 'same')(x)
    x = layers.Conv1D(32,  8, activation="relu", padding = 'same')(x)
    x = layers.MaxPool1D(8)(x) 
    x = layers.Conv1D(32, 16, activation="relu", padding = 'same')(x) 
    x = layers.Conv1D(16, 16, activation="relu", padding = 'same')(x) 
    x = layers.Conv1D(16, 32, activation="relu", padding = 'same')(x) 
    x = layers.Conv1D(16, 32, activation="relu", padding = 'same')(x) 
    x = layers.Conv1D(16 , 32, activation="relu", padding = 'same')(x) 
    
    x = layers.Flatten()(x)
    #position_encoding = tf.cast(np.linspace(0, 16, input_shape[-1]), tf.float32)
    #x *= position_encoding
    
    x = layers.Dense(512, activation="relu")(x) 
    x = layers.Dropout(0.2)(x) 
    x = layers.Dense(512, activation="relu")(x) 
    x = layers.Dropout(0.5)(x) 
    x = layers.Dense(512, activation="relu")(x) 
    x = layers.Dropout(0.5)(x) 
    outputs = layers.Dense(2, activation="relu")(x) 
        
    return keras.Model(inputs, outputs)

def one_hot_cnn_regression(
    input_shape,
    config
):
    inputs = keras.Input(shape=input_shape)
    x = layers.Reshape((input_shape[-1], 1))(inputs)
    x = layers.Conv1D(64,  8, activation="relu", padding = 'same')(x)
    x = layers.Conv1D(64,  8, activation="relu", padding = 'same')(x)
    x = layers.Conv1D(64,  8, activation="relu", padding = 'same')(x)
    x = layers.Conv1D(64,  8, activation="relu", padding = 'same')(x)
    x = layers.Conv1D(64, 16, activation="relu", padding = 'same')(x) 
    x = layers.Conv1D(64, 16, activation="relu", padding = 'same')(x) 
    x = layers.Conv1D(64, 16, activation="relu", padding = 'same')(x) 
    x = layers.Conv1D(64, 16, activation="relu", padding = 'same')(x)
    x = layers.Conv1D(64, 32, activation="relu", padding = 'same')(x) 
    x = layers.Conv1D(64, 32, activation="relu", padding = 'same')(x) 
    x = layers.Conv1D(16, 32, activation="relu", padding = 'same')(x) 
    outputs = layers.Conv1D(1 , 32, activation="sigmoid", padding = 'same')(x) 
    
    #outputs = layers.Flatten()(x)
    
    return keras.Model(inputs, outputs)

def multi_hot_np(x, y, arr_len):
    # x will be a numpy array with the contents of the input to the
    # tf.function
    array = np.zeros(arr_len, dtype=np.float32)
    
    minim = np.minimum(y[0],x[0])
    maxim = np.maximum(y[0],x[0])
    
    array[minim:maxim] = 1
    
    return array
@tf.function(input_signature=[tf.TensorSpec(None, tf.int32), tf.TensorSpec(None, tf.int32), tf.TensorSpec(None, tf.int32) ])
def multi_hot(input_1, input_2, input_3):
    return tf.numpy_function(multi_hot_np, [input_1, input_2, input_3], tf.float32)

def timesToOneHot(times, array_len, duration):
    timeA = times[0]
    timeB = times[1]
    
    A_index = tf.squeeze(tf.cast(array_len/duration * timeA, tf.int32))
    B_index = tf.squeeze(tf.cast(array_len/duration * timeB, tf.int32))
        
    return multi_hot(A_index, B_index, array_len)

def getInput(element):
    return (element['H1_strain'], tf.cast(element['overlap_present'], tf.float16))

def getInputRegression(element):
    return (element['H1_strain'], (tf.math.minimum(element['H1_time_signal_a'], element['H1_time_signal_b']) - 50.0, tf.math.maximum(element['H1_time_signal_a'], element['H1_time_signal_b']) - 50.0))

def getInputRegressionOneHot(element):
    return (element['H1_strain'], timesToOneHot((element['H1_time_signal_a'], element['H1_time_signal_b']), 16_384, 16))

@tf.function
def __regresionFilter(dataset):
    return dataset['overlap_present'] == 1

strategy = setup_CUDA(True, "1,2,3")

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
        model_path = "overlapnet_5"

        model =  build_cnn

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
    input_shape = (32768//2,) #get_element_shape(train_dataset)

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