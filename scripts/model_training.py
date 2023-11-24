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
    x = layers.Dense(512, activation="relu")(x) 
    x = layers.Dropout(0.2)(x) 
    x = layers.Dense(512, activation="relu")(x) 
    x = layers.Dropout(0.5)(x) 
    x = layers.Dense(512, activation="relu")(x) 
    x = layers.Dropout(0.5)(x) 
    outputs = layers.Dense(2, activation="relu")(x) 
        
    return keras.Model(inputs, outputs)

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

    #train_dataset      = dataset['train'].batch(batch_size=training_config["batch_size"])
    #test_dataset       = dataset['test'].batch(batch_size=training_config["batch_size"])
    #validation_dataset = dataset['validate'].batch(batch_size=1)

    # Get Signal Element Shape:
    input_shape = (32768//2,) #get_element_shape(train_dataset)

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