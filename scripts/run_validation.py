import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from tensorflow.keras import mixed_precision
import pandas as pd

import numpy as np

from tensorflow import keras
from tensorflow.data import Dataset
from tensorflow.keras import layers

def toNumpy(element):
    return element.numpy()[0]

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

def getInput(element):
    return (element['H1_strain'], tf.cast(element['overlap_present'], tf.float16))

def getInputRegression_old(element):
    return (element['H1_strain'], (tf.math.minimum(element['H1_time_signal_a'], element['H1_time_signal_b']) - 50.0, tf.math.maximum(element['H1_time_signal_a'], element['H1_time_signal_b']) - 50.0))

def multi_hot_np(x, y, arr_len):
    # x will be a numpy array with the contents of the input to the
    # tf.function
    array = np.zeros(arr_len, dtype=np.float32)
    
    minim = np.minimum(y,x)
    maxim = np.maximum(y,x)
    
    array[minim:maxim] = 1
    
    return array

@tf.function
def __regresionFilter(dataset):
    return dataset['overlap_present'] == 1

@tf.function(input_signature=[tf.TensorSpec(None, tf.int32), tf.TensorSpec(None, tf.int32), tf.TensorSpec(None, tf.int32) ])
def multi_hot(input_1, input_2, input_3):
    return tf.numpy_function(multi_hot_np, [input_1, input_2, input_3], tf.float32)

def timesToOneHot(times, array_len, duration):
    timeA = times[0]
    timeB = times[1]
    
    A_index = tf.squeeze(tf.cast(array_len/duration * timeA, tf.int32))
    B_index = tf.squeeze(tf.cast(array_len/duration * timeB, tf.int32))
        
    return multi_hot(A_index, B_index, array_len)

def getInputRegression(element):
    return (element['H1_strain'], timesToOneHot((element['H1_time_signal_a'], element['H1_time_signal_b']), 16_384, 16))

setup_CUDA(1, "0")

# Load Dataset:
dataset = tfds.load(
    "mloverlaps_dataset",
    data_dir = "../MLOverlaps_data/mloverlaps_dataset_multidetector_v2"
)

regression = False

if regression == False:
    model_path = "overlapnet_5"
    model = tf.keras.models.load_model(model_path)
    
    validation_dataset = dataset['validate'].batch(batch_size=1)
    feature_names = list(validation_dataset.element_spec.keys())
    feature_names.remove('H1_strain')
    feature_names.remove('L1_strain')
    
    results_dataframe = pd.DataFrame(validation_dataset, columns=feature_names).applymap(toNumpy)
    
    results = model.predict(validation_dataset.map(getInput, num_parallel_calls=tf.data.AUTOTUNE))
    results_dataframe['model_prediction'] = results[:,1]
    
    results_dataframe.to_csv('validation_scores_classification.csv')
    results_dataframe.to_pickle('validation_scores_classification.pkl')
else:
    model_path = "model_regression"
    model = tf.keras.models.load_model(model_path)
    
    validation_dataset = dataset['validate'].filter(__regresionFilter).batch(batch_size=1)

    feature_names = list(validation_dataset.element_spec.keys())
    feature_names.remove('H1_strain')
    feature_names.remove('L1_strain')
    
    results_dataframe = pd.DataFrame(validation_dataset, columns=feature_names).applymap(toNumpy)
    
    results_dataframe['H1_time_signal_a_'] = (results_dataframe['H1_time_signal_a'] < results_dataframe['H1_time_signal_b'])*results_dataframe['H1_time_signal_a'] + (results_dataframe['H1_time_signal_b'] <= results_dataframe['H1_time_signal_a'])*results_dataframe['H1_time_signal_b'] - 50
    results_dataframe['H1_time_signal_b_'] = (results_dataframe['H1_time_signal_a'] > results_dataframe['H1_time_signal_b'])*results_dataframe['H1_time_signal_a'] + (results_dataframe['H1_time_signal_b'] >= results_dataframe['H1_time_signal_a'])*results_dataframe['H1_time_signal_b'] - 50

    results_dataframe = results_dataframe.drop(['H1_time_signal_a', 'H1_time_signal_b'], axis=1)

    results = model.predict(validation_dataset.map(getInputRegression_old, num_parallel_calls=tf.data.AUTOTUNE))

    for i, data in enumerate(validation_dataset.map(getInputRegression).take(10)):
        plt.figure()
        plt.plot(results[i])
        #print(getInputRegression(data).numpy()[1])
        #plt.plot(getInputRegression(data).numpy()[1])
        plt.savefig(f"test_plot_{i}.png")

    results_dataframe['results_A'] = results[:,0]
    results_dataframe['results_B'] = results[:,1]

    # = model.predict(validation_dataset.map(getInput, num_parallel_calls=tf.data.AUTOTUNE))[:,1]

    results_dataframe.to_csv('validation_scores_regression.csv')
    results_dataframe.to_pickle('validation_scores_regression.pkl')

print(results_dataframe)
