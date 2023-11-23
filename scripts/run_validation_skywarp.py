import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')

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
    return (dataset['network_SNR_signal_a'] >= 12.0) and (dataset['network_SNR_signal_b'] >= 12.0)

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

def timeNorm(element):    
    return 0.0 + tf.cast((element != 0.0), np.float32)*((element - 60.5) / 4.0)

def massNorm(element):
    return 0.0 + tf.cast((element != 0.0), np.float32)*(element / 100.0)

def distanceNorm(element):
    return 0.0 + tf.cast((element != 0.0), np.float32)*(element / 300.0)

def timeNormI(element):    
    return (element*4.0 + 60.5)

def massNormI(element):
    return element*100.0

def distanceNormI(element):
    return element * 300.0

def getInputRegressionCross(element):
    return tf.transpose(tf.stack([element['H1_strain'], element['L1_strain']], axis=1), perm=[0, 2, 1]), tf.transpose(tf.stack([
             element['mass1_signal_a'],
             element['mass2_signal_a'],
             element['mass1_signal_b'],
             element['mass2_signal_b'],
             element['a1_signal_a'],
             element['spin1x_signal_a'],
             element['spin1y_signal_a'],
             element['spin1z_signal_a'],
             element['a2_signal_a'],
             element['spin2x_signal_a'],
             element['spin2y_signal_a'],
             element['spin2z_signal_a'],
             element['a1_signal_b'],
             element['spin1x_signal_b'], 
             element['spin1y_signal_b'],
             element['spin1z_signal_b'],
             element['a2_signal_b'],
             element['spin2x_signal_b'],
             element['spin2y_signal_b'],
             element['spin2z_signal_b'],
             element['luminosity_distance_signal_a'],
             element['luminosity_distance_signal_b'],
             element['H1_time_signal_a'],
             element['H1_time_signal_b'],
             element['L1_time_signal_a'],
             element['L1_time_signal_b'],
             element['geocent_time_signal_a'],
             element['geocent_time_signal_b'],
             tf.cast(element['overlap_present'], dtype = np.float32)
        ]
    ))

def getInputRegressionCrossOld(element):
    return tf.transpose(tf.stack([element['H1_strain'], element['L1_strain']], axis=1), perm=[0, 2, 1]), tf.transpose(tf.stack([
             massNorm(element['mass1_signal_a']),
             massNorm(element['mass2_signal_a']),
             massNorm(element['mass1_signal_b']),
             massNorm(element['mass2_signal_b']),
             element['a1_signal_a'],
             element['spin1x_signal_a'],
             element['spin1y_signal_a'],
             element['spin1z_signal_a'],
             element['a2_signal_a'],
             element['spin2x_signal_a'],
             element['spin2y_signal_a'],
             element['spin2z_signal_a'],
             element['a1_signal_b'],
             element['spin1x_signal_b'], 
             element['spin1y_signal_b'],
             element['spin1z_signal_b'],
             element['a2_signal_b'],
             element['spin2x_signal_b'],
             element['spin2y_signal_b'],
             element['spin2z_signal_b'],
             distanceNorm(element['luminosity_distance_signal_a']),
             distanceNorm(element['luminosity_distance_signal_b']),
             timeNorm(element['H1_time_signal_a']),
             timeNorm(element['H1_time_signal_b']),
             timeNorm(element['L1_time_signal_a']),
             timeNorm(element['L1_time_signal_b']),
             timeNorm(element['geocent_time_signal_a']),
             timeNorm(element['geocent_time_signal_b']),
             tf.cast(element['overlap_present'], dtype = np.float32)
        ]
    ))

setup_CUDA(1, 0)

# Load Dataset:
dataset = tfds.load(
    "mloverlaps_dataset",
    data_dir = "../MLOverlaps_data/mloverlaps_dataset_multidetector_v2"
)

regression = True

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
    model_path = "skywarp_prototype_regression_best"
    model = tf.keras.models.load_model(model_path)
    
    validation_dataset = dataset['validate'].filter(__regresionFilter).batch(batch_size=1)
        
    feature_names = list(dataset['validate'].element_spec.keys())
    feature_names.remove('H1_strain')
    feature_names.remove('L1_strain')
    
    results_dataframe = pd.DataFrame(validation_dataset, columns=feature_names).applymap(toNumpy)

    results = model.predict(validation_dataset.map(getInputRegressionCross, num_parallel_calls=tf.data.AUTOTUNE))
    
    features = [
        'mass1_signal_a',
        'mass2_signal_a',
        'mass1_signal_b',
        'mass2_signal_b',
        'a1_signal_a',
        'spin1x_signal_a',
        'spin1y_signal_a',
        'spin1z_signal_a',
        'a2_signal_a',
        'spin2x_signal_a',
        'spin2y_signal_a',
        'spin2z_signal_a',
        'a1_signal_b',
        'spin1x_signal_b', 
        'spin1y_signal_b',
        'spin1z_signal_b',
        'a2_signal_b',
        'spin2x_signal_b',
        'spin2y_signal_b',
        'spin2z_signal_b',
        'luminosity_distance_signal_a',
        'luminosity_distance_signal_b',
        'H1_time_signal_a',
        'H1_time_signal_b',
        'L1_time_signal_a',
        'L1_time_signal_b',
        'geocent_time_signal_a',
        'geocent_time_signal_b'
    ]
    
    for index, feature in enumerate(features):
        results_dataframe[feature + "_pred"] = results[:,index]
    
    results_dataframe.to_csv('skywarp_validation_scores_regression.csv')
    results_dataframe.to_pickle('skywarp_validation_scores_regression.pkl')

print(results_dataframe)
