import tensorflow as tf
import tensorflow_datasets as tfds
import os
import h5py
import numpy as np
import json

from tqdm import tqdm

def setup_CUDA(verbose : bool, device_num : str):
    
    """
    Select CUDA visbile devices, and intilise multi-GPU 
    strategy.
    
    Args:
        verbose: Print list of visible devices.
        device_num: Comma separated string to select
                    CUDA_VISIBLE_DEVICES
    
    Returns:
        Tensorflow multi GPY strategy.

    """
    
    # Set CUDA_VISIBLE_DEVICES path name so that
    # CUDA runs on selected devices:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_num)
        
    # Get list of GPUs avalible to CUDA:
    gpus = tf.config.list_logical_devices('GPU')
    
    # Create multi-gpu training stratergy:
    strategy = tf.distribute.MirroredStrategy(gpus)
    
    physical_devices = tf.config.list_physical_devices('GPU')
    
    # Ensure that tensorflow does not absorb all
    # the memory on any given GPU:
    for device in physical_devices:    

        try:
            tf.config.experimental.set_memory_growth(device, True)
        except:
            # Invalid device or cannot modify virtual devices 
            # once initialized.
            pass
    
    # Change the error logging level:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # If verbose print list of avalible GPUs:
    if verbose:
        tf.config.list_physical_devices("GPU")
        
    return strategy

def open_hdf5(file_path : str):
    
    """
    Read hdf5 files containing waveform data.

    Args:
        file_path: path to hdf5 file containing GW data

    Returns:
        List of data read from hdf5 file.
    """
    
    data = []

    # Open input file in read-only mode:
    with h5py.File(file_path, "r") as f:        
        for key in ["SINGLES", "PAIRS"]:
            for element in tqdm(f[key]):
                # Scale data to around 1 for network
                scale_factor = 1#10e21
                # Cast to smaller precision to 
                # save space and processing time
                element = \
                    (element * scale_factor).astype(np.float16)
                data.append(element)
        
    return data
    
def add_feature_to_dict(
        label_element : dict, 
        attribute     : str, 
        features_dict : dict,
        index         : int,
        signal_name   : str
    ):
    
    """
    Add waveform feature read from json file into
    dictonary containing list of features values 
    in order to be converted into tensorflow dataset
    format.
    
    Basically transposing the way features from
        signal[index][feature] to signal[feature][index]

    Args:
        label_element: dictionary of features for signal
        at index
        attribute: feature name to extract from
        label_element.
        feature_dict: transposed dictionary
        index: signal index
        signal_name : in this case a or b.

    Returns:
        Transposed dictonary of features.
    """
    
    label_value = label_element[signal_name][attribute]
             
    if attribute == "overlaps_with":
        label_value = label_value.removeprefix("signal")

    features_dict[f"{attribute}_signal_{signal_name}"][index] = np.float32(label_value)

strategy = setup_CUDA(True, "-1")

if __name__ == "__main__":
    # Path to input data directory:
    data_directory  = "/home/philip.relton/projects/MLOverlaps/data/runfiles/MULTIDETECTOR/"
    
    # Data directory interior path names:
    label_file_names = \
    {
        "test"     : ["H1_test_data.hdf5"    , "L1_test_data.hdf5"    , "MULTIDETECTOR_test_signal_catalog.json"    ],
        "train"    : ["H1_train_data.hdf5"   , "L1_train_data.hdf5"   , "MULTIDETECTOR_train_signal_catalog.json"   ],
        "validate" : ["H1_validate_data.hdf5", "L1_validate_data.hdf5", "MULTIDETECTOR_validate_signal_catalog.json"]
    }

    # Output tensorflow dataset path:
    custom_data_dir = "../MLOverlaps_data/mloverlaps_dataset_multidetector_v1"
    
    # Set up arrays to hold:    
    datasets     = {} 
    attributes_a = []
    attributes_b = []
    
    for key in label_file_names.keys():
        
        print(f"Loading {key} labels...")
        
        dataset_length = 0
        with open(f"{data_directory}/{label_file_names[key][2]}") as label_file:
            
            label_dict = json.load(label_file)
            
            dataset_length = len(label_dict.keys())*2
            
            label_element = label_dict[f"signal0"]
            
            attributes = label_element.keys()
            attributes_a = [f"{attribute}_signal_a" for attribute in attributes]
            attributes_b = [f"{attribute}_signal_b" for attribute in attributes]

            features_dict = {}
            for attribute in attributes_a + attributes_b + ["overlap_present"]:
                features_dict[attribute] = np.zeros([dataset_length], dtype = np.float32)
                        
            num_signals = len(label_dict.keys())
            for index in range(num_signals):
                label_element = {'a' : label_dict[f"signal{index}"]}
                
                for attribute in attributes:
                    add_feature_to_dict(label_element, attribute, features_dict, index, "a")
            
            num_pairs = num_signals//2
            for index in range(num_pairs):
                
                label_element = {
                    'a' : label_dict[f"signal{index}"],
                    'b' : label_dict[f"signal{index + num_pairs}"]
                }
                
                pair_1_index = index + num_signals
                pair_2_index = pair_1_index + num_pairs
                
                pair_indicies = [pair_1_index, pair_2_index]
                
                for pair_index in pair_indicies:
                    for attribute in attributes:
                        add_feature_to_dict(label_element, attribute, features_dict, pair_index, "a")
                        add_feature_to_dict(label_element, attribute, features_dict, pair_index, "b")
                            
                    features_dict[f"overlap_present"][pair_index] = 1
        
        print(f"Loading {key} data...")
        H1_data = open_hdf5(f"{data_directory}/{label_file_names[key][0]}")
        L1_data = open_hdf5(f"{data_directory}/{label_file_names[key][1]}")
                
        data_length = len(H1_data[0])
        print(data_length)
        
        data_dict = {
            "H1_strain"          : H1_data, 
            "L1_strain"          : L1_data,
            "original_index"     : np.linspace(0, dataset_length - 1, dataset_length)
        }
        
        data_dict.update(features_dict)
        
        dataset = tf.data.Dataset.from_tensor_slices(data_dict)
        
        dataset_size = dataset.cardinality().numpy()
        dataset.shuffle(dataset_size)
        
        datasets[key] = dataset
                
    feature_types_dict = {}
    for attribute in attributes_a + attributes_b:
        feature_types_dict[attribute] = tfds.features.Scalar(dtype=np.float32)
    
    other_features = {
        "H1_strain"          : tfds.features.Tensor(shape = (data_length,), dtype=np.float16),
        "L1_strain"          : tfds.features.Tensor(shape = (data_length,), dtype=np.float16),
        "overlap_present"    : tfds.features.ClassLabel(num_classes = 2),
        "original_index"     : tfds.features.Scalar(dtype=np.float32)
    }
    
    feature_types_dict.update(other_features)
                
    # Define the builder:
    builder = tfds.dataset_builders.TfDataBuilder(
        name="mloverlaps_dataset",
        config="strain_and_label",
        version="2.0.0",
        data_dir=custom_data_dir,
        split_datasets = datasets,
        features=tfds.features.FeaturesDict(feature_types_dict),
        description="Dataset of CBC signals; overlapping and single.",
        release_notes={
            "2.0.0": "Multidetector v2",
        }
    )
    
    # Make the builder store the data as a TFDS dataset.
    builder.download_and_prepare()
    
