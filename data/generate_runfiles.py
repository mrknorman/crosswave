 #!/usr/bin/env python3

import os
import shutil
from fileinput import FileInput
import multiprocessing as mp
import h5py
import numpy as np
from pesummary.utils.utils import iterator
import signal_generation
import parameter_drawing
from gwpy.timeseries import TimeSeries

def make_cache(location, detector, name, start, length, new_noise, chunk_size):
    if not new_noise:
        data_location = f"{location}/../GAUSSIAN_NOISE/"
    else:
        data_location = location

    N_files = length // chunk_size

    os.system(
        f"ls {data_location}{detector}_data/{detector}*.gwf | head -n {N_files} | "
        f"lalapps_path2cache > {location}{detector}-{name}-{start}-{length}.lcf"
    )
    return None


def generate_vetoes(
    detectors=["H1", "L1"], length=300, start_time=1260000000,
    original_filepath="./runfiles/default_files/",
    new_filepath="./runfiles/TEST/"
):
    """Copies and modifies the generic veto files to contain the
    correct segment length.

    Parameters
    ----------
    detectors : `list`
        A list of detectors names (as strings)
    length : `int`
        The length of the run in seconds
    start_time : `int`
        The GPS start time of the run
    original_filepath : `str`
        The location of the default veto files
    new_filepath : `str`
        The end location of the converted veto files
    """

    end_time = int(start_time + length)
    other_time = int(start_time - length)
    for filetype in ["science_segments", "veto_definer"]:
        shutil.copyfile(
            src=f"{original_filepath}{filetype}.xml",
            dst=f"{new_filepath}{filetype}.xml"
        )
        with FileInput(f"{new_filepath}{filetype}.xml", inplace=True) as file:
            for line in file:
                if "REPLACE" in line:
                    if "REPLACESTARTTIME" in line:
                        line = line.replace("REPLACESTARTTIME", str(start_time))
                    if "REPLACEENDTIME" in line:
                        line = line.replace("REPLACEENDTIME", str(end_time))
                    if "REPLACEBEFOREALL" in line:
                        line = line.replace("REPLACEBEFOREALL", str(other_time))
                    print(line.rstrip("\n"))
                else:
                    print(line.rstrip("\n"))
    return None


def create_strain_segment(
    detector,
    segment_start_time,
    relevant_signals_dict,
    rootdir,
    label,
    random_seed,
    chunk_size,
    gen_signal_idx
):
    data = signal_generation.generate_noise(
        length=chunk_size,
        start_time=segment_start_time,
        f_samp=1024.0,
        delta_f=0.25,
        f_low=10,
        random_seed=random_seed,
        detector=detector
    )
    '''# Zero noise
    data = TimeSeries(np.zeros(1024*16))
    data.t0 = segment_start_time
    data.dt = 1/1024'''

    # Determine which signals to inject
    relevant_signals_indices = [int(val.split("signal")[-1]) for val in list(relevant_signals_dict.keys())]
    if gen_signal_idx < 2:
        relevant_signals_indices = [relevant_signals_indices[gen_signal_idx]]
    filename = f"{rootdir}{detector}_data/{label}/signal{relevant_signals_indices[0]}.npy"

    # Load and project new waveforms in dictionary
    waveforms_dict = signal_generation.load_relevant_waveforms(location=f"{rootdir}waveforms/", indices=relevant_signals_indices)
    waveforms_dict = signal_generation.project_waveforms(waveforms_dict=waveforms_dict, detector=detector, signals_dict=relevant_signals_dict)

    # Inject waveforms
    for name, waveform in waveforms_dict.items():
        data = signal_generation.add_full_signal_to_data(
            data=data,
            waveform=waveform
        )

    # Convert data to 16 bit intgers
    new_data = np.array(data.data)

    new_data *= 1e22

    new_data = new_data.astype(np.float16)

    # Save data to file
    return new_data


def create_strain_segment_wrapper(args):
    return create_strain_segment(*args)


def create_detector_strain(
    detector,
    checkpoint_idx,
    number_of_segments,
    true_start_time,
    waveform_start_times,
    waveform_end_times,
    rootdir,
    signals_dict,
    label,
    random_seed,
    Nprocesses,
    inject_into_data,
    chunk_size
):

    seg_indices = np.arange(checkpoint_idx, number_of_segments)
    segment_start_times = list(true_start_time + (chunk_size * seg_indices))
    size = len(seg_indices)

    # Get the indicies of signals fully inside, or partially inside this segment
    all_full_wav_indices = []
    all_part_wav_indices = []
    all_all_indices = []
    all_relevant_signals_dicts = []
    for segment_start_time in segment_start_times:
        full_wav_indices, part_wav_indices = signal_generation.get_relevant_signal_indices(segment_start_time, waveform_start_times, waveform_end_times, chunk_size=chunk_size)
        all_indices = np.append(full_wav_indices, part_wav_indices)
        all_full_wav_indices.append(full_wav_indices)
        all_part_wav_indices.append(part_wav_indices)
        all_all_indices.append(all_indices)
        relevant_signals_dict = {}
        for idx in all_indices:
            relevant_signals_dict[f"signal{idx}"] = signals_dict[f"signal{idx}"]
        all_relevant_signals_dicts.append(relevant_signals_dict)

    with mp.Pool(Nprocesses) as pool:
        args = np.array([
            [detector] * size,
            segment_start_times,
            list(np.meshgrid(waveform_start_times, np.zeros(size))[0]),
            list(np.meshgrid(waveform_end_times, np.zeros(size))[0]),
            all_full_wav_indices,
            all_part_wav_indices,
            all_relevant_signals_dicts,
            [rootdir] * size,
            [label] * size,
            [random_seed] * size,
            [inject_into_data] * size,
            [chunk_size] * size
            ],
            dtype=object
        ).T
        result = list(iterator(
            pool.imap(create_strain_segment_wrapper, args),
            tqdm=True,
            desc=f"Generating {detector} data segments",
            total=size,
            bar_format = "{desc} | {percentage:3.0f}% | {n_fmt}/{total_fmt} | {elapsed}"
        ))
    del(result)

    return None

def generate_injection_file_hdf(signals_dict, path):
    """A function to convert the signals dict into a HDF injection file
    for the pipelines
    """

    hdf_file = h5py.File(path, "w")

    pairs = {
        "coa_phase": "phase",
        "distance": "luminosity_distance",
        "f_lower": "f_low",
        "inclination": "inclination",
        "ra": "ra",
        "dec": "dec",
        "mass1": "mass1",
        "mass2": "mass2",
        "mchirp": "chirp_mass",
        "polarization": "polarization",
        "spin1x": "spin1x",
        "spin1y": "spin1y",
        "spin1z": "spin1z",
        "spin2x": "spin2x",
        "spin2y": "spin2y",
        "spin2z": "spin2z",
        "tc": "geocent_time",
    }

    non_zeros = {
        "amp_order": -1,
        "approximant": b"SEOBNRv4PHM",
        "numrel_data": b"",
        "source": b"",
        "taper": b"TAPER_START",
        "waveform": b"SEOBNRv4PHM",
    }

    zeros = ["alpha", "alpha1", "alpha2", "alpha3", "alpha4", "alpha5", "alpha6", "bandpass", "beta", "eff_dist_g", "eff_dist_h", "eff_dist_l", "eff_dist_t", "eff_dist_v", "end_time_gmst", "eta", "f_final", "g_end_time", "g_end_time_ns", "h_end_time", "h_end_time_ns", "l_end_time", "l_end_time_ns", "v_end_time", "v_end_time_ns", "t_end_time", "t_end_time_ns", "numrel_mode_max", "numrel_mode_min", "phi0", "psi0", "psi3", "theta0"]

    hdf_keys = zeros + [key for key in pairs.keys()] + [key for key in non_zeros]
    for key in hdf_keys:
        if key in pairs.keys():
            value = pairs[key]
            data = np.array([signals_dict[signal][value] for signal in signals_dict.keys()])
        elif key in non_zeros:
            value = non_zeros[key]
            data = np.array([value for idx in range(len(signals_dict.keys()))])
        else:
            data = np.zeros(len(signals_dict.keys()))
        hdf_file.create_dataset(key, data=data)

    hdf_file.attrs.create("injtype", "cbc")
    hdf_file.attrs.create("static-args", np.array([], dtype="float64"))
    hdf_file.close()
    return None


def generate_injection_file_xml(signals_dict, path):
    """A function to convert the signals dict into a HDF injection file
    for the pipelines
    """

    # Open the default injection file
    default_xml_file = "./runfiles/default_files/default_injection.xml"
    with open(default_xml_file, "r") as f:
        contents = f.readlines()

    # Loop over elements in signals_dict
    for index, parameters in enumerate(signals_dict.values()):
        # Generate rows
        row = get_xml_injection_rows(index, parameters)
        # Add newline to row
        row += "\n"
        # Add row to file
        injection_start_row = 91
        contents.insert(index+injection_start_row, row)

    # Save injections to new file
    with open(path, "w") as f:
        f.writelines(contents)

    return None


def get_xml_injection_rows(index, parameters):
    pairs = {
        "coa_phase": "phase",
        "distance": "luminosity_distance",
        "inclination": "inclination",
        "longitude": "ra",
        "latitude": "dec",
        "mass1": "mass1",
        "mass2": "mass2",
        "mchirp": "chirp_mass",
        "polarization": "polarization",
        "spin1x": "spin1x",
        "spin1y": "spin1y",
        "spin1z": "spin1z",
        "spin2x": "spin2x",
        "spin2y": "spin2y",
        "spin2z": "spin2z",
    }

    non_zeros = {
        "amp_order": -1,
        "approximant": '"SEOBNRv4PHM"',
        "numrel_data": '""',
        "source": '""',
        "taper": '"TAPER_START"',
        "process_id": '"process:process_id:0"',
    }

    zeros = ["alpha", "alpha1", "alpha2", "alpha3", "alpha4", "alpha5", "alpha6", "bandpass", "beta", "eff_dist_g", "eff_dist_h", "eff_dist_l", "eff_dist_t", "eff_dist_v", "end_time_gmst", "eta", "f_final", "g_end_time", "g_end_time_ns", "h_end_time", "h_end_time_ns", "l_end_time", "l_end_time_ns", "v_end_time", "v_end_time_ns", "t_end_time", "t_end_time_ns", "numrel_mode_max", "numrel_mode_min", "phi0", "psi0", "psi3", "theta0", "geocent_end_time", "geocent_end_time_ns"]

    true_columns = ["alpha", "alpha1", "alpha2", "alpha3", "alpha4", "alpha5", "alpha6", "amp_order", "bandpass", "beta", "coa_phase", "distance", "eff_dist_g", "eff_dist_h", "eff_dist_l", "eff_dist_t", "eff_dist_v", "end_time_gmst", "eta", "f_final", "f_lower", "g_end_time", "g_end_time_ns", "geocent_end_time", "geocent_end_time_ns", "h_end_time", "h_end_time_ns", "inclination", "l_end_time", "l_end_time_ns", "latitude", "longitude", "mass1", "mass2", "mchirp", "numrel_data", "numrel_mode_max", "numrel_mode_min", "phi0", "polarization", "process_id", "psi0", "psi3", "simulation_id", "source", "spin1x", "spin1y", "spin1z", "spin2x", "spin2y", "spin2z", "t_end_time", "t_end_time_ns", "taper", "theta0", "v_end_time", "v_end_time_ns", "waveform"]

    row = "\t\t\t"
    for name in true_columns:
        if name in pairs.keys():
            if type(parameters[pairs[name]]) == float:
                row += f"{parameters[pairs[name]]}"
            else:
                row += str(parameters[pairs[name]])
        elif name in non_zeros:
            row += str(non_zeros[name])
        elif name == "simulation_id":
            row += '"' + f"sim_inspiral:simulation_id:{index}" + '"'
        elif name == "geocent_end_time":
            # REMEMBER: End time in XML is actually the merger time
            row += str(int(divmod(parameters["geocent_time"], 1)[0]))
        elif name == "geocent_end_time_ns":
            # Enforce that this has a certain length (ns)
            geocent_ns = f"{divmod(parameters['geocent_time'], 1)[1]:.9f}"[2:]
            # Recersively reduce this string until the first character is not zero
            while geocent_ns[0] == "0" and len(geocent_ns) > 1:
                geocent_ns = geocent_ns[1:]
            row += geocent_ns
        elif name == "waveform":
            if parameters["spin1x"] == parameters["spin1y"] == parameters["spin2x"] == parameters["spin2y"] == 0.0:
                row += '"SEOBNRv4_opt"'
            else:
                row += '"SEOBNRv4PHM"'
        elif name == "f_lower":
            row += str(parameters["f_low"] - 1)
        else:
            row += "0"
        row += ","

    return row

