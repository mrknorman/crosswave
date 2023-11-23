#!/usr/bin/env python3

import shutil
import os
import json
import argparse
import numpy as np
import h5py
import multiprocessing as mp
from pesummary.utils.utils import iterator
from tqdm import tqdm, trange
from gwpy.timeseries import TimeSeries
import parameter_drawing
import signal_generation
import generate_runfiles
import settings

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--label", dest="label", default="TEST1")
parser.add_argument(
    "-N", "--Nsignals", dest="Nsignals", default=1000, type=float
)
parser.add_argument(
    "-d", "--detectors", dest="detectors", default=["H1", "L1"], nargs="*"
)
parser.add_argument(
    "-n", "--nprocesses", dest="nprocesses", default=1, type=int
)
parser.add_argument(
    "-s", "--no_signal_checkpoint", dest="no_signal_checkpoint", action="store_false"
)
parser.add_argument(
    "-c", "--no_data_checkpoint", dest="no_data_checkpoint", action="store_false"
)
parser.add_argument(
    "-i", "--injections", dest="injections", action="store_true"
)
parser.add_argument(
    "-P", "--fraction_primary_bns", dest="fraction_primary_bns", default=0, type=int
)
parser.add_argument(
    "-S", "--fraction_secondary_bns", dest="fraction_secondary_bns", default=0, type=int
)
parser.add_argument(
    "-g", "--generate_noise", dest="generate_noise", action="store_true"
)
opts = parser.parse_args()

# Set filepaths
rootdir = f"./runfiles/{opts.label}/"

# Set chunk size
chunk_size = 16

if opts.fraction_primary_bns < opts.fraction_secondary_bns:
    print("BNS+BBH overlaps should only have BNS as primary signal and BBH as the secondary")
    print("Please ensure that fraction_secondary_bns is less than fraction_primary_bns to prevent the opposite occuring")
    quit()

if not os.path.isdir(rootdir):
    os.makedirs(rootdir)
if not os.path.isdir(f"{rootdir}waveforms/"):
    os.makedirs(f"{rootdir}waveforms/")
for detector in opts.detectors:
    if not os.path.isdir(f"{rootdir}{detector}_data/"):
        os.makedirs(f"{rootdir}{detector}_data/")

# Load run settings
true_start_time = 50
f_low = 20. # Low f_low to stop hard cut-off in detector
fraction_overlap = 1

#for mod, name in zip([10, 10], ["test", "validate"]):
#for mod, name in zip([10], ["validate"]):
for mod, name in zip([1, 10, 10], ["train", "test", "validate"]):
    print(f"Creating dataset: {name}")

    Nsignals = int(opts.Nsignals) // mod
    setlabel = opts.label + "_" + name
    signal_catalog_file = f"{rootdir}{setlabel}_signal_catalog.json"

    # Remove this post testing
    #Nsignals = 2
    #fraction_overlap = 0

    # Prevent/limit overhang of signals
    buffer_time = 2
    injection_period = (true_start_time + buffer_time, true_start_time + chunk_size)

    # Generate parameters
    signals_dict = parameter_drawing.get_signals_dict(
        signal_catalog_file=signal_catalog_file,
        Nsignals=Nsignals,
        fraction_overlap=fraction_overlap,
        fraction_primary_bns=opts.fraction_primary_bns,
        fraction_secondary_bns=opts.fraction_secondary_bns,
        signals_start=injection_period[0],
        signals_end=injection_period[1],
        detectors=opts.detectors,
        f_low=20,
        f_ref=20,
        chunk_size=chunk_size,
    )

    # This resets Nsignals if signals_dict has been loaded
    # This assumes that the fraction of overlap is unchanged between the runs
    Nsignals = int(len(signals_dict.keys()) / (1 + fraction_overlap))


    # Generate pipeline injections file
    print("Generating injection files")
    singlesA_keys = [f"signal{idx}" for idx in range(Nsignals)]
    singlesB_keys = [f"signal{idx}" for idx in range(Nsignals, (1+fraction_overlap)*Nsignals)]
    singlesA_dict = {key: signals_dict[key] for key in singlesA_keys}
    singlesB_dict = {key: signals_dict[key] for key in singlesB_keys}

    # Split these here and feed in three times with different file names
    injection_names = ["PAIRS", "SINGLESA", "SINGLESB"]
    injection_dicts = [signals_dict, singlesA_dict, singlesB_dict]
    '''for name, dictionary in zip(injection_names, injection_dicts):
        generate_runfiles.generate_injection_file_xml(
            signals_dict=dictionary,
            path=f"{rootdir}/HLV-{name}-INJECTIONS-{true_start_time}-{run_length}.xml"
        )
        print(f"Generated {name} injection file")

    # Create additional files
    print("Generating veto files...")
    generate_runfiles.generate_vetoes(
        detectors=opts.detectors, length=run_length,
        start_time = start_time,
        original_filepath="./runfiles/default_files/", new_filepath=rootdir
    )
    print("All veto files generated")
    '''

    # Hard coded to 20 Hz here to enforce overlap in the visible band
    # Generate and store all signals
    print("Starting waveform generation...")
    print(f"Parallelising generation across {opts.nprocesses} CPUs")
    # Finding checkpoint starting point
    Nsignals_total = Nsignals * (1 + fraction_overlap)
    checkpoint_idx = 0
    if opts.no_signal_checkpoint:
        while os.path.isfile(f"{rootdir}/waveforms/signal{checkpoint_idx}_hp.gwf") and os.path.isfile(f"{rootdir}/waveforms/signal{checkpoint_idx}_hc.gwf"):
            checkpoint_idx += 1

    # Setting sampling frequency to 1024 for waveform generation to save time
    f_samp_wf_gen = 1024.
    print(f"These waveforms are generated at {f_samp_wf_gen} Hz")

    if checkpoint_idx == 0:
        signals_dict = signal_generation.generate_waveforms_from_signals_dict(signals_dict=signals_dict, location=rootdir, f_low=f_low, f_samp=f_samp_wf_gen, nprocesses=opts.nprocesses, checkpoint_start=0)
    else:
        print(f"Checkpointing from stopped run, starting from signal{checkpoint_idx}")
        # Really this section can and should be parallelised Phil
        # Add something that checks for these values in the signals_dict first
        for idx in trange(checkpoint_idx, desc="Reloading parameters from checkpointing"):
            # Load the signal for to set the relevant parameters
            if "geocent_waveform_start_time" not in signals_dict[f"signal{idx}"].keys():
                try:
                    waveform = TimeSeries.read(f"{rootdir}waveforms/signal{idx}_hp.gwf", channel=f"signal{idx}")
                    signals_dict[f"signal{idx}"][f"duration_{f_low}"] = waveform.duration.value
                    signals_dict[f"signal{idx}"]["geocent_waveform_start_time"] = waveform.t0.value
                    signals_dict[f"signal{idx}"]["geocent_waveform_end_time"] = waveform.t0.value + (len(waveform) * waveform.dt.value)
                except RuntimeError:
                    print(f"Failed to update signal{idx}")
                    signals_dict[f"signal{idx}"][f"duration_{f_low}"] = 0
                    signals_dict[f"signal{idx}"]["geocent_waveform_start_time"] = 0
                    signals_dict[f"signal{idx}"]["geocent_waveform_end_time"] = 0
        signals_dict = signal_generation.generate_waveforms_from_signals_dict(signals_dict=signals_dict, location=rootdir, f_low=f_low, f_samp=f_samp_wf_gen, nprocesses=opts.nprocesses, checkpoint_start=checkpoint_idx)

    signals_dict = parameter_drawing.get_network_SNRs(signals_dict, opts.detectors)

    # Resave signals_dict
    parameter_drawing.save_signals_dict(
        signals_dict=signals_dict,
        signal_catalog_file=signal_catalog_file,
    )

    print("All waveforms generated")

    # Project signals into noise
    signalA_keys = list(signals_dict.keys())[:Nsignals]
    signalB_keys = list(signals_dict.keys())[Nsignals:]

    relevant_signals_dicts = []
    for keyA, keyB in zip(signalA_keys, signalB_keys):
        relevant_signals_dicts.append({
            keyA: signals_dict[keyA],
            keyB: signals_dict[keyB]
        })

    for det_idx, detector in enumerate(opts.detectors):
        for gen_signal_idx, label in enumerate(["SINGLESA", "SINGLESB", "PAIRS1", "PAIRS2"]):
            #if gen_signal_idx < 2:
            #    continue
            random_seed = det_idx*int(1e7) + gen_signal_idx*int(1e6) + np.arange(Nsignals)
            with mp.Pool(opts.nprocesses) as pool:
                args = np.array([
                    [detector] * Nsignals,
                    [true_start_time] * Nsignals,
                    relevant_signals_dicts,
                    [rootdir] * Nsignals,
                    [label] * Nsignals,
                    random_seed,
                    [chunk_size] * Nsignals,
                    [gen_signal_idx] * Nsignals
                    ],
                    dtype=object
                ).T
                results = np.array(list(iterator(
                    pool.imap(generate_runfiles.create_strain_segment_wrapper, args),
                    tqdm=True,
                    desc=f"Generating {label} data segments for {detector}",
                    total=Nsignals,
                    bar_format = "{desc} | {percentage:3.0f}% | {n_fmt}/{total_fmt} | {elapsed}"
                )))
            with h5py.File(f"{rootdir}{detector}_{name}_data.hdf5", "a") as f:
                f[label] = results
            del(results)

    # Delete waveforms from checkpointing
    print("Deleting all waveform files...")

    # Generate all filenames
    indices = np.arange(Nsignals*2)

    filenamesA = np.empty(len(indices)).astype(str)
    filenamesB = np.empty(len(indices)).astype(str)
    for idx in indices:
        filenamesA[idx] = f"signal{idx}_hp.gwf"
        filenamesB[idx] = f"signal{idx}_hc.gwf"

    filenames = np.append(filenamesA, filenamesB)

    def remove_files(filename, path=f"{rootdir}/waveforms/"):
        '''Removes filenames based on path'''
        loc = path + filename
        if os.path.isfile(loc):
            os.remove(loc)

    # Delete all waveforms parallelised across cpus
    with mp.Pool(opts.nprocesses) as pool:
        results = np.array(list(iterator(
            pool.imap(remove_files, filenames),
            tqdm=True,
            desc=f"Deleting files",
            total=len(filenames),
            bar_format = "{desc} | {percentage:3.0f}% | {n_fmt}/{total_fmt} | {elapsed}"
        )))

    # Backup removal for stray files
    shutil.rmtree(f"{rootdir}waveforms/")
    os.makedirs(f"{rootdir}waveforms/")
    print("Waveform files deleted")

    for detector in opts.detectors:
        for valA, valB, final in zip(["SINGLESA", "PAIRS1"], ["SINGLESB", "PAIRS2"], ["SINGLES", "PAIRS"]):
            with h5py.File(f"{rootdir}{detector}_{name}_data.hdf5", "r+") as f:
                f[final] = np.append(f[valA], f[valB], axis=0)
                del(f[valA])
                del(f[valB])

    print(f"All data generated and saved in {rootdir}")

    # Convert output datasets to single file with arrays
    # Array contains H1, L1 and Geocent times
    output = []

    # Reorganise SINGLES
    '''datasets = {}
    for detector in opts.detectors:
        datasets[detector] = np.copy(h5py.File(f"{rootdir}{detector}_data/{setlabel}_data.hdf5", "r")["SINGLES"])'''

    #for sig_idx in trange(len(datasets["H1"])):
    for sig_idx in trange(len(signals_dict.keys())):
        tmp = []
        #tmp.append([datasets[opts.detectors[0]], datasets[opts.detectors[1]]])
        times = [
            signals_dict[f"signal{sig_idx}"]["H1_time"],
            0,
            signals_dict[f"signal{sig_idx}"]["L1_time"],
            0,
            signals_dict[f"signal{sig_idx}"]["geocent_time"],
            0,
        ]
        tmp.append(times)
        tmp.append([0])
        output.append(tmp)

    # Reorganise PAIRS
    '''datasets = {}
    for detector in opts.detectors:
        datasets[detector] = np.copy(h5py.File(f"{rootdir}{detector}_data/{setlabel}_data.hdf5", "r")["PAIRS"])'''

    #size = len(datasets["H1"]) // 2
    size = len(signals_dict.keys()) // 2
    # Repeating for PAIRS1 and PAIRS2
    for repeat_idx in range(2):
        for sig_idx in trange(size):
            tmp = []
            #tmp.append(np.array([datasets[opts.detectors[0]], datasets[opts.detectors[1]]]))
            times = np.array([
                signals_dict[f"signal{sig_idx}"]["H1_time"],
                signals_dict[f"signal{size + sig_idx}"]["H1_time"],
                signals_dict[f"signal{sig_idx}"]["L1_time"],
                signals_dict[f"signal{size + sig_idx}"]["L1_time"],
                signals_dict[f"signal{sig_idx}"]["geocent_time"],
                signals_dict[f"signal{size + sig_idx}"]["geocent_time"],
            ])
            tmp.append(times)
            tmp.append(np.array([1]))
            tmp = np.array(tmp)
            output.append(tmp)

    # Output main result file
    output = np.array(output)
    #np.random.shuffle(output)
    np.save(f"{rootdir}{setlabel}_data.npy", output)

    # Make files accessable
    for det in opts.detectors:
        os.chmod(f"{rootdir}{det}_{name}_data.hdf5", 0o664)
    os.chmod(f"{rootdir}{setlabel}_signal_catalog.json", 0o664)
    os.chmod(f"{rootdir}{setlabel}_data.npy", 0o664)
