from gwpy.timeseries import TimeSeries, TimeSeriesDict
from pycbc.waveform import get_td_waveform
from pycbc.detector import Detector
from pycbc.noise.reproduceable import colored_noise
from pycbc.noise import noise_from_psd
from pycbc.psd import interpolate
from pycbc.psd.analytical import from_string as psd_from_string
from pycbc.filter import matched_filter
from tqdm import tqdm
import multiprocessing as mp
from pesummary.utils.utils import iterator
import numpy as np
import parameter_drawing

def generate_waveform_polarisations_from_parameters(name, parameters, detector, f_low, f_samp, location):
    # Generate relevant waveforms
    waveforms_dict = {}
    waveforms_dict[f"{name}_hp"], waveforms_dict[f"{name}_hc"] = get_waveform_as_gwpyTS(
        name=name,
        parameters=parameters,
        detector=detector,
        f_low=f_low,
        f_samp=f_samp,
    )

    SNRs = []
    for det in ["H1", "L1", "V1"]:
        # Project into the detector
        projected_waveform = project_waveform(
            hp=waveforms_dict[f"{name}_hp"],
            hc=waveforms_dict[f"{name}_hc"],
            detector=det,
            parameters=parameters,
            name=name
        )
        # Calculate SNR
        SNRs.append(estimate_SNR(template=projected_waveform, detector=det, f_samp=f_samp))

    # Write each waveform to file
    write_all_waveforms_to_file(waveforms_dict=waveforms_dict, location=f"{location}waveforms/")
    return waveforms_dict[f"{name}_hp"].duration.value, waveforms_dict[f"{name}_hp"].t0.value, waveforms_dict[f"{name}_hp"].t0.value + (len(waveforms_dict[f"{name}_hp"]) / f_samp), SNRs


def generate_waveform_polarisations_from_parameters_wrapper(args):
    return generate_waveform_polarisations_from_parameters(*args)


def generate_waveforms_from_signals_dict(
    signals_dict, location, detector="H1", f_low=20, f_samp=1024.0, nprocesses=1, checkpoint_start=0
):

    with mp.Pool(nprocesses) as pool:
        Nsignals = len(signals_dict.keys()) - checkpoint_start
        args = np.array(
            [list(signals_dict.keys())[checkpoint_start:],
            list(signals_dict.values())[checkpoint_start:],
            [detector] * Nsignals,
            [f_low] * Nsignals,
            [f_samp] * Nsignals,
            [location] * Nsignals,
            ],
            dtype=object
        ).T
        generation_data = np.array(list(iterator(
                pool.imap(generate_waveform_polarisations_from_parameters_wrapper, args),
                tqdm=True,
                desc="Generating waveforms",
                total=Nsignals,
                bar_format = '{desc} | {percentage:3.0f}% | {n_fmt}/{total_fmt} | {elapsed}'
            )),
            dtype=object
        )

    signals = list(signals_dict.keys())[checkpoint_start:]
    for idx, data in enumerate(generation_data):
        signal = signals[idx]
        # Update durations in the signals dict & update start/end times
        signals_dict[signal][f"duration_{int(f_low)}"] = data[0]
        signals_dict[signal]["geocent_waveform_start_time"] = data[1]
        signals_dict[signal]["geocent_waveform_end_time"] = data[2]
        # Re-store waveform low frequency cut-off in case it has been modified for signal melting into noise cut off
        signals_dict[signal]["f_low"] = f_low
        signals_dict[signal]["H1_SNR"] = data[3][0]
        signals_dict[signal]["L1_SNR"] = data[3][1]
        signals_dict[signal]["V1_SNR"] = data[3][2]

    return signals_dict


def get_waveform_as_gwpyTS(
    name, parameters, detector, f_low=20, f_samp=1024.0
):
    """Generates a signal from the given parameters.

    Parameters
    ----------
    parameters : `dictionary`
        A dictionary containing the parameters for the signal
    detector : `string`
        An indication of the detector

    Returns
    ------
    signal : `gwpy.timeseries.TimeSeries`
        A timeseries containing the generated signal.
    """

    try:
        hp, hc = _get_waveform(
            parameters=parameters, f_low=f_low, f_samp=f_samp
        )
        hp, hc = TimeSeries.from_pycbc(hp), TimeSeries.from_pycbc(hc)

        start_offset = parameter_drawing.apply_data_offset(hp.t0.value)
        hp.t0 = start_offset + parameters["geocent_time"]
        hc.t0 = start_offset + parameters["geocent_time"]
        hp.name = hp.channel = hc.name = hc.channel = name
        return hp, hc
    except RuntimeError:
        return None


def get_next_power_of_two(length):
    """Returns the next highest power of two for a number"""

    power = np.log2(length)
    if power == int(power):
        return length
    else:
        return int(2 ** (int(power) + 1))


def estimate_SNR(template, detector, f_samp=1024.):
    start_time = template.t0.value
    signal_length = len(template) * template.dt.value
    #signal_length = get_next_power_of_two(signal_length)
    template = template.to_pycbc().to_frequencyseries()
    noise, psd = generate_noise(
        length=signal_length,
        start_time=start_time,
        f_samp=f_samp,
        delta_f=template.delta_f,
        f_low=20,
        detector=detector,
        random_seed=0,
        return_PSD=True
    )

    noise = noise.to_pycbc().to_frequencyseries()
    noise = interpolate(noise, template.delta_f)

    if len(noise) < len(template):
        template = template[:len(noise)]
    elif len(template) < len(noise):
        noise = noise[:len(template)]

    data = template + noise
    snr = matched_filter(template, data, psd=psd, low_frequency_cutoff=20)
    return float(np.abs(snr).max())


def estimate_SNRs(waveforms_dict, detector, signals_dict, return_array=False):
    for key in waveforms_dict.keys():
        if f"{detector}_SNR" not in signals_dict[key]:
            signals_dict[key][f"{detector}_SNR"] = estimate_SNR(
                 template=waveforms_dict[key],
                 detector=detector
            )
    if return_array:
        return np.array([signals_dict[key][f"{detector}_SNR"] for key in signals_dict.keys()])
    else:
        return signals_dict


def project_waveform(hp, hc, detector, parameters, name):

    fp, fc = Detector(detector).antenna_pattern(
        right_ascension=parameters["ra"],
        declination=parameters["dec"],
        polarization=parameters["polarization"],
        t_gps=parameters[f"{detector}_time"],
    )

    ht = fp * hp + fc * hc

    diff = parameters[f"{detector}_time_delay"]
    ht.t0 = ht.t0.value + diff
    ht.taper()

    ht.name = name
    ht.channel = f"{detector}:GDS-CALIB_STRAIN"

    return ht


def project_waveforms(waveforms_dict, detector, signals_dict):
    # Get the names of all signals to be projected
    keys = list(set([name[:-3] for name in waveforms_dict.keys()]))
    keys.sort()
    for key in keys:
        waveforms_dict[key] = project_waveform(
            hp=waveforms_dict[f"{key}_hp"],
            hc=waveforms_dict[f"{key}_hc"],
            detector=detector,
            parameters=signals_dict[key],
            name=key
        )
        del(waveforms_dict[f"{key}_hp"])
        del(waveforms_dict[f"{key}_hc"])
    return waveforms_dict


def _get_waveform(parameters, f_low, f_samp=1024.0):
    """Generates a signal from a set of parameters

    Parameters
    ----------
    parameters : `dictionary`
        A dictionary containing the parameters for the signal
    detector : `string`
        An indication of the detector
    """

    if parameters["spin1x"] == parameters["spin1y"] == parameters["spin2x"] == parameters["spin2y"] == 0.0:
        approximant = "IMRPhenomT"
    else:
        approximant = "IMRPhenomTPHM"

    return get_td_waveform(
        mass1=parameters["mass1"],
        mass2=parameters["mass2"],
        spin1x=parameters["spin1x"],
        spin1y=parameters["spin1y"],
        spin1z=parameters["spin1z"],
        spin2x=parameters["spin2x"],
        spin2y=parameters["spin2y"],
        spin2z=parameters["spin2z"],
        distance=parameters["luminosity_distance"],
        f_lower=f_low,
        f_ref=parameters["f_ref"],
        coa_phase=parameters["phase"],
        inclination=parameters["inclination"],
        delta_t=1/f_samp,
        approximant=approximant,
        taper=True
    )


def write_waveform_to_file(name, waveform, location):
    if waveform.t0.value < 0:
        n_neg_samples = int(abs(waveform.t0.value) / waveform.dt.value)
        waveform = waveform[n_neg_samples:]
    waveform.write(f"{location}{name}.gwf")
    return None


def write_all_waveforms_to_file(waveforms_dict, location):
    for name, waveform in waveforms_dict.items():
        write_waveform_to_file(name=name, waveform=waveform, location=location)
    return None


def generate_noise(
    length,
    start_time=1260000000,
    f_samp=1024.0,
    delta_f=0.25,
    f_low=10,
    detector="H1",
    random_seed=0,
    return_PSD=False,
):
    """Generates aLIGO detector noise from the Zero-Detuned-High-Power
    PSD.

    Parameters
    ----------
    length : `int`
        The length of the noise in seconds
    start_time : `int`
        The start time of the run
    f_samp : `int`
        The sampling frequency of the data
    delta_f : `float`
        Frequency spacing required for the PSD
    f_low : `int`
        Low frequency cut off for the PSD
    detector : `str`
        Which detector to generate noise for

    Returns
    -------
    noise : `gwpy.timeseries.TimeSeries`
        A timeseries containing the generated noise.
    """

    f_length = int(length * f_samp)

    if detector in ["H1", "L1"]:
        psd_name = "aLIGOZeroDetHighPower"
    elif detector == "V1":
        psd_name = "AdvVirgo"

    psd = psd_from_string(
        psd_name=psd_name, length=f_length, delta_f=delta_f, low_freq_cutoff=f_low
    )

    if return_PSD:
        delta_t = 1.0 / f_samp
        success = False
        while not success:
            try:
                ts = noise_from_psd(f_length, delta_t, psd, seed=0)
                success = True
            except RuntimeError:
                print("Noise generation encountered an error, attempting at a lower delta_t")
                delta_t *= 0.99
    else:
        ts = colored_noise(
            psd=psd,
            start_time=start_time,
            end_time=start_time+length,
            seed=random_seed,
            low_frequency_cutoff=f_low,
            sample_rate=f_samp,
        )
    data = TimeSeries.from_pycbc(ts)
    # Zero noise run
    #data *= 0
    data.t0 = start_time

    if return_PSD:
        return data, psd
    else:
        return data


def get_waveform_times(waveforms_dict):
    # Change this. Single signal only
    start_times = [value.t0.value for value in waveforms_dict.values()]
    end_times = []
    for start_time, waveform in zip(start_times, waveforms_dict.values()):
        end_times.append(start_time + len(waveform) / waveform.dt.value)

    return np.array(start_times), np.array(end_times)


def get_relevant_signal_indices(
    segment_start_time, waveform_start_times, waveform_end_times, chunk_size
):
    seg_end = segment_start_time + chunk_size
    indices = np.arange(len(waveform_start_times))
    full_wav_indices = indices[
        (waveform_start_times < seg_end) & (waveform_end_times < seg_end)
    ]
    part_wav_indices = indices[
        (waveform_start_times < seg_end) & (waveform_end_times >= seg_end)
    ]

    # Remove those that end before current segment
    full_wav_indices = full_wav_indices[
        waveform_end_times[full_wav_indices] >= segment_start_time
    ]
    part_wav_indices = part_wav_indices[
        waveform_end_times[part_wav_indices] >= segment_start_time
    ]

    return full_wav_indices, part_wav_indices


def add_full_signal_to_data(data, waveform):
    # Replacing weird broken gwpy inject method

    start_idx = np.argmin(np.abs(data.times.value - waveform.t0.value))
    indices = (start_idx, start_idx+len(waveform))

    # Copy data from waveform only
    wf_data = np.copy(waveform.data)

    # Copy data from injection region
    noise_data = np.copy(data[indices[0]:indices[1]].data)

    # Recombine
    data[indices[0]:indices[1]] = noise_data + wf_data
    return data

def add_part_signal_to_data(data, signal_name, data_end, waveforms_dict):
    signal = waveforms_dict[signal_name]
    sig_end_idx = int((data_end - signal.t0.value) / signal.dt.value)
    data = data.inject(signal[:sig_end_idx])

    waveforms_dict[signal_name] = waveforms_dict[signal_name][sig_end_idx:]
    return data

def load_all_waveforms(location, names):
    waveforms_dict = TimeSeriesDict()
    for key in names:
        waveforms_dict[f"{key}_hp"] = TimeSeries.read(f"{location}{key}_hp.gwf", channel=key)
        waveforms_dict[f"{key}_hc"] = TimeSeries.read(f"{location}{key}_hc.gwf", channel=key)
    return waveforms_dict

def load_relevant_waveforms(location, indices):
    names = [f"signal{idx}" for idx in indices]
    waveforms_dict = TimeSeriesDict()
    for key in names:
        waveforms_dict[f"{key}_hp"] = TimeSeries.read(f"{location}{key}_hp.gwf", channel=key)
        waveforms_dict[f"{key}_hc"] = TimeSeries.read(f"{location}{key}_hc.gwf", channel=key)
    return waveforms_dict


def get_projected_times(detector, signals_dict):

    waveform_start_times = np.empty(len(signals_dict.keys()))
    waveform_end_times = np.empty(len(signals_dict.keys()))

    for idx, (signal, parameters) in enumerate(signals_dict.items()):
        waveform_start_times[idx] = parameters["geocent_waveform_start_time"] + parameters[f"{detector}_time_delay"]
        waveform_end_times[idx] = parameters["geocent_waveform_end_time"] + parameters[f"{detector}_time_delay"]

    return waveform_start_times, waveform_end_times
