import os
import json
import numpy as np
from scipy.interpolate import interp2d
from pesummary.gw.conversions import spins
from pycbc.waveform import get_td_waveform
from pycbc.detector import Detector
from tqdm import tqdm


def get_signals_dict(
    signal_catalog_file,
    Nsignals,
    fraction_overlap,
    fraction_primary_bns,
    fraction_secondary_bns,
    signals_start,
    signals_end,
    detectors=["H1", "L1"],
    f_low=20,
    f_ref=20,
    chunk_size=4
):
    try:
        print("Attempting to load existing signal parameters...")
        with open(signal_catalog_file, "r") as file:
            signals_dict = json.load(file)
        print("Existing signal dictionary loaded from file")
    except (FileNotFoundError, json.decoder.JSONDecodeError) as exceptions:
        print("No existing run parameters")
        print("Generating new run parameters")
        signals_dict = _get_signals_dict(
            Nsignals=Nsignals,
            fraction_overlap=fraction_overlap,
            fraction_primary_bns=fraction_primary_bns,
            fraction_secondary_bns=fraction_secondary_bns,
            signals_start=signals_start,
            signals_end=signals_end,
            detectors=detectors,
            f_low=f_low,
            f_ref=f_ref,
            chunk_size=chunk_size
        )
        print("New parameters generated")
        print("Writing new parameters to file...")
        save_signals_dict(
            signals_dict=signals_dict,
            signal_catalog_file=signal_catalog_file,
        )
        print("Parameters saved to file")

    return signals_dict


def get_Nsignals_array(Nsignals, fraction_overlap, fraction_primary_bns, fraction_secondary_bns):
    primary = Nsignals * np.array([(1-fraction_primary_bns), fraction_primary_bns])
    Noverlap = Nsignals * fraction_overlap
    secondary = Noverlap * np.array([(1-fraction_secondary_bns), fraction_secondary_bns])
    Nsignals_array = np.array([primary[0], primary[1], secondary[0], secondary[1]])
    assert int(Nsignals_array.sum()) == Nsignals * (1 + fraction_overlap)
    return Nsignals_array


def merge_signals_dicts(signals_dict_array):
    signals_dict = {}
    Nstart = 0
    for sub_dict in signals_dict_array:
        tmp_N = len(sub_dict.keys())
        for idx in range(len(sub_dict.keys())):
            signals_dict[f"signal{Nstart+idx}"] = sub_dict.pop(f"signal{idx}")
        Nstart += tmp_N

    assert len(signals_dict.keys()) == Nstart
    return signals_dict


def _get_signals_dict(
    Nsignals,
    fraction_overlap,
    fraction_primary_bns,
    fraction_secondary_bns,
    signals_start,
    signals_end,
    detectors,
    f_low=20,
    f_ref=20,
    chunk_size=4
):
    """Draws all parameters for a set of signals"""

    # Generate a N signals array: Totals to (1+fOL)Nsignals
    Nsignals_array = get_Nsignals_array(
        Nsignals=Nsignals,
        fraction_overlap=fraction_overlap,
        fraction_primary_bns=fraction_primary_bns,
        fraction_secondary_bns=fraction_secondary_bns
    )

    # Define BBH/BNS array for Nsignals_array comparison
    BNS_truth_array = np.array([False, True, False, True])
    if fraction_primary_bns not in [0, 1]:
        print("Primary BNS and BBH signals may overlap with this configuration")
        print("Phil you need to organise Signal A time layout better if you want to do this")
        quit()

    signals_dict_array = []
    for Nsignals_current, is_BNS in zip(Nsignals_array, BNS_truth_array):
        if Nsignals_current > 0:
            signals_dict_array.append(
                _get_all_parameters(
                    Nsignals=Nsignals_current,
                    signals_start=signals_start,
                    signals_end=signals_end,
                    is_BNS=is_BNS,
                    detectors=detectors,
                    f_low=f_low,
                    f_ref=f_ref,
                    chunk_size=chunk_size
                )
            )
        else:
            signals_dict_array.append({})

    all_parameters_dict = merge_signals_dicts(signals_dict_array)

    # Calculate overlaps from 10 Hz so that the overlap is always in band
    if fraction_primary_bns > 0:
        OL_f_low = 25
    else:
        OL_f_low = f_low

    all_parameters_dict = overlap_signals(
        all_parameters_dict, Nsignals, fraction_overlap, f_low=OL_f_low
    )
    # Enforcing 3 detector times here
    all_parameters_dict = add_detector_times(all_parameters_dict, detectors=["H1", "L1", "V1"])
    return all_parameters_dict


def get_signal_duration(mass1, mass2, f_low=20):
    """Calculates the visible duration of a signal
    (The duration of the signal with f > 20 Hz)
    """
    return get_td_waveform(
        mass1=mass1,
        mass2=mass2,
        approximant="IMRPhenomTPHM",
        f_lower=f_low,
        delta_t=1/1024,
    )[0].duration


def draw_mass_pairs(Nsignals):
    """Draws the requested number of component
    mass pairs from a given 2d distribution
    """

    # These are the defined grid arrays from LVK populations
    # releases
    mass1s = np.linspace(2, 100, 1000)
    mass_ratios = np.linspace(0.1, 1, 500)
    pdf = np.load("runfiles/default_files/distributions/PLP_GWTC3_aLIGO_biased.npy")

    print("Drawing mass pair samples")
    all_samples = rejection_sample_from_2d(
        mass1s, mass_ratios, pdf, size=Nsignals, importance_from_1d=False
    )

    mass1 = all_samples[0][:Nsignals]
    mass2 = all_samples[0][:Nsignals] * all_samples[1][:Nsignals]

    return mass1, mass2


def draw_mass_pairs_BNS(Nsignals):
    """Draws two component masses for BNS signals. Drawn uniformly from 1 to 3
    M_sun.
    """

    mass1 = np.random.uniform(1.14, 3, Nsignals)
    mass2 = np.empty(Nsignals)
    for idx in range(Nsignals):
        mass2[idx] = np.random.uniform(1.14, mass1[idx], 1)

    return mass1, mass2


def draw_spin_magnitude_pairs_BNS(Nsignals, limit=0.05):
    """Draws the spin magnitudes for BNS signals. Limited to ±0.05 unless
    specified.
    """
    return np.random.uniform(-limit, limit, 2*Nsignals).reshape(2, Nsignals)


def draw_spin_orientation_pairs(Nsignals):
    """Draws the requested number of component
    spin orientation pairs from a given 2d distribution
    NOTE: If you find the time you should collapse these into a single
    function.
    """

    '''# These are the defined grid arrays from LVK populations
    # releases
    spin_orientation = np.arccos(np.linspace(-1, 1, 1000))
    pdf = np.load("runfiles/default_files/distributions/PLP_GWTC3_spin_orientation.npy")

    print("Drawing spin orientation samples")
    all_samples = rejection_sample_from_2d(
        spin_orientation, spin_orientation, pdf, size=Nsignals, importance_from_1d=False
    )

    tilt1 = all_samples[0][:Nsignals]
    tilt2 = all_samples[1][:Nsignals]
    '''
    # MLO: Uniform spins
    tilt1 = np.random.uniform(0, np.pi, Nsignals)
    tilt2 = np.random.uniform(0, np.pi, Nsignals)

    return tilt1, tilt2


def draw_spin_magnitude_pairs(Nsignals):
    """Draws the requested number of component
    spin magnitude pairs from a given 2d distribution
    """

    '''# These are the defined grid arrays from LVK populations
    # releases
    spin_magnitude = np.linspace(0, 1, 1000)
    pdf = np.load("runfiles/default_files/distributions/PLP_GWTC3_spin_magnitude.npy")

    print("Drawing spin magnitude samples")
    all_samples = rejection_sample_from_2d(
        spin_magnitude, spin_magnitude, pdf, size=Nsignals, importance_from_1d=False
    )

    a1 = all_samples[0][:Nsignals]
    a2 = all_samples[1][:Nsignals]

    return a1, a2'''
    # MLO: Uniform spin magnitude
    return np.random.uniform(0, 1, Nsignals), np.random.uniform(0, 1, Nsignals)


def _rejection_sample_from_2d(x, y, pdf, size=1000, importance_from_1d=False):
    """
    Credit: Vivien Raymond
    Needs a pdf as a 2D array, defined at values x and y.
    Returns accepted samples, <size**2
    Can turn on importance sampling from the 1D marginal x
    and y PDFs. However this won't help sampling efficiency much
    if x and y are highly correlated.
    """

    # Temporary fix to stop this returning too few samples
    if size < 1000:
        size *= 10
    else:
        size *= 2

    # Building an interpolant of the normalised PDF
    normalised_pdf = normalise_pdf(x=x, y=y, pdf=pdf)
    pdfinterp = interp2d(x, y, normalised_pdf)

    # Getting x and y samples from the 1D marginal PDFs
    if importance_from_1d:
        x_pdf = np.trapz(pdf, y, axis=0)
        x_pdf_interp = interp1d(x, x_pdf)
        samplex = sample_from_1d(x, x_pdf, size=size)

        y_pdf = np.trapz(pdf, x, axis=-1)
        y_pdf_interp = interp1d(y, y_pdf)
        sampley = sample_from_1d(y, y_pdf, size=size)

    # Simply getting uniformly random x and y samples
    else:
        samplex = np.random.uniform(low=x.min(), high=x.max(), size=size)
        sampley = np.random.uniform(low=y.min(), high=y.max(), size=size)

    # Due to the way interp2d works, the samples have to be ordered...
    samplex.sort()
    sampley.sort()
    xx, yy = np.meshgrid(samplex, sampley)

    # The uniform variable to decide acceptance
    u = np.random.uniform(size=(size, size))

    # Rejection sampling, with/without importance sampling.
    if importance_from_1d:
        mask = (
            pdfinterp(samplex, sampley) / (x_pdf_interp(xx) * y_pdf_interp(yy))
            > u
        )

    else:
        mask = pdfinterp(samplex, sampley) > u

    # Selecting the relevant samples
    xsamples = xx[mask]
    ysamples = yy[mask]

    # Shuffling these arrays to remove order
    np.random.shuffle(xsamples)
    np.random.shuffle(ysamples)

    return xsamples[:size], ysamples[:size]


def rejection_sample_from_2d(x, y, pdf, size=1000, importance_from_1d=False):
    """Wrapper for `_rejection_sample_from_2d` to stop failures on very
    long sample sets
    """

    chunk_size = 1000 # Phil add a zero
    if size > chunk_size:
        all_samples_x = []
        all_samples_y = []
        for idx in range(int(np.ceil(size / chunk_size))):
            tmp_x, tmp_y = _rejection_sample_from_2d(
                    x, y, pdf, size=chunk_size, importance_from_1d=importance_from_1d
                )
            all_samples_x.append(list(tmp_x))
            all_samples_y.append(list(tmp_y))

        all_samples_x = np.array(all_samples_x).flatten()[:size]
        all_samples_y = np.array(all_samples_y).flatten()[:size]

    else:
        all_samples_x, all_samples_y = _rejection_sample_from_2d(
            x, y, pdf, size=size, importance_from_1d=importance_from_1d
        )

    return all_samples_x, all_samples_y


def normalise_pdf(x, y, pdf):
    return pdf / np.trapz(np.trapz(pdf, x=x), x=y)


def get_chirp_mass_q(mass1, mass2):
    """Calculates the chirp mass from component masses"""
    top = (mass1 * mass2) ** 3
    bottom = mass1 + mass2
    return (top / bottom) ** (1 / 5), mass2 / mass1


def apply_data_offset(times, true_start=0):
    times = np.atleast_1d(times)

    dt = 1 / 1024
    offsets = np.atleast_1d((times - true_start) % dt)

    upper_idx = np.where((offsets >= dt / 2))[0]
    if len(upper_idx) > 0:
        times[upper_idx] += dt - offsets[upper_idx]

    lower_idx = np.where((offsets < dt / 2))[0]
    if len(lower_idx) > 0:
        times[lower_idx] -= offsets[lower_idx]

    # Fix for very small values offset
    # Otherwise these are ruined by floating point offsets
    # Here the offset is applied and then rounded to an integer number of dt
    times = (times // dt) * dt

    if len(times) < 2:
        return times[0]
    else:
        return times


def add_detector_times(all_parameters_dict, detectors):
    """Calculates the adjusted merger time
    for signals in given detectors
    """

    for detector in list(detectors):
        det = Detector(detector)
        for name, signal in all_parameters_dict.items():
            detector_time_delay = det.time_delay_from_earth_center(
                right_ascension=signal["ra"],
                declination=signal["dec"],
                t_gps=signal["geocent_time"],
            )

            true_offset_time_delay = apply_data_offset(times=detector_time_delay)
            all_parameters_dict[name][f"{detector}_time"] = true_offset_time_delay + signal["geocent_time"]
            all_parameters_dict[name][f"{detector}_time_delay"] = true_offset_time_delay
    return all_parameters_dict


def get_extra_parameters(all_parameters, detectors):
    """Calculates and returns relevant spins, mass parameters for each signal"""

    (
        all_parameters["chirp_mass"],
        all_parameters["mass_ratio"],
    ) = get_chirp_mass_q(all_parameters["mass1"], all_parameters["mass2"])

    component_spins = get_component_spins(
        a1=all_parameters["a1"],
        tilt1=all_parameters["tilt1"],
        a2=all_parameters["a2"],
        tilt2=all_parameters["tilt2"],
        phi=all_parameters["phi"],
    )

    names = [
        "spin1x",
        "spin1y",
        "spin1z",
        "spin2x",
        "spin2y",
        "spin2z",
    ]
    all_parameters.update(dict(zip(names, component_spins.T)))

    all_parameters["chi_p"] = spins.chi_p(
        mass1=all_parameters["mass1"],
        mass2=all_parameters["mass2"],
        spin1x=all_parameters["spin1x"],
        spin1y=all_parameters["spin1y"],
        spin2x=all_parameters["spin2x"],
        spin2y=all_parameters["spin2y"],
    )

    all_parameters["chi_eff"] = spins.chi_eff(
        mass1=all_parameters["mass1"],
        mass2=all_parameters["mass2"],
        spin1z=all_parameters["spin1z"],
        spin2z=all_parameters["spin2z"],
    )

    return all_parameters


def polar_to_cartesian(R, theta, phi):
    R = np.array(R)
    theta = np.array(theta)
    phi = np.array(phi)

    x = R * np.sin(theta) * np.cos(phi)
    y = R * np.sin(theta) * np.sin(phi)
    z = R * np.cos(theta)
    return np.array([x, y, z]).T

def get_component_spins(a1, tilt1, a2, tilt2, phi):
    spin1 = polar_to_cartesian(
        R=a1,
        theta=tilt1,
        phi=phi
    )
    spin2 = polar_to_cartesian(
        R=a2,
        theta=tilt2,
        phi=phi
    )
    return np.hstack([spin1, spin2])


def get_buffered_time_array(start, end, pair_len_max, number):
    """This process spaces the times exactly throughout the observing
    run and then shifts them within the range that they are able to be
    shifted"""
    length = end - start - pair_len_max
    max_number = length / pair_len_max
    diff = (max_number - number) * pair_len_max
    single_diff = diff / number

    grid = np.linspace(start+pair_len_max, end, number)
    diff = np.random.uniform(0, single_diff, number) - single_diff/2
    if diff[-1] > 0:
        diff[-1] *= -1
    return grid + diff


def get_spaced_time_array(start, end, number, chunk_size=4, sep=60, is_BNS=False):
    grid = np.linspace(start, end, number)
    if is_BNS:
        if int((210 + end - start) / 1555) - 2 == number:
            # BNS+BNS run
            pair_len_max = 1555
        else:
            # BNS+BBH run
            pair_len_max = 1435
        grid = get_buffered_time_array(start, end, pair_len_max, number)
    else:
        if (end - start) / number < sep:
            print("Too many signals")
            print("Honestly! You should have sorted this when drawing Nsignals Phil")
            quit()
        # Calculate BBH spaced times
        grid = np.arange(number) * chunk_size + (chunk_size / 2)
        grid = grid + np.random.normal(0, 0.1, number)
        '''for idx in range(number):
            if idx == 0:
                grid[idx] = np.random.uniform(start, grid[idx+1]-sep, 1)
            elif idx == number - 1:
                grid[idx] = np.random.uniform(grid[idx-1]+sep, end, 1)
            else:
                grid[idx] = np.random.uniform(grid[idx-1]+sep, grid[idx+1]-sep, 1)'''
    return grid


def _get_all_parameters(
    Nsignals, signals_start, signals_end, is_BNS, detectors, f_low, f_ref, chunk_size
):
    """Performs drawing for all signal. Returns these as an array"""

    all_parameters = {}

    # Draw the mass pairings
    if is_BNS:
        all_parameters["mass1"], all_parameters["mass2"] = draw_mass_pairs_BNS(
            Nsignals=Nsignals
        )

        # Some of these will fall far too close, setting the minimum distance to be 100 Mpc
        # The upper range of distance is then 800 Mpc
        # This needs playing with to get the right SNR level. PHIL PAY ATTENTION TO THIS!!
        all_parameters["luminosity_distance"] = (
            np.random.power(2, Nsignals) * 145
        ) + 5 # This should probably be between 5 and 200? (200 is mean BNS range for aLIGO Design)

        # Draw the spin pairings
        all_parameters["a1"], all_parameters["a2"] = draw_spin_magnitude_pairs_BNS(
            Nsignals=Nsignals, limit=0.05
        )
        all_parameters["tilt1"], all_parameters["tilt2"] = np.zeros(2*Nsignals).reshape(2, Nsignals)

        # Draw the azimuthal spin angle
        all_parameters["phi"] = np.zeros(Nsignals)
    else:
        # MLO: Draw mass pairs
        # Set q depending on m1 to prevent super low mass events
        all_parameters["mass1"] = np.random.uniform(10, 70, Nsignals)
        all_parameters["mass2"] = np.empty(Nsignals)
        for idx, m1 in enumerate(all_parameters["mass1"]):
            if m1 <= 15:
                all_parameters["mass2"][idx] = m1 * np.random.uniform(0.8, 1, 1)[0]
            elif m1 <= 20:
                all_parameters["mass2"][idx] = m1 * np.random.uniform(0.34, 1, 1)[0]
            elif m1 < 30:
                all_parameters["mass2"][idx] = m1 * np.random.uniform(0.2, 1, 1)[0]
            if m1 >= 30:
                all_parameters["mass2"][idx] = m1 * np.random.uniform(0.1, 1, 1)[0]

        #all_parameters["mass1"], all_parameters["mass2"] = draw_mass_pairs(
        #    Nsignals=Nsignals
        #)

        # Some of these will fall far too close, setting the minimum distance to be 200 Mpc
        # The upper range of distance is then 1300 Mpc
        # MLO: Distance
        all_parameters["luminosity_distance"] = 500 + (100 * np.random.power(3, Nsignals))
        '''(
            np.random.power(2, Nsignals) * 500
        ) + 200'''

        '''# MLO: Fixed zero spins
        for param in ["a1", "a2", "tilt1", "tilt2", "phi"]:
            all_parameters[param] = np.zeros(Nsignals)'''
        # Draw the spin pairings
        all_parameters["a1"], all_parameters["a2"] = draw_spin_magnitude_pairs(
            Nsignals=Nsignals
        )
        all_parameters["tilt1"], all_parameters["tilt2"] = draw_spin_orientation_pairs(
            Nsignals=Nsignals
        )

        # Draw the azimuthal spin angle
        all_parameters["phi"] = np.arccos(np.random.uniform(-1, 1, Nsignals))


    # Draw the extrinsic parameters
    all_parameters.update(
        {
            "f_ref": np.ones(Nsignals) * f_ref,
            "ra": np.random.uniform(0, 2 * np.pi, Nsignals),
            "dec": np.arccos(np.random.uniform(-1, 1, Nsignals)) - np.pi/2,
            "inclination": np.arccos(np.random.uniform(-1, 1, Nsignals)),
            "polarization": np.random.uniform(0, 2 * np.pi, Nsignals),
            "phase": np.random.uniform(0, 2 * np.pi, Nsignals),
            "f_low": np.full(Nsignals, float(f_low)),
        }
    )
    '''# MLO: Fixed sky location to enforce SNR
    all_parameters["ra"] = 2.15 * np.ones(Nsignals)
    all_parameters["dec"] = -0.45 * np.ones(Nsignals)
    all_parameters["inclination"] = 2 * np.ones(Nsignals)
    all_parameters["polarization"] = 5.9 * np.ones(Nsignals)'''

    #all_parameters["geocent_time"] = get_spaced_time_array(signals_start, signals_end, Nsignals, chunk_size=chunk_size, sep=1, is_BNS=is_BNS)
    # MLO: Change these limits for different chunk sizes
    # MLO: Lower limit ~= 2 times the max waveform length which should be half the chunk size
    all_parameters["geocent_time"] = signals_start + np.random.uniform(chunk_size / 2 + 0.5, chunk_size - 3.5, Nsignals)

    # Apply data time sampling offset to times
    all_parameters['geocent_time'] = apply_data_offset(times=all_parameters['geocent_time'])

    # Calculate other spin and mass parameters
    all_parameters = get_extra_parameters(
        all_parameters=all_parameters, detectors=detectors
    )

    # Convert arrays into nested dictionary
    all_parameters_dict = _get_parameters_dict(all_parameters, Nsignals)
    return all_parameters_dict


def _get_parameters_dict(all_parameters, Nsignals):
    """Creates a nested dictionary assigning each drawn parameter
    from an array to a single signal.
    """

    all_parameters.update(
        {"overlaps_with": np.array([None for i in range(Nsignals)])}
    )
    keys = list(all_parameters.keys())

    """ We don't need this unless nested dict transpose is possible
    (I should really replace all of this with some kind of table)
    # Create an ndarray from the input parameter dictionary
    all_parameters_array = np.zeros(Nsignals)
    for key, value in all_parameters.items():
        all_parameters = np.vstack([all_parameters, value])
    all_parameters = all_parameters[1:].T"""

    # Create nested signals dictionary
    # Using a nested for loop for now
    all_parameters_dict = {}
    for idx in range(Nsignals):
        name = f"signal{idx}"
        all_parameters_dict[name] = {}
        for key in keys:
            all_parameters_dict[name][key] = all_parameters[key][idx]

    return all_parameters_dict


def overlap_signals(all_parameters_dict, Nsignals, fraction_overlap, f_low=20):
    """Overlaps the first `Nsignals * fraction_overlap` signals with
    the last `Nsignals * fraction_overlap` signals. The overlap is
    calculated by modifying the merger time of the second signal to
    fall within the visible duration of the first signal.
    """

    Noverlap = int(Nsignals * fraction_overlap)
    up_indices = np.arange(Noverlap).astype(int)
    down_indices = (Nsignals + up_indices).astype(int)
    for idx_up, idx_down in tqdm(
        zip(up_indices, down_indices),
        total=Noverlap,
        desc="Calculating overlaps",
    ):
        '''# Set the new merger time of Signal B
        duration = get_signal_duration(
            all_parameters_dict[f"signal{idx_up}"]["mass1"],
            all_parameters_dict[f"signal{idx_up}"]["mass2"],
            f_low=f_low,
        )
        # MLO reduce duration to stop non-overlaps
        duration -= duration / 8'''
        end = all_parameters_dict[f"signal{idx_up}"]["geocent_time"]
        # Draw time shift from a uniform distribution
        random_shift = np.random.uniform(0, 2, 1)

        all_parameters_dict[f"signal{idx_down}"][
            "geocent_time"
        ] = apply_data_offset(end - random_shift)
        #] = apply_data_offset(end - duration + random_shift)

        # Add note as to the overlap of the two signals:
        all_parameters_dict[f"signal{idx_up}"][
            "overlaps_with"
        ] = f"signal{idx_down}"
        all_parameters_dict[f"signal{idx_down}"][
            "overlaps_with"
        ] = f"signal{idx_up}"
    return all_parameters_dict


def re_overlap_signals(all_parameters_dict, fraction_overlap=1):
    """Recalculates times for all SignalB's under different conditions,
    without messing up the rest of the data
    """

    Nsignals = len(all_parameters_dict.keys()) // 2
    Noverlap = int(Nsignals * fraction_overlap)
    up_indices = np.arange(Noverlap).astype(int)
    down_indices = (Nsignals + up_indices).astype(int)
    for idx_up, idx_down in tqdm(
        zip(up_indices, down_indices),
        total=Noverlap,
        desc="Calculating overlaps",
    ):
        # Set the new merger time of Signal B
        end = all_parameters_dict[f"signal{idx_up}"]["geocent_time"]
        # Draw time shift from a uniform distribution
        # We want some runs to have closer merger times such that BNS overlaps
        # are not too separate in time. This should allow for half the pairs in
        # the significant bias region, and half in the weak bias region
        if idx_up < Nsignals / 2:
            random_shift = np.random.uniform(0, 1e-2, 1)
        else:
            random_shift = np.random.uniform(1e-2, 2, 1)

        all_parameters_dict[f"signal{idx_down}"][
            "geocent_time"
        ] = apply_data_offset(end - random_shift)

    # Recalculate times
    all_parameters_dict = add_detector_times(all_parameters_dict, detectors=["H1", "L1", "V1"])
    return all_parameters_dict

def save_signals_dict(signals_dict, signal_catalog_file):
    if os.path.isfile(signal_catalog_file):
        name = signal_catalog_file.rsplit(".", 1)
        name = ".".join([name[0] + "_old", name[1]])
        os.rename(signal_catalog_file, name)

    with open(signal_catalog_file, "w") as file:
        json.dump(signals_dict, file, indent=4)

    return None


def append_parameter_array(name, array, signals_dict):
    for idx, key in enumerate(signals_dict.keys()):
        signals_dict[key][name] = array[idx]
    return signals_dict


def get_network_SNR(parameters, detectors):
    SNRs = np.empty(len(detectors))
    for idx, detector in enumerate(detectors):
        SNRs[idx] = parameters[f"{detector}_SNR"]
    return np.sqrt(np.square(SNRs).sum())


def get_network_SNRs(signals_dict, detectors):
    for key, value in signals_dict.items():
        signals_dict[key]["network_SNR"] = get_network_SNR(parameters=value, detectors=detectors)
    return signals_dict
