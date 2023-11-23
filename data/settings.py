import numpy as np

def adjust_observing_period(length, unit, chunk_size=4, observing_start_time=1260000000):
    # Converting observing period into seconds:
    possible_units = ["seconds", "minutes", "hours", "days"]
    if unit.lower() not in possible_units:
        print("This is not a valid time unit for the observing run length")
        print(f"Please use one of the following: {possible_units}")
        quit()
    conversions = [60, 60, 24]

    idx = 0
    while possible_units[idx] != unit.lower():
        length *= conversions[idx]
        idx += 1

    # Moving observing period beyond the minimum length
    # Note: This should actually be 518s. But I don't like that number
    min_observing_period = 1024
    if length < min_observing_period:
        print("Observing period too short to run in PyCBC")
        print(f"Setting observing period to {min_observing_period}")
        length = min_observing_period

    # Make sure observing period is an integer and divisible by the chunk size
    additional_time = chunk_size - (length % chunk_size)
    if additional_time not in [0, chunk_size]:
        print(f"Adding {int(additional_time)}s to requested observing period.")
        print(
            "This rounds the observing run up to an integer number of 4s segments"
        )
        length = int(length + additional_time)

    return observing_start_time, observing_start_time + length

def get_Nsignals(length, arbitrary_signal_rate):
    Nsignals_mean = arbitrary_signal_rate * length
    if Nsignals_mean <= 1:
        Nsignals = 1
    else:
        Nsignals_std = Nsignals_mean / 10 
        Nsignals = int(
            np.floor(np.random.normal(Nsignals_mean, Nsignals_std, 1)[0]) / 4
        )
    return Nsignals
