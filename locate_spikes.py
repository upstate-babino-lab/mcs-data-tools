#!/usr/bin/env python
#
# With a .h5 file exported from Multi Channel Systems DataManager
# extract the spikes, and locate the precise timestamp of each spike.
#
import argparse

import h5py
import numpy as np
from h5_tools import (
    display_group_contents,
    get_date_and_duration,
    print_hdf5_structure,
    valid_filename,
)

SPIKE_GROUP = "Data/Recording_0/TimeStampStream/Stream_0"


def get_spike_data(file_path):
    with h5py.File(file_path, "r") as f:
        group = f[SPIKE_GROUP]
        for name in group:
            dataset = group[name]
            data = dataset[:]
            print(f"{name}: shape={data.shape} dtype={data.dtype}")
        return group["TimeStampEntity_109"][()].flatten()  # Microseconds


def main():
    parser = argparse.ArgumentParser(description="Process a single HDF5 (.h5) file.")
    parser.add_argument("filename", type=valid_filename, help="Path to the .h5 file")
    args = parser.parse_args()
    get_date_and_duration(args.filename)

    # print_hdf5_structure(args.filename)
    spike_data = get_spike_data(args.filename)
    spike_times_seconds = spike_data / 1_000_000
    diffs = np.diff(spike_times_seconds)
    np.set_printoptions(formatter={'float': '{:.2f}'.format})
    print("Seconds between spikes:")
    print(diffs)


if __name__ == "__main__":
    main()
