#!/usr/bin/env python
#
# With a .h5 file exported from Multi Channel Systems DataManager
# extract the spikes, and locate the precise timestamp of each spike.
#
import argparse
from h5_tools import print_hdf5_structure, valid_filename


def main():
    parser = argparse.ArgumentParser(description="Process a single HDF5 (.h5) file.")
    parser.add_argument("filename", type=valid_filename, help="Path to the .h5 file")
    args = parser.parse_args()

    print_hdf5_structure(args.filename)


if __name__ == "__main__":
    main()
