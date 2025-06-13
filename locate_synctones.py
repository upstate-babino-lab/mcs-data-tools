#!/usr/bin/env python
#
# With a .h5 file exported from Multi Channel Systems DataManager
# extract the audio data from any analog stream channel containing "audio", 
# and locate the precise timestamp of each synctone.
#

import argparse
import h5py
import numpy as np
from scipy.signal import savgol_filter, find_peaks, resample
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from h5_tools import get_date_and_duration, valid_filename
from plot_data import plot_data
from utils import centered_moving_average, find_square_wave_steps
import os, csv


def find_audio_channel(file_path, desiredLabelStr="audio"):
    """
    Search through all analog streams to find a channel containing the desired label string.
    Returns the group path and channel number if found.
    """
    with h5py.File(file_path, "r") as f:
        # Look for all analog streams
        recording_group = "/Data/Recording_0/AnalogStream"
        
        if recording_group not in f:
            raise Exception(f"Recording group {recording_group} not found in file")
        
        analog_stream_group = f[recording_group]
        
        # Search through all streams
        for stream_name in analog_stream_group.keys():
            if stream_name.startswith("Stream_"):
                stream_path = f"{recording_group}/{stream_name}"
                print(f"Checking {stream_path}...")
                
                try:
                    group = f[stream_path]
                    
                    # Check if this stream has the required components
                    if "InfoChannel" not in group or "ChannelData" not in group:
                        print(f"  Skipping {stream_path} - missing InfoChannel or ChannelData")
                        continue
                    
                    # Check the stream label
                    labelRaw = group.attrs.get("Label", "")
                    if isinstance(labelRaw, bytes):
                        labelStr = labelRaw.decode("utf-8")
                    else:
                        labelStr = str(labelRaw)
                    
                    if "Analog Data" not in labelStr:
                        print(f"  Skipping {stream_path} - not analog data")
                        continue
                    
                    # Search for the desired channel in this stream
                    info = group["InfoChannel"][()]
                    nAnalogChannels = info.shape[0]
                    print(f"  Found {nAnalogChannels} analog channels")
                    
                    for i in range(nAnalogChannels):
                        channel_label = info["Label"][i].decode("utf-8")
                        print(f"    Channel {i}: '{channel_label}'")
                        if desiredLabelStr.lower() in channel_label.lower():
                            print(f"  Found audio channel {i} with label '{channel_label}' in {stream_path}")
                            return stream_path, i
                            
                except Exception as e:
                    print(f"  Error checking {stream_path}: {e}")
                    continue
        
        raise Exception(f"Unable to find any channel with label containing '{desiredLabelStr}' in any analog stream")


def get_analog_data(file_path, desiredLabelStr="audio"):
    stream_path, labeledChannelNumber = find_audio_channel(file_path, desiredLabelStr)
    
    with h5py.File(file_path, "r") as f:
        group = f[stream_path]
        
        info = group["InfoChannel"][()]
        label = info["Label"][labeledChannelNumber].decode("utf-8")
        print(f"Using channel {labeledChannelNumber} with label = '{label}' from {stream_path}")

        microseconds_between_samples = info["Tick"][labeledChannelNumber]
        data_rate = round(1_000_000 / microseconds_between_samples)

        analog_data = group["ChannelData"][labeledChannelNumber, :]
        print("Analog data shape:", analog_data.shape)
        duration_seconds = len(analog_data) / data_rate
        print(
            f"{len(analog_data)} samples @{data_rate}hz"
            + f" duration={round(duration_seconds)} seconds ={duration_seconds / 60:.2f} minutes"
        )
        return (
            analog_data.reshape(-1, 1),  # Reshape to single-column 2D array
            data_rate,
        )


def locate_synctones(file_path, do_plot=False):
    audio_data, _ = get_analog_data(file_path, "audio")
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(audio_data)
    centered = audio_data - np.mean(audio_data)
    squared = scaler.fit_transform(centered**2)
    smoothed = scaler.fit_transform(
        savgol_filter((centered**2).flatten(), 1800, 3).reshape(audio_data.shape)
    )
    peak_indices, _ = find_peaks(smoothed.flatten(), height=0.5, distance=2000)
    if do_plot:
        min = peak_indices[0] - 2000
        max = peak_indices[0] + 2000
        peaks_in_range = peak_indices[(peak_indices > min) & (peak_indices < max)]
        plot_data(
            [scaled, squared, smoothed],
            min,
            max,
            ["scaled", "squared", "smoothed"],
            peaks_in_range - min,
        )

    diffs = (10_000 - np.diff(peak_indices)) / 10  # Diffs in milliseconds
    mean_value = np.mean(diffs)
    min_value = np.min(diffs)
    max_value = np.max(diffs)
    std_dev = np.std(diffs)
    print(
        f"Millisecond diffs mean: {mean_value:.2f} min: {min_value} max: {max_value} stdDev: {std_dev:.2f}"
    )
    return peak_indices / 10_000  # Timestamps in seconds


def main():
    parser = argparse.ArgumentParser(description="Process a single HDF5 (.h5) file.")
    parser.add_argument("filename", type=valid_filename, help="Path to the .h5 file")
    parser.add_argument("-p", "--plot", action="store_true", help="Plot start of data")

    args = parser.parse_args()
    get_date_and_duration(args.filename)

    np.set_printoptions(suppress=True)  # Suppress scientific notation

    synctone_timestamps = locate_synctones(args.filename, args.plot)
    print(f"Found {len(synctone_timestamps)} synctones")
    print(synctone_timestamps)

    # Save timestamps to CSV
    base_filename = os.path.splitext(args.filename)[0]
    csv_filename = f"{base_filename}_synctones.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Synctone Timestamp (s)'])
        for sync in synctone_timestamps:
            writer.writerow([sync])
    print(f"\nSynctone timestamps saved to: {csv_filename}")


if __name__ == "__main__":
    main()
