import h5py
import argparse

import numpy as np


def valid_filename(filename):
    if not filename.endswith(".h5"):
        raise argparse.ArgumentTypeError(
            f"Filename must have'.h5' extension: '{filename}'"
        )
    return filename


def get_date_and_duration(file_path):
    with h5py.File(file_path, "r") as f:
        dateRaw = f["/Data"].attrs.get("Date", "")
        if isinstance(dateRaw, bytes):
            dateStr = dateRaw.decode("utf-8")
        print(f"Filename '{file_path}' recorded {dateStr}")

        duration = f["/Data/Recording_0"].attrs.get("Duration", None)  # Microseconds
        if duration is None:
            raise Exception(f"Duration attribute not found in file {file_path}")
        print(f"Duration: ~{duration / 60_000_000:.2f} minutes")
        return dateStr, duration


def print_structure_with_data(item, name, indent="", max_lines=5, max_width=80):
    """
    Recursive function to display the structure of an HDF5 item,
    including a preview of non-numerical data.

    Args:
        item: The HDF5 item (Group, Dataset, Attribute).
        name (str): The name of the item.
        indent (str): The current indentation level.
        max_lines (int): Maximum number of lines to display for data preview.
        max_width (int): Maximum width of each line in the data preview.
    """
    print(f"{indent}- {name}", end="")
    if isinstance(item, h5py.Group):
        print(" (Group)")
        for key in item.keys():
            print_structure_with_data(
                item[key], key, indent + "  ", max_lines, max_width
            )
    elif isinstance(item, h5py.Dataset):
        print(f" (Dataset: shape={item.shape}, dtype={item.dtype})")
        if np.issubdtype(item.dtype, np.str_) or item.dtype == np.object_:
            print(f"{indent}  Data Preview:")
            data = item[:max_lines]  # Read a limited number of rows
            for row in data:
                formatted_row = str(row)
                if len(formatted_row) > max_width:
                    print(f"{indent}    {formatted_row[:max_width]}...")
                else:
                    print(f"{indent}    {formatted_row}")
            if len(item) > max_lines:
                print(f"{indent}    ...")
    elif isinstance(item, h5py.Attribute):
        print(f" (Attribute: name={item.name}, value={item.value})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Print structure of an HDF5 (.h5) file."
    )
    parser.add_argument("filename", type=valid_filename, help="Path to the .h5 file")
    args = parser.parse_args()
    try:
        with h5py.File(args.filename, "r") as hf:
            for name in hf.keys():
                print_structure_with_data(hf[name], name)
    except FileNotFoundError:
        print(f"Error: File not found at {args.filename}")
    except Exception as e:
        print(f"Error: {e}")
