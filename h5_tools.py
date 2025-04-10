import h5py
import argparse


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


def display_group_contents(group, indent=0):
    print("  " * indent + f"Group: {group.name}")
    for name in group:
        item = group[name]
        print("  " * (indent + 1) + f"- {name}: {type(item)}")
    if group.attrs:
        print("  Attributes:")
        for attr_name, attr_value in group.attrs.items():
            print(f"    - {attr_name}: {attr_value}")

        if isinstance(item, h5py.Group):
            display_group_contents(item, indent + 2)


def print_hdf5_structure(filename):
    def visitor_func(name, obj):
        print(f"Name: {name}")
        print(f"  Type: {type(obj)}")
        if hasattr(obj, "attrs") and obj.attrs:
            print("  Attributes:")
            for key, value in obj.attrs.items():
                print(f"    - {key}: {value}")
        print("-" * 30)

    with h5py.File(filename, "r") as f:
        f.visititems(visitor_func)
