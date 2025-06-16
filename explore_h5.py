#!/usr/bin/env python

import h5py
import sys
# h5 Structure exploration 
def explore_h5_structure(filename, max_depth=4):
    print(f'\n=== Structure of {filename} ===')
    try:
        with h5py.File(filename, 'r') as f:
            def print_structure(name, obj, depth=0):
                if depth > max_depth:
                    return
                indent = '  ' * depth
                if isinstance(obj, h5py.Group):
                    print(f'{indent}{name}/ (Group)')
                    # detailed info support in AnalogStream handling module
                    if 'AnalogStream' in name:
                        for key in obj.keys():
                            print(f'{indent}  {key}/')
                            if key.startswith('Stream_'):
                                try:
                                    stream_obj = obj[key]
                                    if 'InfoChannel' in stream_obj:
                                        info = stream_obj['InfoChannel'][()]
                                        print(f'{indent}    InfoChannel: {len(info)} channels')
                                        for i, label_bytes in enumerate(info['Label']):
                                            label = label_bytes.decode('utf-8') if isinstance(label_bytes, bytes) else str(label_bytes)
                                            print(f'{indent}      Channel {i}: \'{label}\'')
                                    else:
                                        print(f'{indent}    No InfoChannel found')
                                except Exception as e:
                                    print(f'{indent}    Error reading stream: {e}')
                elif isinstance(obj, h5py.Dataset):
                    print(f'{indent}{name} (Dataset) shape: {obj.shape}')
            
            f.visititems(lambda name, obj: print_structure(name, obj, name.count('/')))
    except Exception as e:
        print(f'Error reading {filename}: {e}')

if __name__ == "__main__":
    if len(sys.argv) > 1:
        for filename in sys.argv[1:]:
            explore_h5_structure(filename)
    else:
        print("Usage: python explore_h5.py <filename1> [filename2] ...") 
