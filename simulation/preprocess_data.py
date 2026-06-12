import argparse
import base64
import os
import pickle

import h5py
import numpy as np
import torch
import yaml
from PIL import Image, UnidentifiedImageError
from torchvision.transforms import Resize, ToTensor, transforms
from tqdm import tqdm


# Custom constructor for numpy scalar objects in the YAML
def numpy_scalar_constructor(loader, node):
    # Decode the binary data directly
    binary_data = loader.construct_scalar(node.value[1])
    decoded = base64.b64decode(binary_data)
    # Convert to float using numpy
    return float(np.frombuffer(decoded, dtype=np.float64)[0])  # Assuming dtype is float64


def preprocess_and_store(root_dir, output_file, image_size=(32, 32), include_images=True):
    # Initialize the transformation
    transform = Resize(image_size)
    # Open the unsuccesful_runs.txt file
    with open(os.path.join(root_dir, 'unsuccesful_runs.txt'), 'r') as f:
        skip_indices = f.read().splitlines()

    # Convert the indices to integers
    skip_indices = [int(i) for i in skip_indices]

    # Rest of the code...
    with h5py.File(output_file, 'w') as h5f:
        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if os.path.isdir(folder_path):
                # Extract the index from the folder name
                index = int(folder_name.split('_')[-1])

                # Skip the folder if the index is in the list of indices
                if index in skip_indices:
                    print(f"Skipping {folder_name}")
                    continue

                # Load vectors
                with open(os.path.join(folder_path, 'vectors.pk'), 'rb') as f:
                    vectors = pickle.load(f)

                # Create a group for this sequence in the HDF5 file
                grp = h5f.create_group(folder_name)

                # Store vectors
                grp.create_dataset('vectors', data=vectors)

                # Process and store images
                for i in range(len(vectors)):
                    img_path = os.path.join(folder_path, f'im{i}.png')
                    img = Image.open(img_path)
                    img = transform(img)
                    img = ToTensor()(img)

                    grp.create_dataset(f'image_{i}', data=img.numpy())

                # Store the label
                # Check if the config.yaml file exists
                config_file = os.path.join(folder_path, 'config.yaml')
                if os.path.exists(config_file):
                    # Open the config.yaml file
                    with open(config_file, 'r') as f:
                        # config = yaml.safe_load(f)
                        config = yaml.load(f, Loader=yaml.FullLoader)
                    # Check if "lump" field exists in "model"
                    if config['breast_model']['lump']['include_lump']:
                        label = [1.0]
                    else:
                        label = [0.0]
                else:
                    raise FileNotFoundError(f"Config file not found: {config_file}")

                # Store the label
                grp.create_dataset('label', data=label)

                # Load and process the model imaging image
                model_imaging_path = os.path.join(folder_path, 'model_imaging.png')
                model_imaging = Image.open(model_imaging_path)
                model_imaging = transform(model_imaging)
                model_imaging = ToTensor()(model_imaging)

                # Add the model imaging image to the dataset
                grp.create_dataset('model_imaging', data=model_imaging.numpy())

                # print(f"Processed and stored data for {folder_name}")


def preprocess_and_store_v1(root_dir, output_file, image_size=(32, 32), max_index=1e6):
    # Initialize the transformation
    transform = Resize(image_size)
    # Open the unsuccesful_runs.txt file
    with open(os.path.join(root_dir, 'unsuccesful_runs.txt'), 'r') as f:
        skip_indices = f.read().splitlines()

    # Convert the indices to integers
    skip_indices = [int(i) for i in skip_indices]

    # Rest of the code...
    with h5py.File(output_file, 'w') as h5f:
        h5f.attrs['version'] = 1
        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if os.path.isdir(folder_path):
                # Extract the index from the folder name
                index = int(folder_name.split('_')[-1])

                # Skip the folder if the index is in the list of indices
                if index in skip_indices:
                    print(f"Skipping {folder_name}")
                    continue

                if index > max_index:
                    print(f"Skipping {folder_name} because index > {max_index}")
                    continue

                # Load vectors
                # with open(os.path.join(folder_path, 'vectors.pk'), 'rb') as f:
                #     vectors = pickle.load(f)

                # Create a group for this sequence in the HDF5 file
                grp = h5f.create_group(folder_name)

                # Store vectors
                for trajectory_folder in os.listdir(folder_path):
                    trajectory_path = os.path.join(folder_path, trajectory_folder)
                    if trajectory_folder.startswith('trajectory_'):
                        i = int(trajectory_folder.split('_')[-1])
                    else:
                        continue
                    if os.path.isdir(trajectory_path):
                        with open(os.path.join(trajectory_path, 'vectors.pk'), 'rb') as f:
                            vectors = pickle.load(f)
                        grp.create_dataset(f'vectors_{i}', data=vectors)

                        # Process and store images
                        for j in range(len(vectors)):
                            img_path = os.path.join(trajectory_path, f'im{j}.png')
                            img = Image.open(img_path)
                            img = transform(img)
                            img = ToTensor()(img)

                            grp.create_dataset(f'image_{i}_{j}', data=img.numpy())

                # Store the label
                # Check if the config.yaml file exists
                config_file = os.path.join(folder_path, 'config.yaml')
                if os.path.exists(config_file):
                    # Open the config.yaml file
                    with open(config_file, 'r') as f:
                        config = yaml.safe_load(f)
                    # Check if "lump" field exists in "model"
                    if config['breast_model']['lump']['include_lump'] is True:
                        label = [1.0]
                    else:
                        label = [0.0]
                else:
                    raise FileNotFoundError(f"Config file not found: {config_file}")

                # Store the label
                grp.create_dataset('label', data=label)

                # Load and process the model imaging image
                model_imaging_path = os.path.join(folder_path, 'trajectory_0', 'model_imaging.png')
                model_imaging = Image.open(model_imaging_path)
                model_imaging = transform(model_imaging)
                model_imaging = ToTensor()(model_imaging)

                # Add the model imaging image to the dataset
                grp.create_dataset('model_imaging', data=model_imaging.numpy())

                # Count the number of folders in the group
                num_trajs = len([f for f in os.listdir(folder_path) if
                                 f.startswith('trajectory_') and os.path.isdir(os.path.join(folder_path, f))])
                grp.attrs['num_trajs'] = num_trajs

                # Count the number of images in each folder
                for trajectory_folder in os.listdir(folder_path):
                    if trajectory_folder.startswith('trajectory_'):
                        i = int(trajectory_folder.split('_')[-1])
                    else:
                        continue
                    trajectory_path = os.path.join(folder_path, trajectory_folder)
                    if os.path.isdir(trajectory_path):
                        num_images = len(
                            [f for f in os.listdir(trajectory_path) if f.startswith('im') and f.endswith('.png')])
                        grp.attrs[f'num_images_{i}'] = num_images

                # print(f"Processed and stored data for {folder_name}")


def preprocess_and_store_v2(root_dir, output_file, image_size=(32, 32), imaging_image_size=(128, 128), max_index=1e6,
                            only_probe=False, remove_axis=False, only_lump=False, add_radius=False, no_images=False):
    # Initialize the transformation
    transform = transforms.Compose([
        Resize(image_size),
        transforms.Grayscale()
    ])

    imaging_transform = Resize(imaging_image_size)

    # Open the unsuccesful_runs.txt file
    # with open(os.path.join(root_dir, 'unsuccesful_runs.txt'), 'r') as f:
    #     skip_indices = f.read().splitlines()

    skip_indices = []

    # Convert the indices to integers
    skip_indices = [int(i) for i in skip_indices]

    # Rest of the code...
    with h5py.File(output_file, 'w') as h5f:
        h5f.attrs['version'] = 1
        for folder_name in tqdm(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder_name)
            if os.path.isdir(folder_path):
                # Extract the index from the folder name
                index = int(folder_name.split('_')[-1])

                # Skip the folder if the index is in the list of indices
                if index in skip_indices:
                    print(f"Skipping {folder_name}")
                    continue

                if index > max_index:
                    print(f"Skipping {folder_name} because index > {max_index}")
                    continue

                # Store the label
                # Check if the config.yaml file exists
                config_file = os.path.join(folder_path, 'config.yaml')
                if os.path.exists(config_file):
                    # Open the config.yaml file
                    with open(config_file, 'r') as f:
                        config = yaml.safe_load(f)
                    # Check if "lump" field exists in "model"
                    if config['breast_model']['lump']['include_lump'] is True:
                        label = [1.0]
                    else:
                        label = [0.0]
                else:
                    raise FileNotFoundError(f"Config file not found: {config_file}")

                radius = config['breast_model']['radius']

                if only_lump and label[0] == 0.0:
                    print(f"Skipping {folder_name} because only_lump is set to True and the label is 0.0")
                    continue

                # Load vectors
                # with open(os.path.join(folder_path, 'vectors.pk'), 'rb') as f:
                #     vectors = pickle.load(f)

                # Create a group for this sequence in the HDF5 file
                grp = h5f.create_group(folder_name)
                try:
                    # Store vectors
                    for trajectory_folder in os.listdir(folder_path):
                        trajectory_path = os.path.join(folder_path, trajectory_folder)
                        if trajectory_folder.startswith('trajectory_'):
                            i = int(trajectory_folder.split('_')[-1])
                        else:
                            continue
                        if os.path.isdir(trajectory_path):
                            with open(os.path.join(trajectory_path, 'vectors.pk'), 'rb') as f:
                                vectors = pickle.load(f)
                            grp.create_dataset(f'vectors_{i}', data=vectors)

                            # Process and store images
                            if not no_images:
                                for j in range(len(vectors)):
                                    img_path = os.path.join(trajectory_path, f'im{j}.png')
                                    img = Image.open(img_path)
                                    if only_probe:
                                        img = img.split()[2]
                                    if remove_axis:
                                        img = img.crop((100, 100, 500, 400))

                                    img = transform(img)
                                    img = ToTensor()(img)

                                    grp.create_dataset(f'image_{i}_{j}', data=(img * 255).to(torch.uint8).numpy())
                except FileNotFoundError:
                    print(f"Skipping {folder_name} because file not found")
                    del h5f[folder_name]

                # Store the label
                grp.create_dataset('label', data=label)

                if add_radius:
                    grp.create_dataset('radius', data=radius)

                # Load and process the model imaging image
                model_imaging_path = os.path.join(folder_path, 'trajectory_0', 'model_imaging.png')
                model_imaging = Image.open(model_imaging_path)
                model_imaging = imaging_transform(model_imaging)
                model_imaging = ToTensor()(model_imaging)

                # Add the model imaging image to the dataset
                grp.create_dataset('model_imaging', data=model_imaging.numpy())

                # Count the number of folders in the group
                num_trajs = len([f for f in os.listdir(folder_path) if
                                 f.startswith('trajectory_') and os.path.isdir(os.path.join(folder_path, f))])
                grp.attrs['num_trajs'] = num_trajs

                # Count the number of images in each folder
                for trajectory_folder in os.listdir(folder_path):
                    if trajectory_folder.startswith('trajectory_'):
                        i = int(trajectory_folder.split('_')[-1])
                    else:
                        continue
                    trajectory_path = os.path.join(folder_path, trajectory_folder)
                    if os.path.isdir(trajectory_path):
                        num_images = len(
                            [f for f in os.listdir(trajectory_path) if f.startswith('im') and f.endswith('.png')])
                        grp.attrs[f'num_images_{i}'] = num_images

                print(f"Processed and stored data for {folder_name}")


def preprocess_and_store_change_detection(root_dir, output_file, image_size=(32, 32), imaging_image_size=(128, 128),
                                          max_index=1e6,
                                          only_probe=False, remove_axis=False, only_lump=False, add_radius=False,
                                          no_images=False):
    yaml.add_constructor("tag:yaml.org,2002:python/object/apply:numpy.core.multiarray.scalar", numpy_scalar_constructor)
    # Initialize the transformation
    transform = transforms.Compose([
        Resize(image_size),
        transforms.Grayscale()
    ])

    imaging_transform = Resize(imaging_image_size)

    # Open the unsuccesful_runs.txt file
    with open(os.path.join(root_dir, 'unsuccesful_runs.txt'), 'r') as f:
        skip_indices = f.read().splitlines()

    # Convert the indices to integers
    skip_indices = [int(i) for i in skip_indices]

    # Rest of the code...
    with h5py.File(output_file, 'w') as h5f:
        h5f.attrs['version'] = 3
        for folder_name in tqdm(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder_name)
            if os.path.isdir(folder_path):
                # Extract the index from the folder name
                index = int(folder_name.split('_')[-1])

                # Skip the folder if the index is in the list of indices
                if index in skip_indices:
                    print(f"Skipping {folder_name}")
                    continue

                if index > max_index:
                    print(f"Skipping {folder_name} because index > {max_index}")
                    continue

                # load changing_lump.txt
                with open(os.path.join(folder_path, 'changing_lump.txt'), 'r') as f:
                    changing_lump = f.read().splitlines()
                changing_lump = True if changing_lump[0] == 'True' else False

                # Store the label
                # Check if the config.yaml file exists
                config_file = os.path.join(folder_path, 'trial_0', 'config.yaml')
                if os.path.exists(config_file):
                    # Open the config.yaml file
                    with open(config_file, 'r') as f:
                        # config = yaml.safe_load(f)
                        config = yaml.load(f, Loader=yaml.FullLoader)
                    # Check if "lump" field exists in "model"
                    if config['breast_model']['lump']['include_lump'] is True:
                        if changing_lump is True:
                            label = [2.0]
                        else:
                            label = [1.0]
                    else:
                        label = [0.0]
                else:
                    raise FileNotFoundError(f"Config file not found: {config_file}")

                if only_lump and label[0] == 0.0:
                    print(f"Skipping {folder_name} because only_lump is set to True and the label is 0.0")
                    continue

                # Create a group for this sequence in the HDF5 file
                grp = h5f.create_group(folder_name)
                try:
                    for trial_folder in sorted(os.listdir(folder_path)):
                        if trial_folder.startswith('trial_'):
                            k = int(trial_folder.split('_')[-1])
                        else:
                            continue
                        trial_path = os.path.join(folder_path, trial_folder)
                        config_file = os.path.join(trial_path, 'config.yaml')
                        if os.path.exists(config_file):
                            # Open the config.yaml file
                            with open(config_file, 'r') as f:
                                config = yaml.load(f, Loader=yaml.FullLoader)
                            radius = config['breast_model']['radius']
                        else:
                            raise FileNotFoundError(f"Config file not found: {config_file}")
                        if add_radius:
                            grp.create_dataset(f'radius_{k}', data=[radius])

                        # Load and process the model imaging image
                        model_imaging_path = os.path.join(trial_path, 'trajectory_0', 'model_imaging.png')
                        model_imaging = Image.open(model_imaging_path)
                        model_imaging = imaging_transform(model_imaging)
                        model_imaging = ToTensor()(model_imaging)

                        # Add the model imaging image to the dataset
                        grp.create_dataset(f'model_imaging_{k}', data=model_imaging.numpy())
                        grp.create_dataset(f'lump_center_{k}', data=config['breast_model']['lump']['center'])
                        # Store vectors
                        num_trajs = 0
                        for trajectory_folder in sorted(os.listdir(trial_path)):
                            trajectory_path = os.path.join(trial_path, trajectory_folder)
                            if trajectory_folder.startswith('trajectory_'):
                                i = int(trajectory_folder.split('_')[-1])
                            else:
                                continue
                            num_trajs += 1
                            if os.path.isdir(trajectory_path):
                                with open(os.path.join(trajectory_path, 'vectors.pk'), 'rb') as f:
                                    vectors = pickle.load(f)
                                grp.create_dataset(f'vectors_{k}_{i}', data=vectors)
                                if not no_images:
                                    # Process and store images
                                    for j in range(len(vectors)):
                                        img_path = os.path.join(trajectory_path, f'im{j}.png')
                                        img = Image.open(img_path)
                                        if only_probe:
                                            img = img.split()[2]
                                        if remove_axis:
                                            img = img.crop((100, 100, 500, 400))

                                        img = transform(img)
                                        img = ToTensor()(img)

                                        grp.create_dataset(f'image_{k}_{i}_{j}',
                                                           data=(img * 255).to(torch.uint8).numpy())

                                grp.attrs[f'num_images_{k}_{i}'] = len(vectors)

                        grp.attrs[f'num_trajs_{k}'] = num_trajs

                # except FileNotFoundError and UnidentifiedImageError
                except (FileNotFoundError, UnidentifiedImageError):
                    print(f"Skipping {folder_name} because file not found or bad image")
                    del h5f[folder_name]

                # Store the label
                grp.create_dataset('label', data=label)

                # print(f"Processed and stored data for {folder_name}")


def multiple_h5s(root_dir, final_output_file, max_folder_index, kwargs, single_h5_func):
    # list all folders in root_dir that are under max_folder_index
    folders = [f for f in os.listdir(root_dir) if
               os.path.isdir(os.path.join(root_dir, f)) and int(f) <= max_folder_index]
    print(f"Processing folders: {folders}")
    # create a list of output files
    output_files = [os.path.join(root_dir, f'{f}.h5') for f in folders]
    # For each folder, call the single_process_func with the corresponding output file in a separate process using the multiprocessing module
    import multiprocessing
    processes = []
    for folder, output_file in zip(folders, output_files):
        # Check if the output file already exists
        if os.path.exists(output_file):
            print(f"Output file {output_file} already exists. Skipping...")
            continue
        p = multiprocessing.Process(target=single_h5_func, args=(os.path.join(root_dir, folder), output_file),
                                    kwargs=kwargs)
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    # After all processes are done, combine the output files into a single output file
    with h5py.File(final_output_file, 'w') as h5f:
        h5f.attrs['version'] = 4
        for folder, output_file in zip(folders, output_files):
            with h5py.File(output_file, 'r') as h5f_in:
                for key in h5f_in.keys():
                    h5f_in.copy(key, h5f)
            os.remove(output_file)
        print(f"Total number of groups: {len(h5f.keys())}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess and store data')
    parser.add_argument('--root_dir', type=str, default='/path/to/data',
                        help='Root directory of the data')
    parser.add_argument('--output_file', type=str, default='/path/to/data.h5',
                        help='Output file path')
    parser.add_argument('--max_index', type=int, default=1999, help='Maximum index value')
    parser.add_argument('--version', type=int, default=4, help='Version of the dataset')
    parser.add_argument('--only_probe', action='store_true', help='Only store red channel')
    parser.add_argument('--remove_axis', action='store_true', help='Remove axis from the image')
    parser.add_argument('--only_lump', action='store_true', help='Only store lump data')
    parser.add_argument('--add_radius', action='store_true', help='Add radius to the data')
    parser.add_argument('--no_images', action='store_true', help='Do not store images')

    args = parser.parse_args()

    if args.version == 1:
        preprocess_and_store_v1(args.root_dir, args.output_file, max_index=args.max_index)
    elif args.version == 2:
        preprocess_and_store_v2(args.root_dir, args.output_file, max_index=args.max_index, only_probe=args.only_probe,
                                remove_axis=args.remove_axis, only_lump=args.only_lump, add_radius=args.add_radius,
                                no_images=args.no_images)
    elif args.version == 3:
        preprocess_and_store_change_detection(args.root_dir, args.output_file)
    elif args.version == 4:
        multiple_h5s(args.root_dir, args.output_file, args.max_index,
                     {'only_probe': args.only_probe, 'remove_axis': args.remove_axis, 'only_lump': args.only_lump,
                      'add_radius': args.add_radius, 'no_images': args.no_images},
                     preprocess_and_store_change_detection)


if __name__ == '__main__':
    main()
