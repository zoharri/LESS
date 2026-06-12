import os
import tempfile
from typing import Optional, Tuple
import sys
from pathlib import Path
import pyrallis
import torch
import numpy as np
import torchvision
import random
import seaborn as sns

from matplotlib import pyplot as plt

import wandb
from torch.utils.data import Dataset, DataLoader

from datasets.real_dataset import RealDataset
from datasets.simulation_dataset import SimulationDataset
from models.downstream.predictors.downstream_model import DownstreamModel
from models.model_config import ModelConfig
from models.models_factory import RepModelFactory, DownstreamModelFactory
from models.representation_learning import RepresentationLearningModel
from train_rep_and_downstream_config import *


def split_trajs(tensor: torch.Tensor, traj_lengths: torch.Tensor) -> list:
    """
    Split a concatenated trajectory tensor back into individual trajectories.

    Args:
        tensor:       (total_steps, ...) where total_steps = sum(traj_lengths)
        traj_lengths: (T,) integer lengths of each trajectory

    Returns:
        List of T tensors, each of shape (traj_lengths[i], ...)
    """
    return list(torch.split(tensor, traj_lengths.long().tolist(), dim=0))


def pad_collate(batch):
    """
    Custom collate that pads variable-length locations/forces to the max length
    in the batch and returns a boolean padding_mask (True = valid position).

    Batch items are 8-tuples:
        (images, locations, forces, label, model_images, radius, exp_name, traj_props)
    where locations and forces have variable first dimension (total trajectory steps).

    Returns a 9-tuple where element 8 is the padding_mask (B, max_len).
    """
    images_list, locations_list, forces_list, labels_list, model_images_list, \
        radii_list, exp_names, traj_props_list = zip(*batch)

    max_len = max(loc.shape[0] for loc in locations_list)
    B = len(locations_list)

    padded_locations = torch.zeros(B, max_len, *locations_list[0].shape[1:])
    padded_forces = torch.zeros(B, max_len, *forces_list[0].shape[1:])
    padding_mask = torch.zeros(B, max_len, dtype=torch.bool)

    for i in range(B):
        n = locations_list[i].shape[0]
        padded_locations[i, :n] = locations_list[i]
        padded_forces[i, :n] = forces_list[i]
        padding_mask[i, :n] = True

    images = torch.stack(images_list)
    labels = torch.stack(labels_list)
    model_images = torch.stack(model_images_list)
    radii = torch.stack(radii_list)
    traj_props = torch.stack(traj_props_list)

    return images, padded_locations, padded_forces, labels, model_images, radii, list(exp_names), traj_props, padding_mask


def calculate_dataset_mean_std(loader, device):
    running_sum_locations = 0.0
    running_sum_forces = 0.0
    running_sum_locations_squared = 0.0
    running_sum_forces_squared = 0.0
    total_elements_locations = 0
    total_elements_forces = 0

    for item in loader:
        locations = item[1].double().to(device)  # double precision
        forces = item[2].double().to(device)  # double precision
        # item[8] is the padding_mask produced by pad_collate
        padding_mask = item[8].to(device) if len(item) > 8 else torch.ones(
            locations.shape[0], locations.shape[1], dtype=torch.bool, device=device)

        valid = padding_mask.double().unsqueeze(-1).unsqueeze(-1)  # (B, L, 1, 1)
        num_valid = padding_mask.sum().item()

        # Update running totals — only over valid (non-padded) positions
        running_sum_locations += (locations * valid).sum(dim=(0, 1))
        running_sum_locations_squared += ((locations ** 2) * valid).sum(dim=(0, 1))
        total_elements_locations += num_valid

        running_sum_forces += (forces * valid).sum(dim=(0, 1))
        running_sum_forces_squared += ((forces ** 2) * valid).sum(dim=(0, 1))
        total_elements_forces += num_valid

    mean_locations = running_sum_locations / total_elements_locations
    mean_forces = running_sum_forces / total_elements_forces

    var_locations = (running_sum_locations_squared / total_elements_locations) - mean_locations ** 2
    var_forces = (running_sum_forces_squared / total_elements_forces) - mean_forces ** 2

    # The clamp is a failsafe fo the sqrt later
    var_locations = torch.clamp(var_locations, min=0.0)
    var_forces = torch.clamp(var_forces, min=0.0)

    std_locations = var_locations.sqrt()
    std_forces = var_forces.sqrt()

    return mean_locations.float(), std_locations.float(), mean_forces.float(), std_forces.float()


def normalize_tensor(tensor, mean, std, min_value=-10, max_value=10):
    normalized_tensor = (tensor - mean) / (std + 1e-15).view(1, 1, tensor.shape[2], tensor.shape[3])
    normalized_tensor = torch.clamp(normalized_tensor, min_value, max_value)
    return normalized_tensor


def create_location_image(location, device, image_size=32, binary=False, location_std=0,
                          resize_image=torchvision.transforms.Resize((32, 32)), radius=-1,
                          include_radius=False):
    # create an image tensor with a gaussian blob at the location
    # location is betewen -1.3 and 1.3
    location_ = location.clone() + torch.randn_like(location) * location_std
    location_ = location_.to(device)
    location_[:, :, 1] = -location_[:, :, 1]
    x = torch.linspace(-1.3, 1.3, image_size * 2, device=device)
    y = torch.linspace(-1.3, 1.3, image_size * 2, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    Z = torch.exp(-((X.repeat([location_.shape[0], location_.shape[1], 1, 1]) - location_[:, :, 1].unsqueeze(
        2).unsqueeze(3)) ** 2 + (
                            Y.repeat([location_.shape[0], location_.shape[1], 1, 1]) - location_[:, :, 0].unsqueeze(
                        2).unsqueeze(3)) ** 2) / 0.1)
    if include_radius:
        binary = False

        # Create a mask for the half-circle and integrate into the image
        half_circle_mask = X.unsqueeze(0) ** 2 + Y.unsqueeze(0) ** 2 <= (radius.unsqueeze(1).unsqueeze(2) ** 2)
        Z[half_circle_mask.repeat(location_.shape[1], 1, 1, 1).permute(1, 0, 2, 3)] = Z[half_circle_mask.repeat(
            location_.shape[1], 1, 1, 1).permute(1, 0, 2, 3)] + 0.5  # Modify as needed for binary or non-binary output
    Z = resize_image(Z)

    return Z if not binary else (Z > 0.5).float()


def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return 100 * correct / total


def set_random_seed(seed):
    """
    Set random seed in PyTorch, NumPy, and random modules to ensure reproducibility.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def pop_s_and_next(lst, S):
    """
    Removes the first occurrence of the string `S` from the list `lst`,
    along with the element that immediately follows it, if such an element exists.

    Parameters:
    ----------
    lst : list of str
        The list of strings to modify in-place.
    S : str
        The target string to find and remove, along with the element after it.

    Returns:
    -------
    list of str
        The modified list with `S` and its following element removed, if `S` was found.

    Example:
    --------
    >>> pop_s_and_next(['a', 'b', 'c', 'd'], 'b')
    ['a', 'd']

    >>> pop_s_and_next(['x', 'y', 'z'], 'z')
    ['x', 'y']
    """
    if S in lst:
        idx = lst.index(S)
        if idx + 1 < len(lst):
            lst.pop(idx + 1)
        lst.pop(idx)
    return lst


def pop_item_with_substring(lst, S):
    """
    Removes the first element from the list `lst` that contains `S` as a substring.

    Parameters:
    ----------
    lst : list of str
        The list of strings to modify in-place.
    S : str
        The substring to search for in each element.

    Returns:
    -------
    list of str
        The modified list with the first matching element removed, if any.

    Example:
    --------
    >>> pop_item_with_substring(['apple', 'banana', 'grape'], 'ana')
    ['apple', 'grape']

    >>> pop_item_with_substring(['foo', 'bar', 'baz'], 'qux')
    ['foo', 'bar', 'baz']  # No change
    """
    for i, item in enumerate(lst):
        if S in item:
            lst.pop(i)
            break
    return lst


def create_train_test_dataset(dataset_config: DatasetConfig) -> tuple[Dataset, Dataset]:
    if dataset_config.real_data:
        full_dataset = RealDataset(dataset_config)
    else:
        full_dataset = SimulationDataset(dataset_config)

    train_indices, test_indices = full_dataset.get_split(dataset_config.split_type)
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

    train_size = len(train_dataset)
    test_size = len(test_dataset)
    if dataset_config.amount_of_data is not None and dataset_config.amount_of_data > 0:
        assert dataset_config.amount_of_data <= len(
            full_dataset), f"Amount of data {dataset_config.amount_of_data} is greater than the dataset size {len(full_dataset)}"
        percentage = dataset_config.amount_of_data / len(full_dataset)

        # randomly sample the dataset
        train_indices = np.random.choice(train_size, int(percentage * train_size), replace=False)
        test_indices = np.random.choice(test_size, min(max(int(percentage * test_size), 100), test_size), replace=False)
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        test_dataset = torch.utils.data.Subset(test_dataset, test_indices)

    return train_dataset, test_dataset


def create_rep_and_downstream_models(args: TrainConfig, checkpoint_dir: Path, save_config: bool = False) -> Tuple[RepresentationLearningModel, Optional[DownstreamModel]]:
    path_to_rep_and_downstream_models = checkpoint_dir

    rep_learning_config_path = os.path.join(path_to_rep_and_downstream_models, "rep_learning_model_config.yaml")
    if save_config:
        pyrallis.dump(args.models.rep_learning_model, open(rep_learning_config_path, 'w'))
    else:
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        pyrallis.dump(args.models.rep_learning_model, tmp)
        tmp.close()
        rep_learning_config_path = tmp.name
    rep_learning_model_config = ModelConfig(args.models.rep_learning_model.name, rep_learning_config_path)

    rep_model = RepModelFactory.generate_model(rep_learning_model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.models.rep_learning_model_checkpoint:
        rep_model.load_state_dict(torch.load(args.models.rep_learning_model_checkpoint, map_location=torch.device(device)), strict=False)
        print(f"Loaded model checkpoint from {args.models.rep_learning_model_checkpoint}")

        if args.optimizer.freeze_rep_learning_model:
            print("Freezing representation learning model parameters")
            for param in rep_model.parameters():
                param.requires_grad = False

    downstream_config_path = os.path.join(path_to_rep_and_downstream_models, "downstream_model_config.yaml")
    if save_config:
        pyrallis.dump(args.models.downstream_model, open(downstream_config_path, 'w'))
    else:
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        pyrallis.dump(args.models.downstream_model, tmp)
        tmp.close()
        downstream_config_path = tmp.name

    if not args.optimizer.disable_downstream_model:
        downstream_model_config = ModelConfig(args.models.downstream_model.name, downstream_config_path)
        downstream_model = DownstreamModelFactory.generate_model(downstream_model_config, rep_model.representation_size)
        if args.models.downstream_model_checkpoint:
            downstream_model.load_state_dict(torch.load(args.models.downstream_model_checkpoint, map_location=torch.device(device)), strict=True)
            print(f"Loaded model checkpoint from {args.models.downstream_model_checkpoint}")

            if args.optimizer.freeze_downstream_model:
                print("Freezing downstream model parameters")
                for param in downstream_model.parameters():
                    param.requires_grad = False
    else:
        downstream_model = None

    if save_config:
        full_config_path = os.path.join(path_to_rep_and_downstream_models, "train_config.yaml")
        pyrallis.dump(args, open(full_config_path, 'w'))

    return rep_model, downstream_model


def data_preprocess(locations, forces, model_images, train_mean_locations, train_std_locations,
                    train_mean_forces, train_std_forces, dont_norm_locations, relative_locations, zero_location,
                    zero_forces, shuffle_order, num_training_trajs, device, traj_lengths=None):
    """
    traj_lengths: optional (B, T) integer tensor of per-trajectory step counts used when
                  relative_locations=True to properly subtract each trajectory's start position.
    """
    locations = locations.float().to(device)  # Ensure vectors are in float
    forces = forces.float().to(device)  # Ensure vectors are in float

    forces = normalize_tensor(forces, train_mean_forces, train_std_forces)
    if not dont_norm_locations:
        locations = normalize_tensor(locations, train_mean_locations, train_std_locations)

    if relative_locations:
        if traj_lengths is not None:
            # Variable-length: subtract first position of each trajectory
            for b in range(locations.size(0)):
                start = 0
                for length in traj_lengths[b].long().tolist():
                    if length > 0:
                        locations[b, start:start + length] -= locations[b, start:start + 1]
                    start += length
        else:
            # Fixed-length fallback (assumes uniform trajectory lengths)
            stride = locations.size(1) // num_training_trajs
            locations_shift = locations[:, ::stride]
            locations_shift = locations_shift.repeat_interleave(stride, dim=1)
            locations = locations - locations_shift

    if zero_location:
        locations = 0 * locations
    if zero_forces:
        forces = 0 * forces
    if shuffle_order:
        rand_perm = torch.randperm(locations.size(1))
        locations = locations[:, rand_perm]
        forces = forces[:, rand_perm]

    model_images = model_images.float().to(device)
    return locations, forces, model_images


def parse_args() -> TrainConfig:
    """
    Parse command-line arguments into a TrainConfig object.

    If a checkpoint path is provided, this function attempts to load the
    corresponding `train_config.yaml` from the same directory and re-parses
    arguments from it, while removing any `--config_path` flags from sys.argv.

    Returns
    -------
    TrainConfig
        Parsed configuration object containing training hyperparameters.
    """
    args = pyrallis.parse(TrainConfig)

    if args.models.rep_learning_model_checkpoint:
        possible_config = Path(args.models.rep_learning_model_checkpoint).parent / "train_config.yaml"
        if os.path.exists(possible_config):
            print(f"Loading train config from {possible_config}")
            # Avoid parsing --config_path argument
            sys.argv = pop_s_and_next(sys.argv, "--config_path")
            sys.argv = pop_item_with_substring(sys.argv, "--config_path")
            args = pyrallis.parse(TrainConfig, config_path=possible_config)

    return args


def get_dataset_mean_std(args: DataProcessingConfig,
                         train_loader: DataLoader,
                         device: torch.device,
                         constant_force_norm: float,
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute or load dataset mean and standard deviation statistics.

    If a precomputed `.npz` file is found at `data_stats_load_path`, it is loaded.
    Otherwise, statistics are calculated from the given `train_loader` and saved
    to `save_path` if provided.

    Parameters
    ----------
    args : DataProcessingConfig,
        DataProcessingConfig object.
    train_loader : DataLoader,
        DataLoader for the training set, used to compute statistics if needed.
    device : torch.device
        The device on which tensors will be allocated.
    constant_force_norm: float,
        Constant force normalization factor.
    Returns
    -------
    tuple of torch.Tensor
        Mean and standard deviation tensors for locations and forces:
        (train_mean_locations, train_std_locations, train_mean_forces, train_std_forces)
    """
    print(f"Calculating dataset mean and std...")
    save_path = args.data_stats_save_path
    data_stats_load_path = args.data_stats_load_path

    if (not data_stats_load_path) or (not os.path.exists(data_stats_load_path)):
        train_mean_locations, train_std_locations, train_mean_forces, train_std_forces = calculate_dataset_mean_std(
            train_loader, device)

        stats = {
            "mean_locations": train_mean_locations.cpu().numpy(),
            "std_locations": train_std_locations.cpu().numpy(),
            "mean_forces": train_mean_forces.cpu().numpy(),
            "std_forces": train_std_forces.cpu().numpy(),
        }

        if save_path:
            np.savez(save_path, **stats)
            print(f"Dataset mean and std saved to {save_path}.npz")
    else:

        data_stats = np.load(data_stats_load_path)

        train_mean_locations = torch.from_numpy(data_stats["mean_locations"]).float().to(device)
        train_std_locations = torch.from_numpy(data_stats["std_locations"]).float().to(device)
        train_mean_forces = torch.from_numpy(data_stats["mean_forces"]).float().to(device)
        train_std_forces = torch.from_numpy(data_stats["std_forces"]).float().to(device)
        print(f"Loaded dataset mean and std from {data_stats_load_path}")

    if constant_force_norm is not None:
        train_std_forces[:] = constant_force_norm

    return train_mean_locations, train_std_locations, train_mean_forces, train_std_forces


def get_image_xy_locations(dataset: DatasetConfig) -> torch.Tensor:
    if dataset.real_data:
        return RealDataset.get_image_xy_locations()
    else:
        return SimulationDataset.get_image_xy_locations()

def get_train_and_test_loader(dataset: DatasetConfig, data_loader: DataLoaderConfig) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training and testing datasets.

    This function builds datasets using `create_train_test_dataset` and returns
    corresponding DataLoaders, respecting batch size and worker settings from
    `args`.

    Parameters
    ----------
    dataset : DatasetConfig,
        DatasetConfig object containing the training and testing datasets settings.
    data_loader : DataLoaderConfig,
        DataLoaderConfig object containing dataset and dataloader settings.

    Returns
    -------
    tuple of DataLoader
        (train_loader, test_loader)
    """
    train_dataset, test_dataset = create_train_test_dataset(dataset)
    print(f"Training on {len(train_dataset)} samples")
    print(f"Testing on {len(test_dataset)} samples")

    train_loader = DataLoader(train_dataset, batch_size=data_loader.batch_size, shuffle=True,
                              num_workers=data_loader.num_workers,
                              pin_memory=True, drop_last=len(train_dataset) % data_loader.batch_size == 1,
                              collate_fn=pad_collate)

    test_loader = DataLoader(test_dataset, batch_size=data_loader.batch_size, shuffle=True,
                             num_workers=data_loader.num_workers,
                             pin_memory=True, drop_last=len(test_dataset) % data_loader.batch_size == 1,
                             collate_fn=pad_collate)
    return train_loader, test_loader


def log_conf_matrix(confusion_matrix, epoch):
    """
    Log confusion matrix as an image to WandB.
    """
    num_classes = confusion_matrix.shape[0]
    class_names = [f"class_{i}" for i in range(num_classes)]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(confusion_matrix, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, cbar=False, ax=ax)
    ax.set_ylabel("Predicted")
    ax.set_xlabel("True")
    ax.set_title("Confusion Matrix")
    wandb_log("confusion_matrix_image", wandb.Image(fig), epoch)
    plt.close(fig)


def log_force_prediction(real_data: bool, predicted_forces: torch.Tensor, forces: torch.Tensor,
                         reconstruction_steps, epoch,
                         min_value=-10, max_value=10):
    """
    Log force predictions as images to WandB.
    """
    # plot forces
    if not real_data:
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    else:
        fig, axes = plt.subplots(6, 5, figsize=(18, 15))

    predicted_forces_last = predicted_forces[0, -1].reshape(predicted_forces.shape[2], -1, forces.shape[
        -1])  # first batch, last time step embeddings
    for i, ax in enumerate(axes.flatten()):
        sorted_recon_steps, sorted_indices = torch.sort(reconstruction_steps[0, -1])
        ax.plot(predicted_forces_last[sorted_indices, i, -1].cpu().detach().numpy(), color='red',
                label='predicted_vectors')
        ax.plot(forces[0, sorted_recon_steps][:, i, -1].cpu().detach().numpy(),
                color='blue',
                label='vectors_target')
        ax.autoscale(enable=True, axis='both', tight=True)
        ax.set_ylim(min_value, max_value)
    plt.suptitle('Red: Predicted, Blue: Target')
    plt.tight_layout()
    wandb_log("Forces_test", wandb.Image(fig), epoch)
    plt.close(fig)


def log_forces_error(predicted_forces, forces, reconstruction_steps, input_steps, epoch):
    batch_indices = torch.arange(predicted_forces.shape[0], device=forces.device).unsqueeze(-1).unsqueeze(-1)
    forces_target = forces.view(forces.size(0), forces.size(1), -1)
    forces_target = forces_target.unsqueeze(1).expand(-1, forces_target.size(1), -1, -1)
    forces_target = forces_target[
        batch_indices, input_steps.unsqueeze(-1), reconstruction_steps]
    all_recon_loss = torch.mean((predicted_forces - forces_target) ** 2, dim=-1).mean(dim=-1)
    num_plots_to_plot = min(8, len(all_recon_loss))
    fig, axes = plt.subplots(num_plots_to_plot, 1, figsize=(12, 4 * num_plots_to_plot))
    sorted_input_steps, sorted_indices = torch.sort(input_steps, dim=1)
    sorted_input_steps = sorted_input_steps.cpu().detach().numpy()
    sorted_indices = sorted_indices.cpu().detach().numpy()
    all_recon_loss = all_recon_loss.cpu().detach().numpy()
    for i in range(num_plots_to_plot):
        axes[i].plot(sorted_input_steps[i], all_recon_loss[i, sorted_indices[i]])
    plt.tight_layout()
    wandb_log("Forces Reconstruction Loss", wandb.Image(fig), epoch)
    plt.close(fig)


def log_downstream_models(downstream_model: DownstreamModel, epoch: int) -> None:
    """
    Log downstream models statistics.
    Parameters
    ----------
    downstream_model : DownstreamModel,
    DownstreamModel object.
    epoch: the epoch number
    """
    downstream_average_metrics = downstream_model.get_avg_metrics()
    for key, value in downstream_average_metrics.items():
        if key != 'confusion_matrix':
            wandb_log(key, value, epoch)


def reconstruct_tensor(stack: torch.Tensor) -> np.ndarray:
    """

    Parameters
    ----------
    stack (torch.Tensor): a tensor of shape (K, H, W)

    Returns
    -------
    A reconstructed np array of shape (K, H, W) with 0 as background, 127 insert, 200 pillar and 255 lump

    """
    K, H, W = stack.shape

    reconstructed = np.zeros((K, H, W), dtype=np.uint8)  # background = 0
    stack_cpu = stack.cpu().numpy()
    reconstructed[stack_cpu == 1] = 127  # insert
    reconstructed[stack_cpu == 2] = 200  # pillar
    reconstructed[stack_cpu == 3] = 255  # lump

    return reconstructed

def wandb_log(name: str, data, step):
    if step is None:
        wandb.log({name: data})
    else:
        wandb.log({name: data}, step=step)
