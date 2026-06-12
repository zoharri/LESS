import os

import h5py
import numpy as np
import torch

from datasets.base_dataset import BaseDataset
from train_rep_and_downstream_config import DatasetConfig


class SimulationDataset(BaseDataset):
    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        self.file_path = os.path.join(config.data_folder, config.h5_name)
        self.include_radius = False
        self.only_healthy = config.only_healthy
        self.only_sick = False
        self.no_images = config.no_image_input
        self.uniform_dist_traj_subsample = config.uniform_dist_traj_subsample
        self.random_first_traj_index = config.random_first_traj_index
        self.orig_label = False

        with h5py.File(self.file_path, 'r') as file:
            all_num_trajs = [attr for attr in file[list(file.keys())[0]].attrs.keys() if "num_trajs" in attr]
            self.num_trials = len(all_num_trajs)
            self.all_keys = []
            for k, group in file.items():
                if group['label'][()] != 2 and self.only_sick or group['label'][()] != 1 and self.only_healthy:
                    continue
                if self.keep_keyword and self.keep_keyword not in k:
                    continue
                if self.remove_keyword and self.remove_keyword in k:
                    continue
                for i in range(self.num_trials):
                    self.all_keys.append(f"{k}_{i}")

    def get_sequence_length(self):
        idx = 0
        totlen = 0
        with h5py.File(self.file_path, 'r') as file:
            group_key = self.all_keys[idx]
            trial_number = int(group_key.split('_')[-1])
            experiment_key = group_key[:-(len(str(trial_number)) + 1)]
            group = file[experiment_key]
            for i in range(self.num_trajs):
                if self.num_last_steps == -1:
                    totlen += group.attrs[f'num_images_{trial_number}_{i}']
                else:
                    totlen += min(group.attrs[f'num_images_{trial_number}_{i}'], self.num_last_steps)
        return totlen

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as file:
            if idx >= len(self.all_keys):
                raise IndexError("Index out of range")

            group_key = self.all_keys[idx]
            trial_number = int(group_key.split('_')[-1])
            experiment_key = group_key[:-(len(str(trial_number)) + 1)]
            group = file[experiment_key]
            # Load images and vectors; we load the first trajectory and then a random selection of the remaining trajectories
            if self.num_trajs > group.attrs[f'num_trajs_{trial_number}']:
                raise ValueError(
                    f"num_trajs ({self.num_trajs}) is greater than the number of trajectories in the group ({group.attrs['num_trajs']}).")

            if self.close_to_lump_trajs:
                # get lump center
                lump_center = np.array(group[f'lump_center_{trial_number}'][:])
                # get last location of each trajectory
                all_locations = np.mean(np.array([group[f'vectors_{trial_number}_{i}'][:][-1, :16] for i in
                                                  range(group.attrs[f'num_trajs_{trial_number}'])]), axis=1)
                all_distances = np.linalg.norm(all_locations - lump_center, axis=1)
                traj_indices = np.argsort(all_distances)[:self.num_trajs]
                # shuffle the indices
                if not self.dont_permute_trajs:
                    traj_indices = np.random.permutation(traj_indices).tolist()
            else:
                if self.uniform_dist_traj_subsample:
                    # determenistic sampling (uniform jumps)
                    start_index = np.random.randint(0, group.attrs[
                        f'num_trajs_{trial_number}'] - self.num_trajs) if self.random_first_traj_index else 0
                    jumps = group.attrs[f'num_trajs_{trial_number}'] / self.num_trajs
                    traj_indices = [start_index + int(i * jumps) for i in range(self.num_trajs)]
                else:
                    start_index = np.random.randint(0, group.attrs[
                        f'num_trajs_{trial_number}'] - self.num_trajs) if self.random_first_traj_index else 0
                    traj_indices = list(range(start_index, self.num_trajs))
                if not self.dont_permute_trajs:
                    traj_indices = np.random.permutation(traj_indices).tolist()

            vectors_per_traj = [torch.tensor(group[f'vectors_{trial_number}_{traj_indices[i]}'][:]) for i in
                                range(self.num_trajs)]
            for i in range(len(vectors_per_traj)):
                if self.num_last_steps != -1:
                    curr_num_vectors = min(vectors_per_traj[i].shape[0], self.num_last_steps)
                    vectors_per_traj[i] = vectors_per_traj[i][-curr_num_vectors:]

            traj_starts = []
            traj_ends = []
            for traj_vectors in vectors_per_traj:
                traj_locations = traj_vectors[:, :16].clone()
                traj_starts.append(traj_locations[0, :, :2].mean(dim=0))
                traj_ends.append(traj_locations[-1, :, :2].mean(dim=0))
            traj_starts = torch.stack(traj_starts, dim=0)
            traj_ends = torch.stack(traj_ends, dim=0)

            vectors = torch.cat(vectors_per_traj, dim=0)
            locations = vectors[:, :16]
            locations[:, :, 1] = -locations[:, :, 1]

            if self.shift_trial_origin:
                # mean over dim 0, 1
                mean_location = torch.mean(locations, dim=(0, 1), keepdim=True)
                locations = locations - mean_location
                mean_xy = mean_location.squeeze(0).squeeze(0)[:2]
                traj_starts = traj_starts - mean_xy.unsqueeze(0)
                traj_ends = traj_ends - mean_xy.unsqueeze(0)

            forces = vectors[:, 16:]
            if not self.no_images:
                images = []
                for i in range(self.num_trajs):
                    num_images = group.attrs[f'num_images_{trial_number}_{traj_indices[i]}']
                    traj_images = [torch.tensor(group[f'image_{trial_number}_{traj_indices[i]}_{j}'][:]) for j in
                                   range(num_images)]
                    if self.num_last_steps != -1:
                        curr_num_images = min(num_images, self.num_last_steps)
                        traj_images = traj_images[-curr_num_images:]
                    images.extend(traj_images)
                images = torch.stack(images)
            else:
                images = torch.zeros(1, 1, 1, 1)
            label = torch.tensor(group['label'][:])
            # turn label 2 into 1
            if not self.orig_label:
                label = torch.where(label == 2, torch.tensor(1), label)
            model_images = torch.tensor(group[f'model_imaging_{trial_number}'][:])
            model_images = self.semantic_classification_image(model_images)
            if self.include_radius:
                radius = torch.tensor(group['radius'][()])
            else:
                radius = torch.tensor([0])

            traj_const = torch.full((traj_starts.shape[0],), -1.0, dtype=torch.float32)
            traj_lengths_tensor = torch.tensor(
                [v.shape[0] for v in vectors_per_traj], dtype=torch.float32)
            traj_info = torch.stack([
                traj_starts[:, 0],
                traj_starts[:, 1],
                traj_ends[:, 0],
                traj_ends[:, 1],
                traj_const,
                traj_lengths_tensor,
            ], dim=0)  # (6, T)

        return images, locations, forces, label, model_images, radius, experiment_key, traj_info

    def semantic_classification_image(self, batch_images):
        # Constants for classification
        # Assuming image pixel values are scaled between 0 and 1
        background_threshold = 250 / 256
        object_threshold = 0 / 256
        lump_threshold = 0 / 256

        # Assuming batch_images is in shape [B, C, H, W]
        # Extract R, G, B channels
        if len(batch_images.shape) == 3:
            R = batch_images[0, :, :]
            G = batch_images[1, :, :]
            B = batch_images[2, :, :]
        else:
            R = batch_images[:, 0, :, :]
            G = batch_images[:, 1, :, :]
            B = batch_images[:, 2, :, :]

        # Initialize classification tensor with zeros (background)
        classification = torch.zeros_like(R, dtype=torch.long)

        # Object classification (predominantly blue and above object_threshold)
        is_object = (B >= G) & (B > R) & (B > object_threshold)
        classification[is_object] = 1

        # Lump classification (predominantly green and above lump_threshold)
        is_lump = (G > B) & (G > R) & (G > lump_threshold)
        classification[is_lump] = 2

        # Background classification (high RGB values indicating white or near white)
        is_background = (R > background_threshold) & (G > background_threshold) & (B > background_threshold)
        classification[is_background] = 0

        return classification

    def get_split(self, split_type: str):
        if split_type == "random":
            train_size = int(0.8 * len(self))
            return list(range(train_size)), list(range(train_size, len(self)))
        else:
            raise ValueError(f"Unknown split type for sim dataset: {split_type}")

    @staticmethod
    def get_image_xy_locations():
        return torch.tensor([[-1.25, -0.7], [1.25, 1.1]])

    @staticmethod
    def plot_locations_on_model_image(model_images, locations):
        from matplotlib import pyplot as plt
        plt.scatter(locations.mean(1)[:, 0], locations.mean(1)[:, 1])
        plt.imshow(model_images, extent=(-1.25, 1.25, 1.1, -0.7))
