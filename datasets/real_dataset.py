import os
from typing import Dict

from real_data_utils import *

from datasets.base_dataset import BaseDataset
from train_rep_and_downstream_config import DatasetConfig


class RealDataset(BaseDataset):
    def __init__(self, config: DatasetConfig) -> None:
        super().__init__(config)
        self.dir_path = Path(config.data_folder)
        self.filter_data = None
        self.max_traj_length = config.max_traj_length
        self.synthetic_mri_image = config.synthetic_mri_image
        self.sensor_subsample = config.sensor_subsample
        self.subsampled_traj_len = config.max_traj_length // config.sensor_subsample
        self.only_z = config.only_z
        self.rotation_aug = config.use_rotation_aug
        self.flip_aug = config.use_flip_aug
        self.no_angles = config.no_angles
        self.shift_trial_orientation = config.shift_trial_orientation
        self.shuffle_keys = config.shuffle_keys
        self.randomly_choose_mri = config.randomly_choose_mri
        self.merge01_image_class = config.merge01_image_class
        self.filter_hard_trials = config.filter_hard_trials
        self.rel_forces = config.rel_forces
        self.location_noise_std = config.location_noise_std
        self.angle_noise_std = config.angle_noise_std
        self.three_dim = config.three_dim
        self.use_est_loc = config.use_est_loc
        self.shift_timestamps = config.shift_timestamps
        self.raise_warnings = config.raise_warnings
        self.balance_training_keyword = config.balance_training_keyword
        self.use_header_shift = config.use_header_shift
        self.z_velocity_threshold = config.z_velocity_threshold
        self.z_velocity_window = config.z_velocity_window
        self.pose_calibration_path = config.pose_calibration_path or None

        self.PRIMITITIVES_NAMES = {"jab": 0, "left_and_right_2": 1, "back_and_forth": 2}

        mri_dir_name = "mri_images_3D" if self.three_dim else "mri_images"
        self.mri_images_dir = os.path.join(config.data_folder, mri_dir_name)

        # set all keys to the names of the hdf5 files in the directory
        self.all_keys = []
        for d in os.listdir(config.data_folder):
            if os.path.isdir(os.path.join(config.data_folder, d)):
                for f in os.listdir(os.path.join(config.data_folder, d)):
                    if os.path.isfile(os.path.join(config.data_folder, d, f)) and f.endswith('.hdf5'):
                        if self.keep_keyword and self.keep_keyword not in d:
                            continue
                        if self.remove_keyword and self.remove_keyword in d:
                            continue

                        if self.filter_hard_trials is not None:
                            data = load_h5_data(self.dir_path / os.path.join(d, f),
                                                max_traj_length=self.max_traj_length)
                            if is_hard_trial(data, self.filter_hard_trials, 70):
                                continue

                        self.all_keys.append(os.path.join(d, f))
        # sort the keys
        self.all_keys.sort()

        if self.shuffle_keys:
            np.random.shuffle(self.all_keys)
            self.all_keys = list(self.all_keys)

    def split_first_trial(self):
        """
        We sampled many of the phantoms multiple times, train only on the first trial.
        """
        first_try_indices = []
        second_try_indices = []

        for i, key in enumerate(self.all_keys):
            p = Path(key)
            if len(p.parent.name.split("_")) < 3:
                first_try_indices.append(i)
            else:
                second_try_indices.append(i)
        return first_try_indices, second_try_indices

    def get_sequence_length(self):
        if self.num_last_steps == -1:
            return min(self.max_traj_length, self.num_last_steps) * self.num_trajs
        else:
            return self.num_last_steps * self.num_trajs

    def split_full_leaveout(self, trial_list: List[str]):
        """
        Leave entire trials (all rotations) in the test set.
        """
        train_indices = []
        test_indices = []
        for i, key in enumerate(self.all_keys):
            p = Path(key)
            if p.parent.name in trial_list:
                test_indices.append(i)
            else:
                train_indices.append(i)
        return train_indices, test_indices

    def split_full_leaveout_change_human_study(self):
        """
        Split for the human study comparison.
        """
        full_leaveout_trial_list = ["S1_St8C13_250331", "S1_St12C13_250504", "S1_St14C13_250506",
                                    "S1_St8D15", "S1_St12D15", "S1_St14D15"]
        return self.split_full_leaveout(full_leaveout_trial_list)

    def split_full_leaveout_change(self):
        full_leaveout_trial_list = ["S1_St8C13_250331", "S1_St12C13_250504", "S1_St12C13", "S1_St14C13_250506",
                                    "S1_St14C13",
                                    "S1_St12D15", "S1_St14D15", "S1_St14D15_250507", "S1_St12C5_250330",
                                    "S1_St8C5_250327"]
        return self.split_full_leaveout(full_leaveout_trial_list)

    def split_half_leaveout(self, trial_list: List[str], num_orientation: int = 5,
                            fix_angles_dict: Optional[Dict[str, int]] = None):
        """
        A split that can be used for change detection.
        For some lump locations (e.g C5, C13, D15, D10), leave *half* of the rotations in test for *all* sizes.
        If fix angles is True - take constant angles, else take random angles.
        """
        count_leaveout_trial = {trial: 0 for trial in trial_list}
        fix_angles = fix_angles_dict is not None
        train_indices = []
        test_indices = []
        for i, key in enumerate(self.all_keys):
            p = Path(key)
            name = p.parent.name
            model_name = os.path.basename(key).split(".")[0]
            orientation = int(model_name.split("_")[-1])
            correct_orientation = False
            for loc in fix_angles_dict:
                if loc in name:
                    correct_orientation = orientation in fix_angles_dict[loc]
                    break
            correct_orientation = correct_orientation or (
                    not fix_angles and name in count_leaveout_trial.keys() and count_leaveout_trial[
                name] < num_orientation)
            if name in count_leaveout_trial.keys() and correct_orientation:
                test_indices.append(i)
                count_leaveout_trial[p.parent.name] += 1
            else:
                train_indices.append(i)
        return train_indices, test_indices

    def split_half_leaveout_change_human_study(self, fix_angles=True):
        """
        Split for the human study comparison.
        """
        trial_list = ["S1_St8C13_250331", "S1_St12C13_250504", "S1_St12C13", "S1_St14C13_250506",
                      "S1_St14C13", "S1_St8D15",
                      "S1_St12D15", "S1_St14D15", "S1_St14D15_250507",
                      "S1_St12C5_250330", "S1_St8C5_250327", "S1_St12C5_250428",
                      "S1_St14C5_250330", "S1_St14C5_250428", "S1_St14C5_250505"]
        fix_angles_dict = {"C5": [0, 1, 2, 3, 5, 7], "C13": [1, 2, 3, 4, 6, 7], "D15": [0, 1, 3, 4, 5, 6],
                           "D10": [0, 2, 4, 5, 6, 7]}
        fix_angles_dict = fix_angles_dict if fix_angles else None
        return self.split_half_leaveout(trial_list, fix_angles_dict=fix_angles_dict)

    def split_half_leaveout_change(self, fix_angles=False):
        trial_list = ["S1_St8C13_250331", "S1_St12C13_250504", "S1_St12C13", "S1_St14C13_250506",
                      "S1_St14C13",
                      "S1_St12D15", "S1_St14D15", "S1_St14D15_250507",
                      "S1_St12C5_250330", "S1_St8C5_250327", "S1_St12C5_250428",
                      "S1_St14C5_250330", "S1_St14C5_250428", "S1_St14C5_250505"]
        fix_angles_dict = {"C5": [0, 1, 2, 3, 5, 7], "C13": [1, 2, 3, 4, 6, 7], "D15": [0, 1, 3, 4, 5, 6],
                           "D10": [0, 2, 4, 5, 6, 7]}

        fix_angles_dict = fix_angles_dict if fix_angles else None
        return self.split_half_leaveout(trial_list, fix_angles_dict=fix_angles_dict)

    def split_keyword(self, keyword: str):
        """
        Split the dataset based on a keyword in the file paths.
        All files containing the keyword are split between train and test sets (80/20),
        the rest of the dataset goes to train.
        """
        # handling 'keyword_' prefix as per your snippet
        len_keyword = len("keyword_")
        keyword = keyword[len_keyword:]

        # 1. Identify all indices containing the keyword
        matched_indices = [i for i, key in enumerate(self.all_keys) if keyword in key]

        # 2. Determine the split point for the matched items
        train_size_matched = int(0.8 * len(matched_indices))

        # 3. Assign the last 20% of matches to Test
        test_indices = matched_indices[train_size_matched:]

        # 4. Assign everything else to Train
        test_set_lookup = set(test_indices)
        train_indices = [i for i in range(len(self.all_keys)) if i not in test_set_lookup]

        return train_indices, test_indices

    def __len__(self):
        return len(self.all_keys)

    def __getitem__(self, idx):
        if idx >= len(self.all_keys):
            raise IndexError("Index out of range")
        file_path = self.dir_path / self.all_keys[idx]

        model_name = os.path.basename(file_path).split(".")[0]
        model_images = create_mri_image(model_name, self.synthetic_mri_image, self.three_dim, Path(self.mri_images_dir),
                                        randomly_choose_mri=self.randomly_choose_mri,
                                        merge01_image_class=self.merge01_image_class,
                                        raise_warnings=self.raise_warnings)

        # load data from the file
        data = load_h5_data(file_path, max_traj_length=self.max_traj_length, data_filter=self.filter_data,
                            num_trajectories_to_keep=self.num_trajs, close_to_lump_trajs=self.close_to_lump_trajs,
                            shift_origin=self.shift_trial_origin, shift_orientation=self.shift_trial_orientation,
                            rel_forces=self.rel_forces, use_est_loc=self.use_est_loc,
                            shift_timestamps=self.shift_timestamps, raise_warnings=self.raise_warnings,
                            dont_permute_trajs=self.dont_permute_trajs, primitive_names=self.PRIMITITIVES_NAMES, use_header_shift=self.use_header_shift,
                            z_velocity_threshold=self.z_velocity_threshold, z_velocity_window=self.z_velocity_window,
                            pose_calibration_path=self.pose_calibration_path)
        if self.rotation_aug:
            angle = np.random.uniform(0, 360)
            data = rotate_locations(data, angle)

            model_images = rotate_tensor(model_images, angle, fill=0, three_dim=self.three_dim)

        if self.flip_aug:
            flip = np.random.choice([True, False])
            if flip:
                data = flip_x_locations(data)

                model_images = hflip_tensor(model_images, three_dim=self.three_dim)

        # The arrays are big now so we need to use contiguous() on them
        model_images = torch.as_tensor(model_images, dtype=torch.long).contiguous()

        images = torch.zeros(1, 1, 1, 1)  # currently no image support

        # Full padded tensors from load_h5_data
        all_locations = torch.tensor(data.ee_positions, dtype=torch.float32)  # (T, max_len, 3)
        all_locations = all_locations + self.location_noise_std * torch.randn_like(all_locations)
        if not self.no_angles:
            all_angles = torch.tensor(data.ee_rotations, dtype=torch.float32)
            all_angles = all_angles + self.angle_noise_std * torch.randn_like(all_angles)
            all_locations = torch.cat((all_locations, all_angles), dim=2).unsqueeze(2)  # (T, max_len, 1, D)
        else:
            all_locations = all_locations.unsqueeze(2)  # (T, max_len, 1, 3)
        all_forces = torch.tensor(data.xela, dtype=torch.float32)  # (T, max_len, S, 3)

        # Build variable-length sequences by trimming each trajectory to its actual length
        loc_parts = []
        force_parts = []
        traj_lens = []
        traj_lengths_arr = data.traj_lengths if data.traj_lengths is not None else np.full(data.num_trajectories, data.trajectory_length, dtype=np.int64)
        for i in range(data.num_trajectories):
            actual_len = int(traj_lengths_arr[i])
            loc_i = all_locations[i, :actual_len:self.sensor_subsample]  # (L_i, 1, D)
            f_i = all_forces[i, :actual_len:self.sensor_subsample]       # (L_i, S, 3)
            if self.num_last_steps != -1:
                loc_i = loc_i[-self.num_last_steps:]
                f_i = f_i[-self.num_last_steps:]
            loc_parts.append(loc_i)
            force_parts.append(f_i)
            traj_lens.append(loc_i.shape[0])

        locations = torch.cat(loc_parts, dim=0)  # (total_steps, 1, D)
        locations = locations.reshape(locations.shape[0], 1, -1)
        locations[:, :, 0] = -locations[:, :, 0]  # The original x axis is flipped
        forces = torch.cat(force_parts, dim=0)    # (total_steps, S, 3)
        forces = forces.reshape(forces.shape[0], forces.shape[1], -1)
        if self.only_z:
            forces = forces[:, :, 2:3]

        label = torch.tensor([0])
        radius = torch.tensor([0])
        model_images = model_images.long().contiguous()

        # traj_props row 5 = per-trajectory step counts after subsampling/trimming
        traj_props = torch.tensor(np.array([data.traj_press_forces, data.traj_press_angles,
                                            data.traj_press_dx, data.traj_press_dy,
                                            data.traj_primitives,
                                            np.array(traj_lens, dtype=np.float32)]),
                                  dtype=torch.float32)  # (6, T)

        return images, locations, forces, label, model_images, radius, self.all_keys[idx], traj_props

    def get_split_(self, split_type: str):
        if split_type == "split_second_try":
            return self.split_first_trial()
        elif split_type == "split_change_half":
            return self.split_half_leaveout_change(fix_angles=False)
        elif split_type == "split_change_half_fixed":
            return self.split_half_leaveout_change(fix_angles=True)
        elif split_type == "split_change_full":
            return self.split_full_leaveout_change()
        elif split_type == "split_change_half_human_study":
            return self.split_half_leaveout_change_human_study()
        elif split_type == "split_change_full_human_study":
            return self.split_full_leaveout_change_human_study()
        elif split_type.startswith("keyword_"):
            return self.split_keyword(split_type)
        elif split_type == "random":
            train_size = int(0.8 * len(self))
            return list(range(train_size)), list(range(train_size, len(self)))
        else:
            raise ValueError(f"Unknown split type for real dataset: {split_type}")

    def get_split(self, split_type: str):
        train_indices, test_indices = self.get_split_(split_type)
        if self.balance_training_keyword != "":
            # balance the training set based on the keyword
            train_indices_with_keyword = [i for i in train_indices if self.balance_training_keyword in self.all_keys[i]]
            train_indices_without_keyword = [i for i in train_indices if
                                             self.balance_training_keyword not in self.all_keys[i]]
            train_indices_with_keyword_balance = np.random.choice(train_indices_with_keyword,
                                                                  size=len(train_indices_without_keyword),
                                                                  replace=True).tolist()
            train_indices = train_indices_without_keyword + train_indices_with_keyword_balance
            if self.shuffle_keys:
                np.random.shuffle(train_indices)

        return train_indices, test_indices

    @staticmethod
    def get_image_xy_locations():
        return torch.tensor([[-0.03, -0.03], [0.03, 0.03]])

    @staticmethod
    def plot_locations_on_model_image(model_images, locations):
        from matplotlib import pyplot as plt
        plt.scatter(locations.mean(1)[:, 0], locations.mean(1)[:, 1])
        plt.imshow(model_images, extent=(-0.03, 0.03, 0.03, -0.03))
