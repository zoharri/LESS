from torch.utils.data import Dataset

from train_rep_and_downstream_config import DatasetConfig


class BaseDataset(Dataset):
    """Shared initialisation and __len__ for both real and simulation datasets."""

    def __init__(self, config: DatasetConfig) -> None:
        self.num_trajs = config.num_training_trajs
        self.close_to_lump_trajs = config.close_to_lump_trajs
        self.dont_permute_trajs = config.dont_permute_trajs
        self.shift_trial_origin = config.shift_trial_origin
        self.num_last_steps = config.num_last_steps
        self.keep_keyword = config.keep_keyword
        self.remove_keyword = config.remove_keyword
        self.all_keys: list = []

    def __len__(self) -> int:
        return len(self.all_keys)
