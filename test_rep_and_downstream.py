"""
Evaluation script for trained representation + downstream model checkpoints.

Runs one or more checkpoint seeds against a test split, aggregates metrics
across seeds (mean ± std), and logs everything to WandB.

Usage:
    Fill in the checkpoint IDs in the TESTS registry and update the dataset
    path constants at the top of the file, then:

        python test_rep_and_downstream.py --test <TEST_NAME>

Each entry in TESTS is a TestConfig that specifies:
  - checkpoints: one or more run IDs (folder names under checkpoints/) per seed
  - dataset path overrides relative to the training config
  - inference-time flags (big_phantom, non_background_threshold, etc.)
  - test-data-only overrides (use_est_loc, pose_calibration_path, etc.) applied
    only to the test_data_folder dataset, not the train normalisation loader

For each checkpoint the script:
  1. Builds a pyrallis TrainConfig, restoring model architecture from the
     saved train_config.yaml in the checkpoint directory.
  2. Runs the downstream model over the test set.
  3. Collects per-sample metrics and averages them across seeds.

Final WandB log contains per-seed metrics plus cross-seed AVG and STD.
"""

import argparse
import dataclasses
import os
import sys
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import wandb
import numpy as np
import pyrallis
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

from datasets.real_dataset import RealDataset
from datasets.simulation_dataset import SimulationDataset
from train_rep_and_downstream_config import TrainConfig
from train_utils import (
    create_rep_and_downstream_models,
    data_preprocess, set_random_seed, pop_s_and_next,
    pop_item_with_substring, get_dataset_mean_std, get_image_xy_locations,
    pad_collate, get_train_and_test_loader,
)


# ---------------------------------------------------------------------------
# Dataset paths — replace these with your local paths before running
# ---------------------------------------------------------------------------
DATASET_POKE_PRIMITIVE = "data/data_poke_primitive"
DATASET_BIG_SET = "data/data_poke_big"
DATASET_MULTI_SET = "data/data_poke_multi"
DATASET_HANDHELD = "data/data_handheld"

# ---------------------------------------------------------------------------
# Test registry
# ---------------------------------------------------------------------------

@dataclass
class TestConfig:
    checkpoints: List[str]
    is_3d: bool = True
    is_sim: bool = False
    train_data_folder: Optional[str] = None
    test_data_folder: Optional[str] = None
    norm_file: Optional[str] = None
    sensor_subsample: Optional[int] = None
    num_training_trajs: int = -1
    big_phantom: bool = False
    log_images: bool = False
    split_type: Optional[str] = None
    non_background_threshold: Optional[float] = None
    remove_keyword: Optional[str] = None
    keep_keyword: Optional[str] = None
    balance_training_keyword: Optional[str] = None
    # Test-data-only fields (applied only to the test_data_folder dataset, not the train loader)
    pose_calibration_path: Optional[str] = None
    use_est_loc: Optional[bool] = None
    max_traj_length: Optional[int] = None
    dont_permute_trajs: Optional[bool] = None
    shift_trial_orientation: Optional[bool] = None
    shift_trial_origin: Optional[bool] = None


# Fill in checkpoint run IDs (folder names under checkpoints/) after running the
# corresponding Stage 2 sweep. Each entry lists one ID per seed.
TESTS: Dict[str, TestConfig] = {
    # sweeps/global_baseline/global_TCNN2D_poke_seeds.yaml
    "GLOBAL_2D_POKE": TestConfig(
        checkpoints=["", "", ""],
        is_3d=False,
    ),
    # sweeps/global_baseline/global_TCNN3D_poke_seeds.yaml
    "GLOBAL_3D_POKE": TestConfig(
        checkpoints=["", "", ""],
    ),
    # sweeps/global_baseline/global_TCNN3D_poke_seeds.yaml
    "GLOBAL_3D_BIG_POKE": TestConfig(
        checkpoints=["", "", ""],
        test_data_folder=DATASET_BIG_SET,
        big_phantom=True,
    ),
    # sweeps/global_baseline/global_TCNN3D_poke_seeds.yaml
    "GLOBAL_MULTI_POKE": TestConfig(
        checkpoints=["", "", ""],
        test_data_folder=DATASET_MULTI_SET,
    ),
    # sweeps/less_poke/particle_TCNN_poke_seeds.yaml
    "LESS_POKE": TestConfig(
        checkpoints=["", "", ""],
    ),
    # sweeps/less_poke/particle_TCNN_poke_seeds.yaml
    "LESS_MULTI_POKE": TestConfig(
        checkpoints=["", "", ""],
        test_data_folder=DATASET_MULTI_SET,
    ),
    # sweeps/less_poke/particle_TCNN_poke_seeds.yaml
    "LESS_BIG_POKE": TestConfig(
        checkpoints=["", "", ""],
        test_data_folder=DATASET_BIG_SET,
        big_phantom=True,
    ),
    # sweeps/less_primitive/particle_TCNN_primitive_seeds.yaml
    "LESS_JOINT": TestConfig(
        checkpoints=["", "", ""],
    ),
    # sweeps/less_primitive/particle_TCNN_primitive_seeds.yaml
    "LESS_PRIMITIVE": TestConfig(
        checkpoints=["", "", ""],
    ),
    # sweeps/less_poke/particle_TCNN_poke_seeds.yaml — poke model evaluated on primitive split
    "LESS_TRAIN_POKE_TEST_PRIMITIVE": TestConfig(
        checkpoints=["", "", ""],
        train_data_folder=DATASET_POKE_PRIMITIVE,
        split_type="keyword_primitives",
        remove_keyword="",
    ),
    # sweeps/less_poke/particle_TCNN_poke_seeds.yaml
    "LESS_HANDHELD_POKE": TestConfig(
        checkpoints=["", "", ""],
        test_data_folder=DATASET_HANDHELD,
        use_est_loc=True,
        pose_calibration_path=os.path.join(DATASET_HANDHELD, "est_to_robot_calibration.npz"),
        max_traj_length=300,
        shift_trial_orientation=False,
        shift_trial_origin=False,
        non_background_threshold=0.8,
        remove_keyword="", keep_keyword="", balance_training_keyword="",
    ),
    # sweeps/less_primitive/particle_TCNN_primitive_seeds.yaml
    "LESS_HANDHELD_JOINT": TestConfig(
        checkpoints=["", "", ""],
        test_data_folder=DATASET_HANDHELD,
        use_est_loc=True,
        pose_calibration_path=os.path.join(DATASET_HANDHELD, "est_to_robot_calibration.npz"),
        max_traj_length=300,
        shift_trial_orientation=False,
        shift_trial_origin=False,
        non_background_threshold=0.8,
        remove_keyword="", keep_keyword="", balance_training_keyword="",
    ),
}


def load_test_dataset(file_path, dataset_config):
    cfg = dataclasses.replace(dataset_config, data_folder=file_path)
    return RealDataset(cfg) if cfg.real_data else SimulationDataset(cfg)


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def main(args, test_cfg, big_phantom, log_images=False):
    if args.models.rep_learning_model_checkpoint:
        possible_config = Path(args.models.rep_learning_model_checkpoint).parent / "train_config.yaml"
        if os.path.exists(possible_config):
            print(f"Loading train config from {possible_config}")
            sys.argv = pop_s_and_next(sys.argv, "--config_path")
            sys.argv = pop_item_with_substring(sys.argv, "--config_path")
            args = pyrallis.parse(TrainConfig, config_path=possible_config)

    set_random_seed(args.random_seed)
    rep_model, downstream_model = create_rep_and_downstream_models(
        args, Path(args.models.rep_learning_model_checkpoint).parent)

    # Validate: test-data-only fields should only be set when test_data_folder is also set.
    if test_cfg is not None and args.dataset.test_data_folder == "":
        active_test_only = [f for f in _TEST_DATA_ONLY_FIELDS if getattr(test_cfg, f) is not None]
        if active_test_only:
            print(f"WARNING: test-data-only fields {active_test_only} are set but test_data_folder is empty.")

    train_loader, test_loader = get_train_and_test_loader(args.dataset, args.data_loader)
    if args.dataset.test_data_folder != "":
        # Apply test-data-only overrides exclusively to the test dataset config.
        test_dataset_cfg = args.dataset
        if test_cfg is not None:
            overrides = {f: getattr(test_cfg, f) for f in _TEST_DATA_ONLY_FIELDS
                         if getattr(test_cfg, f) is not None}
            if overrides:
                test_dataset_cfg = dataclasses.replace(args.dataset, **overrides)
        test_dataset = load_test_dataset(args.dataset.test_data_folder, test_dataset_cfg)
        print(f"Testing on {len(test_dataset)} samples (test_data_folder override)")
        test_loader = DataLoader(test_dataset, batch_size=args.data_loader.batch_size, shuffle=False,
                                 num_workers=args.data_loader.num_workers,
                                 pin_memory=True,
                                 drop_last=len(test_dataset) % args.data_loader.batch_size == 1,
                                 collate_fn=pad_collate)

    train_mean_locations, train_std_locations, train_mean_forces, train_std_forces = get_dataset_mean_std(
        args.data_processing, train_loader, rep_model.device, args.constant_force_norm)

    img_coords = get_image_xy_locations(args.dataset)
    img_coords = img_coords.to(rep_model.device)
    img_coords = (img_coords - train_mean_locations.mean(0)[:2]) / (train_std_locations.mean(0)[:2])

    if big_phantom:
        img_coords = img_coords * 2

    downstream_model.eval()
    rep_model.eval()
    print("Starting Prediction...")
    downstream_model.reset_metrics()
    for _, locations, forces, _, model_images, _, exp_name, traj_props, padding_mask in test_loader:
        with torch.no_grad():
            traj_lengths = traj_props[:, 5, :].long() if traj_props.shape[1] > 5 else None
            locations, forces, model_images = data_preprocess(
                locations, forces, model_images, train_mean_locations, train_std_locations,
                train_mean_forces, train_std_forces, args.dont_norm_locations, args.relative_locations,
                args.zero_location, args.zero_forces, args.shuffle_order, args.dataset.num_training_trajs,
                rep_model.device, traj_lengths=traj_lengths)
            padding_mask = padding_mask.to(rep_model.device)

            representation, rep_loss, rep_loss_info, rep_inference_results = rep_model(forces, locations,
                                                                                        padding_mask=padding_mask)

            if args.models.downstream_model.name == "PhantomIndexClassifier":
                downstream_kwargs = {"exp_name": exp_name}
            elif args.models.downstream_model.name in ["TransposedConvParRepPred", "TransposedConvParRepPred3D"]:
                downstream_kwargs = {
                    "particles_locations": rep_inference_results["particles_locations"],
                    "active_particles_mask": rep_inference_results["active_particles_mask"],
                    "img_coords": img_coords,
                }
            else:
                downstream_kwargs = {}

            downstream_prediction, downstream_loss, downstream_loss_info, downstream_results = downstream_model(
                representation, model_images, is_train=False, **downstream_kwargs)

            if log_images:
                vis_downstream_results = {k: v.clone() for k, v in downstream_results.items()}
                vis_model_images = model_images.clone()
                for i in range(vis_model_images.shape[0]):
                    curr_exp_name = exp_name[i:i + 1]
                    curr_vis_model_image = vis_model_images[i:i + 1]
                    curr_downstream_results = {k: v[i:i + 1] for k, v in vis_downstream_results.items()}
                    fig = downstream_model.visualize_predictions(curr_downstream_results, curr_vis_model_image)
                    if fig is not None:
                        wandb.log({f'Predictions/{str(curr_exp_name[0])}': wandb.Image(fig)})

    downstream_average_metrics = downstream_model.get_avg_metrics()
    for key, value in downstream_average_metrics.items():
        wandb.log({key: value})
    return downstream_average_metrics, args


# TestConfig fields forwarded to pyrallis as CLI overrides — applied to both
# the train and test loaders via the full args object.
_FIELD_TO_ARGV: Dict[str, str] = {
    "train_data_folder": "--dataset.data_folder",
    "test_data_folder":  "--dataset.test_data_folder",
    "norm_file":         "--data_processing.data_stats_load_path",
    "sensor_subsample":  "--dataset.sensor_subsample",
    "split_type":        "--dataset.split_type",
    "remove_keyword":    "--dataset.remove_keyword",
}

# TestConfig fields applied only to the test_data_folder dataset via
# dataclasses.replace — not to the train loader used for normalisation statistics.
_TEST_DATA_ONLY_FIELDS = [
    "use_est_loc",
    "pose_calibration_path",
    "max_traj_length",
    "dont_permute_trajs",
    "shift_trial_orientation",
    "shift_trial_origin",
    "remove_keyword",
    "keep_keyword",
    "balance_training_keyword",
]

def run_tests(tests: Dict[str, TestConfig]):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--test", required=True, choices=list(tests.keys()), help="Test to run")
    args_known, remaining = parser.parse_known_args()
    active_test = args_known.test
    sys.argv = [sys.argv[0]] + remaining

    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    test = tests[active_test]

    from models.downstream.configs import DownstreamModelConfig, ImagingDownstreamModelConfig
    imaging_cfg_scopes = [
        f.name for f in dataclasses.fields(DownstreamModelConfig)
        if f.default_factory is not dataclasses.MISSING
        and isinstance(f.default_factory(), ImagingDownstreamModelConfig)
    ]

    if test.is_3d:
        config_path = "configs/GRU_reconstruction_model_real_3d.yaml"
    elif test.is_sim:
        config_path = "configs/GRU_reconstruction_model_sim.yaml"
    else:
        config_path = "configs/GRU_reconstruction_model_real.yaml"

    wandb.init(project="imaging", name=f"RSS_TEST_{active_test}_{current_date}")
    print(f"Running test: {active_test}")

    all_downstream_metrics = []
    for checkpoint in tqdm(test.checkpoints):
        rep_ckpt = f"checkpoints/{checkpoint}/best_rep_model.pt"
        ds_ckpt = f"checkpoints/{checkpoint}/best_downstream_model.pt"

        argv = [
            sys.argv[0],
            "--config_path", config_path,
            "--models.rep_learning_model_checkpoint", rep_ckpt,
            "--models.downstream_model_checkpoint", ds_ckpt,
            "--data_loader.batch_size", "1",
            "--models.rep_learning_model.reconstruction_model.mask_percentage", "0",
            "--models.rep_learning_model.particles_reconstruction_model.mask_percentage", "0",
        ]
        for field, key in _FIELD_TO_ARGV.items():
            val = getattr(test, field)
            if val is not None:
                argv += [key, str(val)]
        if test.num_training_trajs != -1:
            argv += ["--dataset.num_training_trajs", str(test.num_training_trajs)]
        for scope in imaging_cfg_scopes:
            if test.big_phantom:
                argv += [f"--models.downstream_model.{scope}.inference_on_big_phantom", "True"]
            if test.non_background_threshold is not None:
                argv += [f"--models.downstream_model.{scope}.non_background_threshold",
                         str(test.non_background_threshold)]
        sys.argv = argv

        args = pyrallis.parse(TrainConfig)
        downstream_average_metrics, _ = main(args, test, test.big_phantom, log_images=test.log_images)
        all_downstream_metrics.append(downstream_average_metrics)

    avg_metrics = {}
    for key in all_downstream_metrics[0].keys():
        values = [metrics[key] for metrics in all_downstream_metrics]
        avg_metrics[f"{key}_AVG"] = np.mean(values)
        avg_metrics[f"{key}_STD"] = np.std(values, ddof=1)
    wandb.log(avg_metrics)


if __name__ == "__main__":
    run_tests(TESTS)
