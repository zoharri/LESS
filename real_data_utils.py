import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict

import cv2
import h5py
import numpy as np
import torch
import torchvision.transforms.functional as TF
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.transform import Rotation as Rot, Slerp


@dataclass
class TrialData:
    traj_press_forces: np.ndarray
    traj_press_angles: np.ndarray
    traj_press_dx: np.ndarray
    traj_press_dy: np.ndarray
    traj_primitives: np.ndarray
    traj_names: List
    xela: np.ndarray
    xela_z: np.ndarray
    xela_norm: np.ndarray
    ee_positions: np.ndarray
    ee_rotations: np.ndarray
    traj_name_indices: dict
    num_sensors: int
    num_trajectories: int
    trajectory_length: int
    lump_center: np.ndarray
    traj_lengths: np.ndarray = None  # actual xela step counts before padding, shape (T,)


@dataclass
class DataFilter:
    press_force: Optional[List[float]] = None
    press_angle: Optional[List[float]] = None
    press_dx: Optional[List[float]] = None
    press_dy: Optional[List[float]] = None


def name_to_lump_center(name: str):
    return np.array([0, 0, 0])


def filter_data(trial_data: TrialData, data_filter: DataFilter) -> TrialData:
    """
    Filter the data based on the data_filter.
    :param trial_data: trial data
    :param data_filter: data filter
    :return: filtered trial data
    """
    traj_indices = np.arange(trial_data.num_trajectories)
    if data_filter.press_force is not None:
        traj_indices = np.where(~np.isin(trial_data.traj_press_forces, data_filter.press_force))[0]
    if data_filter.press_angle is not None:
        traj_indices = np.where(~np.isin(trial_data.traj_press_angles, data_filter.press_angle))[0]
    if data_filter.press_dx is not None:
        traj_indices = np.where(~np.isin(trial_data.traj_press_dx, data_filter.press_dx))[0]
    if data_filter.press_dy is not None:
        traj_indices = np.where(~np.isin(trial_data.traj_press_dy, data_filter.press_dy))[0]

    filtered_lengths = trial_data.traj_lengths[traj_indices] if trial_data.traj_lengths is not None else None
    return TrialData(trial_data.traj_press_forces[traj_indices], trial_data.traj_press_angles[traj_indices],
                     trial_data.traj_press_dx[traj_indices], trial_data.traj_press_dy[traj_indices],
                     trial_data.traj_primitives[traj_indices],
                     [trial_data.traj_names[i] for i in traj_indices], trial_data.xela[traj_indices],
                     trial_data.xela_z[traj_indices], trial_data.xela_norm[traj_indices],
                     trial_data.ee_positions[traj_indices], trial_data.ee_rotations[traj_indices],
                     {name: i for i, name in enumerate(trial_data.traj_names)}, trial_data.num_sensors,
                     len(traj_indices), trial_data.trajectory_length, trial_data.lump_center,
                     filtered_lengths)


def write_traj_video_from_h5file(h5_file_path: Path, traj_name: str, out_path: Path):
    """
    Extracts camera frames of a specific trajectory from an HDF5 file and writes them to a video file.

    Parameters
    ----------
    h5_file_path : Path
        Path to the input HDF5 file containing trajectory data.
    traj_name : str
        Name of the trajectory group within the HDF5 file, which contains the camera frames and timestamps.
    out_path : Path
        Path to the output video file to be written (should have `.mp4` extension).
    """
    with h5py.File(str(h5_file_path), "r") as f:
        camera_frames = f[traj_name]["camera"]["frames"][:]
        camera_timestamps = f[traj_name]["camera"]["timestamps"][:]
        # calc fps from timestamps (nano seconds)
        fps = 1 / np.mean(np.diff(camera_timestamps))

    # write video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (camera_frames.shape[2], camera_frames.shape[1]))
    for frame in camera_frames:
        out.write(frame)
    out.release()
    print(f"Video written to {out_path}")


def R_t_align_origins(seq1_pos, seq2_pos):
    """
    Returns R,t that aligns the origins of the sequences via
    X_1 = X_2@R + t
    """
    RANSAC_ITERATIONS = 1000
    RANSAC_THRESHOLD = 0.02  # 2cm threshold for inliers

    def fit_rigid_transform(A, B):
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        AA = A - centroid_A
        BB = B - centroid_B
        H = np.dot(AA.T, BB)
        U, S, Vt = np.linalg.svd(H)
        R_est = np.dot(Vt.T, U.T)
        if np.linalg.det(R_est) < 0:
            Vt[2, :] *= -1
            R_est = np.dot(Vt.T, U.T)
        t_est = centroid_B - np.dot(R_est, centroid_A)
        return R_est, t_est

    n_points = seq1_pos.shape[0]
    if n_points < 4:
        return None, None, 0

    best_inliers_count = 0
    best_R = np.eye(3)
    best_t = np.zeros(3)
    best_inliers_mask = np.zeros(n_points, dtype=bool)

    for i in range(RANSAC_ITERATIONS):
        indices = np.random.choice(n_points, 3, replace=False)
        src = seq1_pos[indices]
        dst = seq2_pos[indices]

        try:
            R_curr, t_curr = fit_rigid_transform(src, dst)
        except np.linalg.LinAlgError:
            continue

        pts_rob_est = (np.dot(R_curr, seq1_pos.T).T) + t_curr
        errors = np.linalg.norm(seq2_pos - pts_rob_est, axis=1)
        inliers_mask = errors < RANSAC_THRESHOLD
        inliers_count = np.sum(inliers_mask)

        if inliers_count > best_inliers_count:
            best_inliers_count = inliers_count
            best_R = R_curr
            best_t = t_curr
            best_inliers_mask = inliers_mask

    if best_inliers_count > 3:
        final_R, final_t = fit_rigid_transform(seq1_pos[best_inliers_mask], seq2_pos[best_inliers_mask])
        return final_R, final_t, best_inliers_count
    return best_R, best_t, best_inliers_count


def slerp_rotations(t_src, R_src, t_tgt, clip=True):
    """
    t_src: (N,) source times (must be increasing)
    R_src: (N,3,3) source rotation matrices
    t_tgt: (M,) target times
    clip: if True, clamp t_tgt into [t_src[0], t_src[-1]]
    """
    t_src = np.asarray(t_src).astype(float)
    t_tgt = np.asarray(t_tgt).astype(float)
    R_src = np.asarray(R_src).astype(float)

    # Ensure increasing times
    order = np.argsort(t_src)
    t_src = t_src[order]
    R_src = R_src[order]

    t_src, unique_indices = np.unique(t_src, return_index=True)
    R_src = R_src[unique_indices]

    if clip:
        t_tgt = np.clip(t_tgt, t_src[0], t_src[-1])

    r_src = Rot.from_matrix(R_src)
    slerp = Slerp(t_src, r_src)
    r_tgt = slerp(t_tgt)
    return r_tgt.as_matrix()


def _angle_deg_from_rel(R_rel):
    """
    R_rel: (...,3,3)
    returns angle in degrees, shape (...)
    """
    tr = np.trace(R_rel, axis1=-2, axis2=-1)
    # cos(theta) = (trace(R)-1)/2
    c = (tr - 1.0) * 0.5
    c = np.clip(c, -1.0, 1.0)
    ang = np.arccos(c)
    return np.degrees(ang)


def _mean_rotation_SO3(Rs, max_iter=50, tol=1e-12):
    """
    Rs: (K,3,3)
    Returns mean rotation matrix (3,3) using iterative log/exp on SO(3).
    """
    R_mean = Rot.identity()
    Rs_obj = Rot.from_matrix(Rs)
    for _ in range(max_iter):
        # tangent-space deltas: log(R_mean^{-1} * R_i)
        deltas = (R_mean.inv() * Rs_obj).as_rotvec()  # (K,3)
        delta = deltas.mean(axis=0)
        if np.linalg.norm(delta) < tol:
            break
        R_mean = R_mean * Rot.from_rotvec(delta)
    return R_mean.as_matrix()


def estimate_rotation_ransac_fast(
        cam_rots,
        rob_rots,
        iterations=300,
        thresh_deg=5.0,
        min_inliers=50,
        seed=None,
        early_stop_ratio=0.90,
):
    """
    Estimate constant R_cal s.t. rob[i] ~= cam[i] @ R_cal

    cam_rots, rob_rots: (N,3,3)
    Returns (R_cal(3,3), inliers(N,), info)
    """
    cam_rots = np.asarray(cam_rots, dtype=float)
    rob_rots = np.asarray(rob_rots, dtype=float)
    if cam_rots.shape != rob_rots.shape or cam_rots.ndim != 3 or cam_rots.shape[1:] != (3, 3):
        raise ValueError("cam_rots and rob_rots must both be (N,3,3) and same shape.")

    N = cam_rots.shape[0]
    if N < 1:
        raise ValueError("Need at least 1 rotation pair.")

    rng = np.random.default_rng(seed)

    # D_i = cam_i^{-1} * rob_i = cam_i^T * rob_i
    D = np.einsum("nij,njk->nik", np.transpose(cam_rots, (0, 2, 1)), rob_rots)  # (N,3,3)

    best_count = 0
    best_inliers = np.zeros(N, dtype=bool)
    best_hyp = np.eye(3)
    thresh = float(thresh_deg)

    # RANSAC: sample one D_j as hypothesis
    for it in range(int(iterations)):
        j = rng.integers(0, N)
        R_hyp = D[j]

        # relative rotations: R_hyp^{-1} * D_i = R_hyp^T * D_i
        R_rel = np.einsum("ij,njk->nik", R_hyp.T, D)  # (N,3,3)
        errs = _angle_deg_from_rel(R_rel)  # (N,)

        inliers = errs < thresh
        count = int(inliers.sum())

        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_hyp = R_hyp

            # optional early stop if it's already very good
            if best_count >= int(early_stop_ratio * N):
                break

    # refinement
    if best_count >= min_inliers:
        R_cal = _mean_rotation_SO3(D[best_inliers])
        # recompute inliers with refined R_cal
        R_rel = np.einsum("ij,njk->nik", R_cal.T, D)
        errs = _angle_deg_from_rel(R_rel)
        final_inliers = errs < thresh

        # one more refine on final inliers if enough
        if int(final_inliers.sum()) >= min_inliers:
            R_cal = _mean_rotation_SO3(D[final_inliers])
    else:
        # fallback: mean of all
        R_cal = _mean_rotation_SO3(D)
        final_inliers = np.ones(N, dtype=bool)

    info = {
        "num_pairs": N,
        "best_ransac_inliers": best_count,
        "final_inliers": int(final_inliers.sum()),
        "thresh_deg": thresh_deg,
        "iterations": int(iterations),
    }
    return R_cal, final_inliers, info


def calc_calibration(seq1_pos, seq1_rot, seq2_pos, seq2_rot):
    """

    Calculate the Rotation + translation that would fit the world origins (r_origins, t_origins) and the object orientation (R_obj)
    Seq 1 is the camera and 2 is the robot. It fits 1 into 2.
    """
    r_origins, t_origins, inliners = R_t_align_origins(seq1_pos, seq2_pos)
    R_obj, inliers, info = estimate_rotation_ransac_fast(seq1_rot, seq2_rot)

    return r_origins, t_origins, R_obj


def calibrate_sequence(seq_pos, seq_rot, r_origins, t_origins, R_obj):
    """
    Apply the transformations on the sequence
    """
    seq_pos = seq_pos @ r_origins.T + t_origins
    seq_rot = seq_rot @ R_obj
    return seq_pos, seq_rot


def print_warning(message: str, raise_warnings: bool):
    if raise_warnings:
        print(f"Warning: {message}")


def load_h5_data(h5_file_path: Path, max_traj_length: int = 500, shift_origin: bool = True,
                 rel_forces: bool = True,
                 shift_orientation: bool = False,
                 data_filter: Optional[DataFilter] = None, num_trajectories_to_keep: int = -1,
                 close_to_lump_trajs: bool = False, use_est_loc: bool = False, shift_timestamps: bool = False,
                 raise_warnings=True, dont_permute_trajs=False,
                 primitive_names: Optional[Dict[str, int]] = None,
                 use_header_shift: bool = False,
                 z_velocity_threshold: float = 2.0,
                 z_velocity_window: float = 0.2,
                 pose_calibration_path: Optional[Path] = None) -> TrialData:
    """
    Load data from an HDF5 file containing multiple trajectories.
    h5_file_path : Path
        Path to the input HDF5 file containing trajectory data.
    max_traj_length : int
        Maximum length of the trajectories to pad to.
    shift_origin : bool
        Whether to shift the origin of the end effector positions
    rel_forces : bool
        Whether to use relative forces (forces relative to the first position in each trajectory)
    shift_orientation : bool
        Whether to shift the orientation of the end effector positions
    data_filter : DataFilter
        Data filter to filter the data
    num_trajectories_to_keep : int
        Number of trajectories to subsample
    close_to_lump_trajs : bool
        Whether to select the closest to the lump center
    use_est_loc : bool
        Whether to use the estimated location of the end effector
    shift_timestamps : bool
        Whether to shift the timestamps to start from zero
    raise_warnings: bool
        Whether to raise warnings for inconsistent data
    dont_permute_trajs: bool
        Whether to avoid permuting the trajectories when subsampling
    primitive_names: Optional[Dict[str, int]]
    use_header_shift: bool
        When True, use the file-level 'position' and 'orientation' attrs as the shift reference
        instead of computing the mean from the data. Raises ValueError if the required attr is missing.
        Mapping from primitive name strings to integer labels
    pose_calibration_path: Optional[Path]
        Path to a .npz file containing a rigid pose calibration.

    Returns
    -------
    TrialData
        A dataclass containing the loaded data.
    """

    traj_press_forces = []
    traj_press_angles = []
    traj_press_dx = []
    traj_press_dy = []
    traj_primitives = []
    traj_names = []
    all_xela_data = []
    all_ee_positions = []
    all_ee_rotations = []
    all_est_ee_positions = []
    all_est_ee_rotations = []
    traj_actual_lengths = []  # xela step count before padding
    file_name_indices = {}
    header_position = None
    header_R = None

    with h5py.File(str(h5_file_path), "r") as f:
        if use_header_shift:
            if shift_origin:
                if "position" not in f.attrs:
                    raise ValueError(
                        f"use_header_shift=True but file {h5_file_path} has no 'position' attribute")
                header_position = np.array(f.attrs["position"])
            if shift_orientation:
                if "orientation" not in f.attrs:
                    raise ValueError(
                        f"use_header_shift=True but file {h5_file_path} has no 'orientation' attribute")
                header_R = np.array(f.attrs["orientation"]).reshape(3, 3)

        all_keys = list(f.keys())
        all_keys = [int(x.split("_")[1]) for x in all_keys]
        all_keys.sort()
        all_keys = ["traj_" + str(x) for x in all_keys]
        if not dont_permute_trajs:
            np.random.shuffle(all_keys)
        for i, traj_name in enumerate(all_keys):
            xela_data = f[traj_name]["xela"]["data"][:]
            xela_timestamps = f[traj_name]["xela"]["timestamps"][:]

            if xela_data.shape[0] < 10:
                print_warning(f"Trajectory {traj_name} in {h5_file_path} has no data, skipping", raise_warnings)
                continue

            if shift_timestamps:
                xela_timestamps = xela_timestamps - xela_timestamps[0]

            if not use_est_loc:
                ee_position = f[traj_name]["robot"]["ee_position"][:]
                ee_rotation = f[traj_name]["robot"]["ee_rotation"][:]  # rotation is a 3x3 matrix
                robot_timestamps = f[traj_name]["robot"]["timestamps"][:]
                if shift_timestamps:
                    robot_timestamps = robot_timestamps - robot_timestamps[0]
                if len(robot_timestamps) != ee_position.shape[0] or len(robot_timestamps) != ee_rotation.shape[0]:
                    print_warning(
                        f"Trajectory {traj_name} in {h5_file_path} has inconsistent robot timestamps and positions, skipping",
                        raise_warnings)
                    continue
                if len(robot_timestamps) < 10:
                    print_warning(f"Trajectory {traj_name} in {h5_file_path} has no robot data, skipping",
                                  raise_warnings)
                    continue

                if robot_timestamps[-1] < xela_timestamps[0] or robot_timestamps[0] > xela_timestamps[-1]:
                    print_warning(
                        f"Trajectory {traj_name} in {h5_file_path} has no overlapping timestamps between robot and xela, skipping",
                        raise_warnings)
                    continue

                # Take overlapping times to robot timestamps to avoid "early" or "late" data
                valid_indices_xela = \
                    np.where((xela_timestamps >= robot_timestamps[0]) & (xela_timestamps <= robot_timestamps[-1]))[0]
                valid_indices_robot = \
                    np.where((robot_timestamps >= xela_timestamps[0]) & (robot_timestamps <= xela_timestamps[-1]))[0]
                xela_timestamps = xela_timestamps[valid_indices_xela]
                xela_data = xela_data[valid_indices_xela]
                robot_timestamps = robot_timestamps[valid_indices_robot]
                ee_position = ee_position[valid_indices_robot]
                ee_rotation = ee_rotation[valid_indices_robot]

                if z_velocity_threshold > 0.0:
                    z_interp_mm = np.interp(xela_timestamps, robot_timestamps, ee_position[:, 2]) * 1000.0
                    dt_median = float(np.median(np.diff(xela_timestamps))) if len(xela_timestamps) > 1 else 0.01
                    window = max(1, int(round(z_velocity_window / dt_median)))
                    if len(z_interp_mm) > window:
                        dz = z_interp_mm[window:] - z_interp_mm[:-window]
                        dt = xela_timestamps[window:] - xela_timestamps[:-window]
                        dt = np.where(dt > 0, dt, 1e-6)
                        vel = dz / dt
                        hits = np.where(np.abs(vel) > z_velocity_threshold)[0]
                    else:
                        hits = np.array([], dtype=int)
                    if not len(hits):
                        print_warning(
                            f"Trajectory {traj_name} in {h5_file_path} never exceeds z_velocity_threshold "
                            f"{z_velocity_threshold} mm/s, skipping.",
                            raise_warnings)
                        continue
                    start_idx = int(hits[0])
                    xela_data = xela_data[start_idx:]
                    xela_timestamps = xela_timestamps[start_idx:]
                    # Trim robot to cover the new xela range so the length check below still passes
                    valid_robot_after_trim = np.where(robot_timestamps >= xela_timestamps[0])[0]
                    if len(valid_robot_after_trim) < 2:
                        # Need at least 2 robot poses for Slerp interpolation
                        print_warning(
                            f"Trajectory {traj_name} in {h5_file_path} has fewer than 2 robot poses "
                            f"after z-velocity trim, skipping.",
                            raise_warnings)
                        continue
                    robot_timestamps = robot_timestamps[valid_robot_after_trim[0]:]
                    ee_position = ee_position[valid_robot_after_trim[0]:]
                    ee_rotation = ee_rotation[valid_robot_after_trim[0]:]

                if xela_data.shape[0] > max_traj_length:
                    print_warning(
                        f"Trajectory {traj_name} in {h5_file_path} has length greater than max_traj_length, skipping. xela length: {xela_data.shape[0]}",
                        raise_warnings)
                    continue

            if not xela_data.shape[0] <= max_traj_length:
                print_warning(
                    f"Trajectory {traj_name} in {h5_file_path} has length greater than max_traj_length, skipping. Lengths: xela {xela_data.shape[0]}",
                    raise_warnings)
                continue

            # Record actual length before padding
            traj_actual_lengths.append(xela_data.shape[0])

            # Pad xela data to max_traj_length
            xela_timestamps = np.pad(xela_timestamps, (0, max_traj_length - xela_timestamps.shape[0]), mode="edge")
            xela_data = np.pad(xela_data, ((0, max_traj_length - xela_data.shape[0]), (0, 0)), mode="edge")

            if not use_est_loc:
                ee_position_interp = np.zeros((len(xela_timestamps), ee_position.shape[1]))
                for j in range(ee_position.shape[1]):
                    ee_position_interp[:, j] = np.interp(xela_timestamps, robot_timestamps, ee_position[:, j])

                ee_position = ee_position_interp
                ee_rotation = slerp_rotations(
                    t_src=robot_timestamps,
                    R_src=ee_rotation,
                    t_tgt=xela_timestamps
                )

            if use_est_loc:
                if "pos_estimation" not in f[traj_name] or len(f[traj_name]["pos_estimation"]["ee_position"]) == 0:
                    print_warning(f"Trajectory {traj_name} in {h5_file_path} has no estimated position data, skipping",
                                  raise_warnings)
                    continue
                else:
                    est_ee_position = f[traj_name]["pos_estimation"]["ee_position"][:]
                    est_ee_rotation = f[traj_name]["pos_estimation"]["ee_rotation"][:]
                    est_ee_timestamps = f[traj_name]["pos_estimation"]["timestamps"][:]
                    if shift_timestamps:
                        est_ee_timestamps = est_ee_timestamps - est_ee_timestamps[0]
                    est_ee_position_interp = np.zeros((len(xela_timestamps), est_ee_position.shape[1]))
                    for j in range(est_ee_position.shape[1]):
                        est_ee_position_interp[:, j] = np.interp(xela_timestamps, est_ee_timestamps,
                                                                 est_ee_position[:, j])
                    est_ee_position = est_ee_position_interp
                    est_ee_rotation = slerp_rotations(
                        t_src=est_ee_timestamps,
                        R_src=est_ee_rotation,
                        t_tgt=xela_timestamps
                    )
                    all_est_ee_positions.append(est_ee_position)
                    all_est_ee_rotations.append(est_ee_rotation)

            traj_names.append(traj_name)
            file_name_indices[traj_name] = i
            traj_press_forces.append(f[traj_name].attrs["force"])
            traj_press_angles.append(f[traj_name].attrs["angle"])
            traj_press_dx.append(f[traj_name].attrs["dx"])
            traj_press_dy.append(f[traj_name].attrs["dy"])
            if primitive_names is not None and "primitive" in f[traj_name].attrs:
                primitive_name = f[traj_name].attrs["primitive"]
                if primitive_name in primitive_names:
                    traj_primitives.append(primitive_names[primitive_name])
                else:
                    traj_primitives.append(-1)
            else:
                traj_primitives.append(-1)
            all_xela_data.append(xela_data)
            if not use_est_loc:
                all_ee_positions.append(ee_position)
                all_ee_rotations.append(ee_rotation)

    traj_press_forces = np.array(traj_press_forces)
    traj_press_angles = np.array(traj_press_angles)
    traj_press_dx = np.array(traj_press_dx)
    traj_press_dy = np.array(traj_press_dy)
    traj_primitives = np.array(traj_primitives)
    all_xela_data = np.array(all_xela_data)
    all_ee_positions = np.array(all_ee_positions)
    all_ee_rotations = np.array(all_ee_rotations)

    if use_est_loc and len(all_est_ee_positions) > 0:
        all_est_ee_positions = np.array(all_est_ee_positions)
        all_est_ee_rotations = np.array(all_est_ee_rotations)
        all_ee_positions = all_est_ee_positions
        all_ee_rotations = all_est_ee_rotations

    if pose_calibration_path is not None:
        cal = np.load(pose_calibration_path)
        all_ee_positions = all_ee_positions @ cal['r_pos'].T + cal['t_pos']
        all_ee_rotations = all_ee_rotations @ cal['R_ori']

    all_xela_data = all_xela_data.reshape(
        (all_xela_data.shape[0], all_xela_data.shape[1], all_xela_data.shape[2] // 3, 3))
    if rel_forces:
        # make forces relative to the first position in each trajectory
        all_xela_data = all_xela_data - all_xela_data[:, :1, :, :]

    # Determine reference rotation for the rigid body transform (rotation matrices still intact here)
    R_ref = None
    if shift_orientation:
        if use_header_shift:
            R_ref = header_R
        else:
            # Project the mean rotation matrix onto SO(3) via SVD
            mean_R = np.mean(all_ee_rotations.reshape(-1, 3, 3), axis=0)
            U, _, Vt = np.linalg.svd(mean_R)
            R_ref = U @ Vt
            if np.linalg.det(R_ref) < 0:
                U[:, -1] *= -1
                R_ref = U @ Vt

    # Translate
    if shift_origin:
        t_ref = header_position if use_header_shift else np.mean(all_ee_positions, axis=(0, 1), keepdims=True)
        all_ee_positions = all_ee_positions - t_ref

    # Rotate positions and rotation matrices into the reference frame
    if R_ref is not None:
        all_ee_positions = all_ee_positions @ R_ref          # (T, N, 3) @ (3, 3) -> (T, N, 3)
        all_ee_rotations = R_ref.T @ all_ee_rotations

    # Convert from rotation matrix to euler angles
    if len(all_ee_rotations.shape) == 4:
        x_angle = np.arctan2(all_ee_rotations[:, :, 2, 1], all_ee_rotations[:, :, 2, 2])
        y_angle = np.arctan2(-all_ee_rotations[:, :, 2, 0],
                             np.sqrt(all_ee_rotations[:, :, 2, 1] ** 2 + all_ee_rotations[:, :, 2, 2] ** 2))
        z_angle = np.arctan2(all_ee_rotations[:, :, 1, 0], all_ee_rotations[:, :, 0, 0])
        all_ee_rotations = np.stack([x_angle, y_angle, z_angle], axis=-1)
    all_xela_z = all_xela_data[:, :, :, 2]
    all_xela_norm = np.sqrt(np.sum(all_xela_data ** 2, axis=3))
    num_sensors = all_xela_data.shape[2]
    num_trajectories = len(traj_names)
    trajectory_length = all_xela_data.shape[1]

    trial_data = TrialData(traj_press_forces, traj_press_angles, traj_press_dx, traj_press_dy, traj_primitives, traj_names,
                           all_xela_data,
                           all_xela_z, all_xela_norm, all_ee_positions, all_ee_rotations, file_name_indices,
                           num_sensors,
                           num_trajectories,
                           trajectory_length, name_to_lump_center(h5_file_path.stem),
                           np.array(traj_actual_lengths, dtype=np.int64))

    if data_filter is not None:
        trial_data = filter_data(trial_data, data_filter)
    if trial_data.num_trajectories < 105:
        print_warning(f"Number of trajectories in {h5_file_path} is less than 105: {trial_data.num_trajectories}",
                      raise_warnings)

    if num_trajectories_to_keep > 0:
        trial_data = subsample_data(trial_data, num_trajectories_to_keep, h5_file_path, close_to_lump_trajs, dont_permute_trajs, raise_warnings)

    return trial_data


def subsample_data(trial_data: TrialData, num_trajectories: int, h5_file_path: str, close_to_lump_trajs: bool = False,
                   dont_permute_trajs: bool = False, raise_warnings:bool = False):
    """
    Subsample the trial data to num_trajectories, either randomly or by selecting the closest to the lump center.
    :param trial_data: trial data
    :param num_trajectories: number of trajectories to subsample
    :param h5_file_path: path of h5 file, for error messaging
    :param close_to_lump_trajs: whether to select the closest to the lump center
    :param dont_permute_trajs: whether to avoid permuting the trajectories when subsampling
    :return: subsampled trial data
    """
    if num_trajectories > trial_data.num_trajectories:
        print_warning(
            f"num_trajs ({num_trajectories}) is greater than the number of trajectories available "
            f"({trial_data.num_trajectories}) in {h5_file_path}. "
            f"Using all trajectories and randomly repeating to reach {num_trajectories}.",
            raise_warnings=raise_warnings)
        base_indices = list(range(trial_data.num_trajectories))
        extra = np.random.choice(base_indices, size=num_trajectories - trial_data.num_trajectories, replace=True).tolist()
        traj_indices = base_indices + extra
    elif close_to_lump_trajs:
        # get lump center
        lump_center = trial_data.lump_center
        all_locations = trial_data.ee_positions[:, -1, :]
        all_distances = np.linalg.norm(all_locations - np.expand_dims(lump_center, axis=0), axis=-1)
        traj_indices = np.argsort(all_distances)[:num_trajectories]
        # shuffle the indices
        traj_indices = np.random.permutation(traj_indices).tolist()
    else:
        if dont_permute_trajs:
            traj_indices = list(range(0, num_trajectories))
        else:
            traj_indices = np.random.choice(range(0, trial_data.num_trajectories),
                                            size=num_trajectories,
                                            replace=False).tolist()

    subsampled_lengths = trial_data.traj_lengths[traj_indices] if trial_data.traj_lengths is not None else None
    return TrialData(trial_data.traj_press_forces[traj_indices], trial_data.traj_press_angles[traj_indices],
                     trial_data.traj_press_dx[traj_indices], trial_data.traj_press_dy[traj_indices],
                     trial_data.traj_primitives[traj_indices],
                     [trial_data.traj_names[i] for i in traj_indices], trial_data.xela[traj_indices],
                     trial_data.xela_z[traj_indices], trial_data.xela_norm[traj_indices],
                     trial_data.ee_positions[traj_indices], trial_data.ee_rotations[traj_indices],
                     {name: i for i, name in enumerate(trial_data.traj_names)}, trial_data.num_sensors,
                     num_trajectories, trial_data.trajectory_length, trial_data.lump_center,
                     subsampled_lengths)


def get_nn_indices(trial_data_1: TrialData, trial_data_2: TrialData, num_neighbors: int = 1):
    """
    Get the nearest neighbors indices of the end effector positions and orientations of trial_data_2 in trial_data_1.
    :param trial_data_1: trial data 1
    :param trial_data_2: trial data 2
    :param num_neighbors: number of neighbors to find
    :return: indices of the nearest neighbors
    """
    ee_pos_1 = trial_data_1.ee_positions.reshape(-1, 3)
    ee_pos_2 = trial_data_2.ee_positions.reshape(-1, 3)

    # rotation matrices
    ee_ori_1 = trial_data_1.ee_rotations.reshape(-1, 3 * 3)
    ee_ori_2 = trial_data_2.ee_rotations.reshape(-1, 3 * 3)

    ee_vec_1 = np.concatenate([ee_pos_1, ee_ori_1], axis=1)
    ee_vec_2 = np.concatenate([ee_pos_2, ee_ori_2], axis=1)

    nn = KNeighborsClassifier(n_neighbors=num_neighbors, algorithm='kd_tree')
    return nn.fit(ee_vec_1, np.arange(ee_vec_1.shape[0])).kneighbors(ee_vec_2, return_distance=False)


def create_mri_image(model_name: str, synthetic: bool = False, three_dim: bool = False, mri_images_dir: Path = None,
                     randomly_choose_mri: bool = False, merge01_image_class: bool = False, raise_warnings: bool = True):
    """
    Create an MRI image based on the model name and whether it is synthetic or not.
    :param model_name: Model name in the format 'phantom_insert_orientation'
    :param synthetic: Whether to create a synthetic MRI image or load a real one
    :param three_dim: Whether to create a three-dimensional MRI image
    :param mri_images_dir: Directory containing the real MRI images
    :param randomly_choose_mri: Whether to randomly choose an MRI out of two images or choose the first one
    :param Merge01_image_class: merge class 0 1 in the image
    :param raise_warnings: Whether to raise warnings for inconsistent data
    :return: MRI image as a torch tensor
    """
    if len(model_name.split("_")) != 3:
        raise ValueError(f"model_name {model_name} is not in the expected format 'phantom_insert_orientation'")

    phantom_name, insert_name, orientation = model_name.split("_")
    orientation = int(orientation)

    if synthetic and three_dim:
        raise NotImplementedError("synthetic and 3D images are currently not supported.")

    if synthetic:
        mri_scan = create_synthetic_mri_image(insert_name)
    else:
        insert_name_lowercase = insert_name.lower()
        if not randomly_choose_mri:
            mri_number = 1
        else:
            inserts_with_missing_idx = {"st14d15", "st14c13"}
            if insert_name_lowercase in inserts_with_missing_idx:
                mri_number = int(np.random.choice([0, 1]))
            else:
                mri_number = int(np.random.choice([0, 1, 2]))

        mri_file_path = mri_images_dir / f"{insert_name_lowercase}_{mri_number}"
        if three_dim:
            if not mri_file_path.with_suffix('.npy').exists():
                print_warning(f"MRI file {mri_file_path.with_suffix('.npy')} does not exist", raise_warnings)
                mri_image_orig = np.zeros((26, 128, 128))
            else:
                mri_image_orig = np.load(str(mri_file_path.with_suffix('.npy')))
        else:
            if not mri_file_path.with_suffix('.png').exists():
                print_warning(f"MRI file {mri_file_path.with_suffix('.png')} does not exist", raise_warnings)
                mri_image_orig = np.zeros((128, 128))
            else:
                mri_image_orig = cv2.imread(str(mri_file_path.with_suffix('.png')))
                mri_image_orig = cv2.cvtColor(mri_image_orig, cv2.COLOR_BGR2GRAY)

        mri_image_orig = torch.from_numpy(mri_image_orig)
        mri_scan = torch.zeros(*mri_image_orig.shape)

        if merge01_image_class:
            if three_dim:
                mri_scan[(mri_image_orig == 0) | (mri_image_orig == 127)] = 0
                mri_scan[(mri_image_orig == 200)] = 2  # pillar
                mri_scan[(mri_image_orig == 255)] = 3  # lump
            else:
                mri_scan[(mri_image_orig == 255)] = 2
        else:
            if three_dim:
                mri_scan[(mri_image_orig == 0)] = 0
                mri_scan[(mri_image_orig == 127)] = 1
                mri_scan[(mri_image_orig == 200)] = 2  # pillar
                mri_scan[(mri_image_orig == 255)] = 3  # lump
            else:
                # The logic before
                mri_scan[(mri_image_orig == 127) | (mri_image_orig == 255)] = 0
                mri_scan[(mri_image_orig == 127)] = 1
                mri_scan[(mri_image_orig == 255)] = 2

    angle = -orientation * 45
    mri_scan_rotated = rotate_tensor(mri_scan, angle, fill=0, three_dim=three_dim)

    return mri_scan_rotated


def create_synthetic_mri_image(insert_name):
    """
    Create a synthetic MRI image based on the insert name.
    :param insert_name: Insert name in the format 'LumpRadiusAngleDistance'
    :return: Synthetic MRI image as a torch tensor
    """
    match = re.match(r"([a-zA-Z]+)(\d+)([a-zA-Z]+)(\d+)", insert_name)
    if match:
        lump_radius = int(match.group(2)) / 2
        if match.group(3) == "D":
            angle = 22.5
        elif match.group(3) == "C":
            angle = 45
        else:
            raise ValueError("String does not match the expected format")
        dist = int(match.group(4))
    else:
        raise ValueError("String does not match the expected format")
    angle = angle - 45  # the 0 orientation is shifted anti-clockwise
    y = -dist * np.cos(np.deg2rad(angle))
    x = dist * np.sin(np.deg2rad(angle))
    # create a synthetic image with a '1' circle of radius 30 center (0,0), '2' circle of radius lump_radius center (x,y) and the rest '0'
    xs = torch.linspace(-40, 40, 128)
    ys = torch.linspace(-40, 40, 128)
    Y, X = torch.meshgrid(ys, xs, indexing='ij')  # corrected indexing for image-like coordinate system

    image = torch.zeros_like(X)

    # Circle 1: radius 30 at (0, 0) -> value 1
    circle1 = (X ** 2 + Y ** 2) <= 30 ** 2
    image[circle1] = 1

    # Circle 2: radius lump_radius at (x, y) -> value 2
    circle2 = ((X - x) ** 2 + (Y - y) ** 2) <= lump_radius ** 2
    image[circle2] = 2

    # Now turn the image to RGB by setting '2' to green '1' to blue and '0' to white
    image_rgb = torch.ones((4, 128, 128))  # white background
    image_rgb[0][(image == 1) | (image == 2)] = 0
    image_rgb[1][image == 1] = 0
    image_rgb[2][image == 2] = 0
    return image_rgb


def rotate_locations(trial_data: TrialData, angle: float) -> TrialData:
    """
    Rotate the end effector positions and orientations by a given angle around the z-axis at the center-of-mass of the trajectory.
    :param trial_data:
        The trial data to rotate.
    :param angle:
        The angle in degrees to rotate the data.
    :return:
        The rotated trial data.
    """

    # calculate the center of mass of the trajectory based of axis 0 and 1
    # set the com to the mean in x but 34.7% in y
    com_x = np.mean(trial_data.ee_positions, axis=(0, 1))[0]
    max_y = np.max(trial_data.ee_positions, axis=(0, 1))[1]
    min_y = np.min(trial_data.ee_positions, axis=(0, 1))[1]
    com_y = min_y + (max_y - min_y) * (1 - 0.347)
    com = np.array([com_x, com_y, 0])
    # create the rotation matrix
    angle = np.deg2rad(angle)
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                [np.sin(angle), np.cos(angle), 0],
                                [0, 0, 1]])
    # rotate the end effector positions
    rotated_positions = np.dot(trial_data.ee_positions - com, rotation_matrix.T) + com
    # rotate the end effector orientations
    rotated_orientations = np.dot(trial_data.ee_rotations, rotation_matrix.T)
    # create a new trial data object with the rotated data
    rotated_trial_data = TrialData(
        traj_press_forces=trial_data.traj_press_forces,
        traj_press_angles=trial_data.traj_press_angles,
        traj_press_dx=trial_data.traj_press_dx,
        traj_press_dy=trial_data.traj_press_dy,
        traj_primitives=trial_data.traj_primitives,
        traj_names=trial_data.traj_names,
        xela=trial_data.xela,
        xela_z=trial_data.xela_z,
        xela_norm=trial_data.xela_norm,
        ee_positions=rotated_positions,
        ee_rotations=rotated_orientations,
        traj_name_indices=trial_data.traj_name_indices,
        num_sensors=trial_data.num_sensors,
        num_trajectories=trial_data.num_trajectories,
        trajectory_length=trial_data.trajectory_length,
        lump_center=trial_data.lump_center
    )
    return rotated_trial_data


def is_hard_trial(trial_data: TrialData, thresh_num_hard_trajectories: int, thresh_hard_traj_force: int) -> bool:
    forces_diff = np.abs(trial_data.xela_z - trial_data.xela_z[:, :1]).max(axis=(1, 2))
    forces_diff_thresh = forces_diff > thresh_hard_traj_force
    return forces_diff_thresh.sum() > thresh_num_hard_trajectories


def flip_x_locations(trial_data: TrialData) -> TrialData:
    """
    Flip the end effector positions and orientations around the x-axis, including flipping the xela data.
    :param trial_data: The trial data to flip
    :return: The flipped trial data
    """

    # Calc the center of mass and the y axis orientation
    from scipy.spatial.transform import Rotation as R
    positions = trial_data.ee_positions  # shape (N, K, 3)
    orientations = trial_data.ee_rotations  # shape (N, K, 3), Euler angles (XYZ)

    x_axis = np.array([1, 0, 0])  # x-axis in world frame
    y_axis = np.array([0, 1, 0])  # y-axis in world frame
    x_axis = x_axis / np.linalg.norm(x_axis)  # shape (3,)
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)

    R_world_to_local = np.stack([x_axis, y_axis, z_axis], axis=1)  # (3, 3)
    R_local_to_world = R_world_to_local.T  # (3, 3)

    # Flatten to (N*K, 3)
    N, K, _ = positions.shape
    positions_flat = positions.reshape(-1, 3)
    orientations_flat = orientations.reshape(-1, 3)

    # --- Flip positions ---
    positions_local = (R_world_to_local.T @ positions_flat.T).T  # (N*K, 3)
    positions_local[:, 0] *= -1
    flipped_positions_flat = (R_local_to_world @ positions_local.T).T  # (N*K, 3)
    flipped_positions = flipped_positions_flat.reshape(N, K, 3)

    # --- Flip orientations ---
    rot = R.from_euler('xyz', orientations_flat)  # (N*K,)
    rot_matrices = rot.as_matrix()  # (N*K, 3, 3)

    # Transform rotations into local frame
    rot_local_matrices = R_world_to_local.T @ rot_matrices @ R_world_to_local  # (N*K, 3, 3)
    rot_local = R.from_matrix(rot_local_matrices)
    eul_local = rot_local.as_euler('xyz')  # (N*K, 3)
    eul_local[:, 0] *= -1
    eul_local[:, 2] *= -1

    # Transform back to world frame
    rot_flipped_local = R.from_euler('xyz', eul_local)
    rot_flipped_local_matrices = rot_flipped_local.as_matrix()
    rot_flipped_world_matrices = R_local_to_world @ rot_flipped_local_matrices @ R_world_to_local
    flipped_orientations_flat = R.from_matrix(rot_flipped_world_matrices).as_euler('xyz')  # (N*K, 3)
    flipped_orientations = flipped_orientations_flat.reshape(N, K, 3)
    flipped_indices = [25, 26, 27, 28, 29, 21, 22, 23, 24, 15, 16, 17, 18, 19, 20]
    flipped_xela = trial_data.xela.copy()

    # Apply the swaps
    for i, j in enumerate(flipped_indices):
        flipped_xela[:, :, i] = trial_data.xela[:, :, j]
        flipped_xela[:, :, j] = trial_data.xela[:, :, i]
    flipped_xela_z = flipped_xela[:, :, :, 2]
    flipped_xela_norm = np.sqrt(np.sum(flipped_xela ** 2, axis=3))

    return TrialData(
        traj_press_forces=trial_data.traj_press_forces,
        traj_press_angles=trial_data.traj_press_angles,
        traj_press_dx=trial_data.traj_press_dx,
        traj_press_dy=trial_data.traj_press_dy,
        traj_primitives=trial_data.traj_primitives,
        traj_names=trial_data.traj_names,
        xela=flipped_xela,
        xela_z=flipped_xela_z,
        xela_norm=flipped_xela_norm,
        ee_positions=flipped_positions,
        ee_rotations=flipped_orientations,
        traj_name_indices=trial_data.traj_name_indices,
        num_sensors=trial_data.num_sensors,
        num_trajectories=trial_data.num_trajectories,
        trajectory_length=trial_data.trajectory_length,
        lump_center=trial_data.lump_center
    )


def rotate_tensor(tensor: torch.Tensor, angle: float, fill: float, three_dim: bool = False) -> torch.Tensor:
    """
    Rotate a 2D or 3D tensor by the given angle.

    Parameters
    ----------
    tensor : torch.Tensor
        Input image tensor. Shape (C, H, W) for 2D or (C, K, H, W) for 3D.
    angle : float
        Rotation angle in degrees. Positive values mean counter-clockwise.
    fill : float
        Value to fill empty pixels after rotation
    three_dim : bool, optional
        If True, applies rotation slice-by-slice over the depth dimension.

    Returns
    -------
    torch.Tensor
        Rotated tensor (in-place modified if three_dim=True).
    """
    if three_dim:
        for k in range(tensor.shape[0]):  # assume shape (C, D, H, W)
            tensor[k, :, :] = TF.rotate(
                tensor[k, :, :].unsqueeze(0),
                angle,
                fill=fill
            ).squeeze(0)
        return tensor
    else:
        return TF.rotate(tensor.unsqueeze(0), angle, fill=fill).squeeze(0)


def hflip_tensor(tensor: torch.Tensor, three_dim: bool = False) -> torch.Tensor:
    """
    Horizontally flip a 2D or 3D tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        Input image tensor. Shape (C, H, W) for 2D or (C, K, H, W) for 3D.
    three_dim : bool, optional
        If True, applies flip slice-by-slice over the depth dimension.

    Returns
    -------
    torch.Tensor
        Flipped tensor (in-place modified if three_dim=True).
    """
    if three_dim:
        for k in range(tensor.shape[0]):  # (C, D, H, W)
            tensor[k, :, :] = TF.hflip(tensor[k, :, :].unsqueeze(0)).squeeze(0)
        return tensor
    else:
        return TF.hflip(tensor.unsqueeze(0)).squeeze(0)
