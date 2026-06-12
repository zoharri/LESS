import argparse
import contextlib
import io
import multiprocessing
import os
import pickle
import random

import numpy as np
import pyrallis
import torch

from breast_palpation import breast_model
from configs import PalpConfig
from shapes import create_circular_probe, SoftShapeFiniteElement
from soft_object_sim import SoftObjectSimulation
from trajectory import *


def random_uniform_ring(center=np.array([0, 0]), R=1, r=0, nsamples=1):
    """
    generate point uniformly distributed in a ring
    """
    nd = len(center)
    x = np.random.normal(size=(nsamples, nd))
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)  # generate on unit sphere
    # using the inverse cdf method
    u = np.random.uniform(size=(nsamples))
    sc = (u * (R ** nd - r ** nd) + r ** nd) ** (
            1 / nd)  # this is inverse the cdf of ring volume as a function of radius
    return x * sc[:, None] + center


def run_single_experiment(config, breast_shape, add_mirror_lump=False):
    # Instantiate a breast model
    if config.breast_model.lump.include_lump is True:
        lump_center = np.array(config.breast_model.lump.center)
        lump_radius = config.breast_model.lump.radius
        lump_young_modulus = config.breast_model.lump.young_modulus
        lump_poisson_ratio = config.breast_model.lump.poisson_ratio
        breast_shape.add_lump(lump_center, lump_radius, lump_young_modulus, lump_poisson_ratio)
        if add_mirror_lump:
            breast_shape.add_lump(np.array([-lump_center[0], lump_center[1]]), lump_radius, lump_young_modulus,
                                  lump_poisson_ratio)

    for i, trajectory in enumerate(
            zip(config.probe.trajectories.type, config.probe.trajectories.frames, config.probe.trajectories.params)):
        trajectory_type, trajectory_frames, trajectory_params = trajectory
        probe_trajectory = Trajectory.from_dict(trajectory_type, trajectory_params)
        initial_position = probe_trajectory.get_initial_position()
        probe = create_circular_probe(radius=config.probe.radius,
                                      num_points=config.probe.num_points,
                                      x_center=initial_position[0], y_center=initial_position[1])
        probe.trajectory = probe_trajectory
        config.simulation.frames = trajectory_frames  # set the number of frames for the simulation

        simulation = SoftObjectSimulation([breast_shape, probe], config.simulation)
        # save the SoftObjectSimulation object to a file
        folder_path = os.path.join(config.simulation.save_folder, f"trajectory_{i}")
        os.makedirs(folder_path, exist_ok=True)
        with open(os.path.join(folder_path, 'simulation.pkl'), 'wb') as f:
            pickle.dump(simulation, f)
        simulation.reset_positions()
        simulation.draw_model(save_folder=folder_path)
        simulation.run_sim(save_folder=folder_path)


# Version information
# Version 1: One feel trajectory and K poke trajectories at random angles
# Version 2: One feel trajectory, K long press trajectories at angles covering pi at 5 increasing penetration values
# Version 3: K long press at random angles and penetration values
def create_experiment(data_folder, version, change_detection_data=False,
                      number_of_trajectories=11, experiment_number=0, config_path="config.yaml", fix_size_bias=True,
                      no_back_noise=False, less_back_noise=False, small_lumps=True):
    if change_detection_data:
        if small_lumps:
            number_of_trials = 2
        else:
            number_of_trials = 5
    else:
        number_of_trials = 1

    print(f"Running experiment {experiment_number}")
    # check if experiment was already run
    if os.path.exists(os.path.join(data_folder, f"experiment_{experiment_number}", f"trial_{number_of_trials - 1}",
                                   f"trajectory_{number_of_trajectories - 1}", "simulation.pkl")):
        print(f"Experiment {experiment_number} already exists.")
        return ""
    # Load the config
    config = pyrallis.parse(config_class=PalpConfig, config_path=config_path, args=[])
    if no_back_noise:
        config.breast_model.poisson_ratio_var = 0
        config.breast_model.young_modulus_var = 0
    if less_back_noise:
        config.breast_model.poisson_ratio_var *= 0.1
        config.breast_model.young_modulus_var *= 0.1
        config.breast_model.young_modulus *= 0.5

    if change_detection_data:
        breast_model_poisson_ratio_0 = 0.1 * random.uniform(0.9, 1.1)
        breast_model_young_modulus_0 = 0.003 * random.uniform(0.9, 1.1)
        breast_model_radius_0 = random.uniform(1, 1.05)

    else:
        breast_model_poisson_ratio_0 = 1 * random.uniform(0.1, 0.1)
        breast_model_young_modulus_0 = 0.2 * random.uniform(0.01, 0.01)
        breast_model_radius_0 = random.uniform(1, 1.05)

    include_lump = random.random() < 1
    changing_lump = random.random() < 0.1

    experiment_folder = os.path.join(data_folder, f"experiment_{experiment_number}")
    os.makedirs(experiment_folder, exist_ok=True)

    # Save changing lump parameters to txt file
    with open(os.path.join(experiment_folder, 'changing_lump.txt'), 'w') as f:
        f.write(str(changing_lump))

    config.simulation.opaque_model = True

    if include_lump:
        config.breast_model.lump.include_lump = True
        config.breast_model.lump.young_modulus = 0.01
        config.breast_model.lump.poisson_ratio = 0.1
        if changing_lump is False and fix_size_bias is True:
            if small_lumps:
                lump_radius_0 = 0.1 * random.uniform(1.1, 2.1)
            else:
                lump_radius_0 = 0.1 * random.uniform(1.1, 4.1)
        else:
            if small_lumps:
                lump_radius_0 = 0.1 * random.uniform(0.7, 1.5)
            else:
                lump_radius_0 = 0.1 * random.uniform(1, 3)
        # lump_center_0 = [0 + random.uniform(-0.2, 0.2), 0.4 + random.uniform(-0.2, 0.2)]
        if small_lumps:
            lump_center_0 = random_uniform_ring(center=np.array([0, 0]), R=0.55, r=0.4, nsamples=1)[0]
        else:
            lump_center_0 = random_uniform_ring(center=np.array([0, 0]), R=0.5, r=0.3, nsamples=1)[0]

        lump_center_0 = [1.3 * float(lump_center_0[0]), 0.2 + float(abs(lump_center_0[1]))]
        config.breast_model.lump.radius = lump_radius_0
    else:
        config.breast_model.lump.include_lump = False

    static_trajectory = StaticTrajectory(1)

    runs_out = ""

    config.breast_model.radius = breast_model_radius_0 + random.gauss(0, 0.01)
    for i in range(number_of_trials):
        # Sample breast model parameters
        config.breast_model.poisson_ratio = breast_model_poisson_ratio_0 + random.gauss(0, 0.01)
        config.breast_model.young_modulus = breast_model_young_modulus_0 + random.gauss(0, 0.0002)

        # Generate breast model
        perimeter_vertices, internal_vertices = breast_model(config.breast_model)
        vertices = np.array(perimeter_vertices + internal_vertices)
        boundary_vertices = [i for i, vertex in enumerate(vertices) if tuple(vertex) in perimeter_vertices or (
                tuple(vertex) in internal_vertices and vertex[1] < 0.001)]

        fixed_indices = [i for i, vertex in enumerate(vertices) if vertex[1] < 0.001]
        breast_shape = SoftShapeFiniteElement(vertices, fixed_indices, static_trajectory, boundary_vertices,
                                              default_young_modulus=config.breast_model.young_modulus,
                                              default_poisson_ratio=config.breast_model.poisson_ratio,
                                              young_modulus_var=config.breast_model.young_modulus_var,
                                              poisson_ratio_var=config.breast_model.poisson_ratio_var)

        # Sample breast model parameters
        if include_lump:
            config.breast_model.lump.center = [lump_center_0[0], lump_center_0[1]]
            if changing_lump is True:
                if small_lumps:
                    config.breast_model.lump.radius = config.breast_model.lump.radius + abs(random.gauss(0.03, 0.01))
                else:
                    config.breast_model.lump.radius = config.breast_model.lump.radius + abs(random.gauss(0.01, 0.01))
            else:
                # config.breast_model.lump.radius = lump_radius_0 + random.gauss(0, 0.005)
                config.breast_model.lump.radius = lump_radius_0

        config.probe.trajectories.type = []
        config.probe.trajectories.frames = []
        config.probe.trajectories.params = []
        trial_folder = os.path.join(experiment_folder,
                                    f"trial_{i}") if change_detection_data is True else experiment_folder
        os.makedirs(trial_folder, exist_ok=True)
        config.simulation.save_folder = trial_folder
        if version == 1:
            # create N trajectories. First one is feel trajectory, the rest are poke trajectories
            for j in range(number_of_trajectories):
                if j == 0:
                    points = feel_trajectory_points(config.breast_model.radius, penetration=0.95, angle=math.pi / 8)
                    trajectory = {
                        'type': 'FourPointTrajectory',
                        'frames': 100,
                        'params': {
                            'T': 100 * config.simulation.dt,
                            'x0': points[0][0],
                            'y0': points[0][1],
                            'x1': points[1][0],
                            'y1': points[1][1],
                            'x2': points[2][0],
                            'y2': points[2][1],
                            'x3': points[3][0],
                            'y3': points[3][1],
                        }
                    }
                else:
                    angle = random.uniform(math.pi / 8, math.pi - math.pi / 8)
                    points = poke_trajectory_points(config.breast_model.radius, penetration=0.99, angle=angle)
                    trajectory = {
                        'type': 'TwoPointTrajectory',
                        'frames': 10,
                        'params': {
                            'T': 10 * config.simulation.dt,
                            'x0': points[0][0],
                            'y0': points[0][1],
                            'x1': points[1][0],
                            'y1': points[1][1],
                        }
                    }
                config.probe.trajectories.type.append(trajectory['type'])
                config.probe.trajectories.frames.append(trajectory['frames'])
                config.probe.trajectories.params.append(trajectory['params'])
        elif version == 2:
            points = feel_trajectory_points(config.breast_model.radius, penetration=0.95, angle=math.pi / 8)
            trajectory = {
                'type': 'FourPointTrajectory',
                'frames': 100,
                'params': {
                    'T': 100 * config.simulation.dt,
                    'x0': float(points[0][0]),
                    'y0': float(points[0][1]),
                    'x1': float(points[1][0]),
                    'y1': float(points[1][1]),
                    'x2': float(points[2][0]),
                    'y2': float(points[2][1]),
                    'x3': float(points[3][0]),
                    'y3': float(points[3][1]),
                }
            }
            config.probe.trajectories.type.append(trajectory['type'])
            config.probe.trajectories.frames.append(trajectory['frames'])
            config.probe.trajectories.params.append(trajectory['params'])

            angles = np.linspace(math.pi / 8, math.pi - math.pi / 8, number_of_trajectories)
            penetration_values = np.linspace(0.9, 0.99, 5)
            for angle in angles:
                for penetration in penetration_values:
                    points = long_press_trajectory_points(config.breast_model.radius, penetration=penetration,
                                                          angle=angle, opening=math.pi / 8)
                    trajectory = {
                        'type': 'FourPointTrajectory',
                        'frames': 30,
                        'params': {
                            'T': 30 * config.simulation.dt,
                            'x0': float(points[0][0]),
                            'y0': float(points[0][1]),
                            'x1': float(points[1][0]),
                            'y1': float(points[1][1]),
                            'x2': float(points[2][0]),
                            'y2': float(points[2][1]),
                            'x3': float(points[3][0]),
                            'y3': float(points[3][1]),
                        }
                    }
                    config.probe.trajectories.type.append(trajectory['type'])
                    config.probe.trajectories.frames.append(trajectory['frames'])
                    config.probe.trajectories.params.append(trajectory['params'])
        elif version == 3:
            # Choose random angles between pi/8 and 7pi/8 and penetration values between 0.9 and 0.99
            angles = np.random.uniform(math.pi / 8, 7 * math.pi / 8, number_of_trajectories)
            penetration_values = np.random.uniform(0.9, 0.99, number_of_trajectories)

            for angle, penetration in zip(angles, penetration_values):
                points = long_press_trajectory_points(config.breast_model.radius, penetration=penetration, angle=angle,
                                                      opening=math.pi / 8)
                trajectory = {
                    'type': 'FourPointTrajectory',
                    'frames': 30,
                    'params': {
                        'T': 30 * config.simulation.dt,
                        'x0': float(points[0][0]),
                        'y0': float(points[0][1]),
                        'x1': float(points[1][0]),
                        'y1': float(points[1][1]),
                        'x2': float(points[2][0]),
                        'y2': float(points[2][1]),
                        'x3': float(points[3][0]),
                        'y3': float(points[3][1]),
                    }
                }
                config.probe.trajectories.type.append(trajectory['type'])
                config.probe.trajectories.frames.append(trajectory['frames'])
                config.probe.trajectories.params.append(trajectory['params'])
        elif version == 4:
            angles = np.linspace(math.pi / 8, math.pi - math.pi / 8, number_of_trajectories)
            angles += np.random.normal(0, math.pi / 8 / 10, number_of_trajectories)
            penetration_values = np.random.uniform(0.9, 0.92, number_of_trajectories)
            for angle, penetration in zip(angles, penetration_values):
                points = long_press_trajectory_points(config.breast_model.radius, penetration=penetration,
                                                      angle=angle, opening=math.pi / 8)
                trajectory = {
                    'type': 'FourPointTrajectory',
                    'frames': 30,
                    'params': {
                        'T': 30 * config.simulation.dt,
                        'x0': float(points[0][0]),
                        'y0': float(points[0][1]),
                        'x1': float(points[1][0]),
                        'y1': float(points[1][1]),
                        'x2': float(points[2][0]),
                        'y2': float(points[2][1]),
                        'x3': float(points[3][0]),
                        'y3': float(points[3][1]),
                    }
                }
                config.probe.trajectories.type.append(trajectory['type'])
                config.probe.trajectories.frames.append(trajectory['frames'])
                config.probe.trajectories.params.append(trajectory['params'])
        elif version == 5:
            angles = np.linspace(math.pi / 8, math.pi - math.pi / 8, number_of_trajectories)
            angles += np.random.normal(0, math.pi / 8 / 10, number_of_trajectories)
            penetration_values = np.random.uniform(0.96, 0.98, number_of_trajectories)
            for angle, penetration in zip(angles, penetration_values):
                points = poke_trajectory_points(config.breast_model.radius, penetration=penetration,
                                                angle=angle)
                trajectory = {
                    'type': 'TwoPointTrajectory',
                    'frames': 10,
                    'params': {
                        'T': 10 * config.simulation.dt,
                        'x0': float(points[0][0]),
                        'y0': float(points[0][1]),
                        'x1': float(points[1][0]),
                        'y1': float(points[1][1]),
                    }
                }
                config.probe.trajectories.type.append(trajectory['type'])
                config.probe.trajectories.frames.append(trajectory['frames'])
                config.probe.trajectories.params.append(trajectory['params'])
        elif version == 6:
            # close to lump trajectories
            # lump angle
            if include_lump:
                lump_angle = math.atan2(lump_center_0[1], lump_center_0[0])
            else:
                lump_angle = np.random.uniform(math.pi / 4, math.pi - math.pi / 4)
            lump_angle = min(lump_angle, math.pi - math.pi / 4)
            lump_angle = max(lump_angle, math.pi / 4)
            angles = np.linspace(lump_angle - math.pi / 8, lump_angle + math.pi / 8, number_of_trajectories)
            angles += np.random.normal(0, math.pi / 8 / 10, number_of_trajectories)
            penetration_values = np.random.uniform(0.96, 0.98, number_of_trajectories)

            for angle, penetration in zip(angles, penetration_values):
                points = poke_trajectory_points(config.breast_model.radius, penetration=penetration,
                                                angle=angle)
                trajectory = {
                    'type': 'TwoPointTrajectory',
                    'frames': 10,
                    'params': {
                        'T': 10 * config.simulation.dt,
                        'x0': float(points[0][0]),
                        'y0': float(points[0][1]),
                        'x1': float(points[1][0]),
                        'y1': float(points[1][1]),
                    }
                }
                config.probe.trajectories.type.append(trajectory['type'])
                config.probe.trajectories.frames.append(trajectory['frames'])
                config.probe.trajectories.params.append(trajectory['params'])
        elif version == 7:
            angles = np.linspace(math.pi / 8, math.pi - math.pi / 8, number_of_trajectories)
            angles += np.random.normal(0, math.pi / 8 / 10, number_of_trajectories)
            penetration_values = np.random.uniform(0.96, 0.98, number_of_trajectories)

            for angle, penetration in zip(angles, penetration_values):
                points = poke_trajectory_points(config.breast_model.radius, penetration=penetration,
                                                angle=angle)
                trajectory = {
                    'type': 'TwoPointTrajectory',
                    'frames': 10,
                    'params': {
                        'T': 10 * config.simulation.dt,
                        'x0': float(points[0][0]),
                        'y0': float(points[0][1]),
                        'x1': float(points[1][0]),
                        'y1': float(points[1][1]),
                    }
                }
                config.probe.trajectories.type.append(trajectory['type'])
                config.probe.trajectories.frames.append(trajectory['frames'])
                config.probe.trajectories.params.append(trajectory['params'])

        # Save the modified config
        with open(os.path.join(trial_folder, 'config.yaml'), 'w') as f:
            pyrallis.dump(config, f)

        stdout_capture = io.StringIO()

        # Redirect stdout
        with contextlib.redirect_stdout(stdout_capture):
            run_single_experiment(config, breast_shape)
        runs_out += stdout_capture.getvalue()
        stdout_capture.close()

    return runs_out


parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", default="/data/dataset_name",
                    help="Folder to store experiment data")
parser.add_argument("--num_experiments", type=int, default=2000, help="Number of experiments to run")
parser.add_argument("--start_index", type=int, default=0, help="Starting index for experiments")
parser.add_argument("--num_processes", type=int, default=30, help="Number of processes")
parser.add_argument("--change_detection_data", type=bool, default=True, help="Gather data for change detection")
parser.add_argument("--number_of_trajectories", type=int, default=100, help="Number of trajectories per trial")
parser.add_argument("--palp_config_path", type=str, default="config.yaml", help="Path to palpation config file")
parser.add_argument("--fix_size_bias", type=bool, default=True, help="Fix size bias in the lump size")
parser.add_argument("--no_back_noise", type=bool, default=False, help="Remove background noise")
parser.add_argument("--less_back_noise", type=bool, default=True, help="Remove background noise")
parser.add_argument("--small_lumps", type=bool, default=True, help="Use small lumps for change detection data")

args = parser.parse_args()

data_folder = args.data_folder
data_folder = os.path.join(data_folder, str(args.start_index))
os.makedirs(data_folder, exist_ok=True)
version = 7
# Add version.txt file with the version number
version_file = os.path.join(data_folder, "version.txt")
with open(version_file, "w") as f:
    f.write(str(version))

N0 = args.start_index
N = args.num_experiments
number_of_processes = args.num_processes
change_detection_data = args.change_detection_data
number_of_trajectories = args.number_of_trajectories
fix_size_bias = args.fix_size_bias
no_back_noise = args.no_back_noise
less_back_noise = args.less_back_noise
small_lumps = args.small_lumps

inputs = list(range(N0, N0 + N))
if number_of_processes == 1:
    outputs = []
    for i in range(N):
        # create_experiment_multiprocess = partial(create_experiment, data_folder=data_folder, version=version, change_detection_data=change_detection_data,
        #  number_of_trajectories=number_of_trajectories)
        def create_experiment_multiprocess(i):
            return create_experiment(data_folder, version, change_detection_data, number_of_trajectories, i,
                                     args.palp_config_path, fix_size_bias, no_back_noise, less_back_noise, small_lumps)


        output = create_experiment_multiprocess(inputs[i])
        outputs.append(output)
else:
    def create_experiment_multiprocess(i):
        random.seed(i)
        np.random.seed(i)
        torch.manual_seed(i)
        try:
            return create_experiment(data_folder, version, change_detection_data, number_of_trajectories, i,
                                     args.palp_config_path, fix_size_bias, no_back_noise, less_back_noise, small_lumps)
        except Exception as e:
            print(f"Experiment {i} failed with error: {e}")
            raise e


    # create_experiment_multiprocess = partial(create_experiment, data_folder=data_folder, version=version, change_detection_data=change_detection_data,
    #  number_of_trajectories=number_of_trajectories)
    pool = multiprocessing.Pool(processes=number_of_processes)

    outputs = pool.map(create_experiment_multiprocess, inputs)

unsuccesful_experiments = []
for i in range(N):
    if "Did not converge." in outputs[i]:
        print(f"Experiment {N0 + i} did not converge.")
        unsuccesful_experiments.append(N0 + i)

# Save the indices of unsuccessful experiments to a text file
with open(os.path.join(data_folder, "unsuccesful_runs.txt"), "w") as f:
    f.write("\n".join(map(str, unsuccesful_experiments)))
