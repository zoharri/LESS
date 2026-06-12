import os

import pyrallis
import pickle

from breast_palpation import *
from configs import PalpConfig
from soft_object_sim import SoftObjectSimulation, SoftShapeFiniteElement, create_circular_probe
from trajectory import *


def main():
    config = pyrallis.parse(config_class=PalpConfig)

    # Instantiate a breast model
    perimeter_vertices, internal_vertices = breast_model(config.breast_model)

    vertices = np.array(perimeter_vertices + internal_vertices)
    boundary_vertices = [i for i, vertex in enumerate(vertices) if tuple(vertex) in perimeter_vertices or (
            tuple(vertex) in internal_vertices and vertex[1] < 0.001)]

    fixed_indices = [i for i, vertex in enumerate(vertices) if vertex[1] < 0.001]
    static_trajectory = StaticTrajectory(1)

    breast_shape = SoftShapeFiniteElement(vertices, fixed_indices, static_trajectory, boundary_vertices,
                                          default_young_modulus=config.breast_model.young_modulus,
                                          default_poisson_ratio=config.breast_model.poisson_ratio,
                                          young_modulus_var=config.breast_model.young_modulus_var,
                                          poisson_ratio_var=config.breast_model.poisson_ratio_var)

    if config.breast_model.lump.include_lump is True:
        lump_center = np.array(config.breast_model.lump.center)
        lump_radius = config.breast_model.lump.radius
        lump_young_modulus = config.breast_model.lump.young_modulus
        lump_poisson_ratio = config.breast_model.lump.poisson_ratio
        breast_shape.add_lump(lump_center, lump_radius, lump_young_modulus, lump_poisson_ratio)

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


if __name__ == '__main__':
    main()
