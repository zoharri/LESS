import numpy as np
import torch
from scipy.spatial import Delaunay

from trajectory import *


class SoftShape():
    def __init__(self, vertices, fixed_vertices_idx, trajectory, boundary_idx):
        self.vertices = torch.tensor(vertices, dtype=torch.float32,
                                     requires_grad=True)  # List of indices of vertices of object
        self.initial_vertices = torch.zeros_like(self.vertices, requires_grad=False)
        self.initial_vertices.data = self.vertices.data.clone()
        self.fixed_vertices_idx = fixed_vertices_idx
        self.boundary_idx = boundary_idx
        self.frozen_mask = torch.ones_like(self.vertices, requires_grad=False)
        self.frozen_mask[self.fixed_vertices_idx] = 0  # Set the mask to 1 for non-fixed vertices
        self.trajectory = trajectory

    def reset_positions(self):
        '''Reset the positions of the vertices to their original positions.'''
        with torch.no_grad():
            self.vertices.data = self.initial_vertices.data.clone()


class SoftShapeSpringMass(SoftShape):
    def __init__(self, vertices, fixed_vertices_idx, trajectory, links, link_weights, boundary_idx):
        super().__init__(vertices, fixed_vertices_idx, trajectory, boundary_idx)
        self.links = links
        if link_weights is None:
            self.link_weights = [0.1 for _ in range(len(links))]
        else:
            self.link_weights = link_weights
        self.edges = []
        self.rest_lengths = []
        self.compute_edges()
        self.compute_rest_lengths()

    def compute_edges(self):
        self.edges = [self.boundary_idx[i:i + 2] for i in range(len(self.boundary_idx) - 1)] + [
            [self.boundary_idx[-1], self.boundary_idx[0]]]

    def compute_rest_lengths(self):
        self.rest_lengths = []
        for link in self.links:
            v1, v2 = link
            rest_length = torch.norm(self.vertices[v1] - self.vertices[v2], p=2).detach()
            self.rest_lengths.append(rest_length)

    def update_positions(self, dt):
        '''Update the positions of the fixed vertices based on the their velocities and the time step.'''
        velocity = torch.tensor(self.trajectory.step(dt), dtype=torch.float32, requires_grad=False)
        with torch.no_grad():
            self.vertices[self.fixed_vertices_idx] = self.vertices[self.fixed_vertices_idx] + dt * velocity

    def internal_energy(self):
        '''Compute the spring energy of the object.'''
        energy = 0
        for link, rest_length, weight in zip(self.links, self.rest_lengths, self.link_weights):
            v1, v2 = link
            current_length = torch.norm(self.vertices[v1] - self.vertices[v2], p=2)
            energy += 0.5 * weight * (current_length - rest_length) ** 2
        return energy


class SoftShapeFiniteElement(SoftShape):
    def __init__(self, vertices, fixed_vertices_idx, trajectory, boundary_idx, default_young_modulus=0.01,
                 default_poisson_ratio=0.5,
                 young_modulus_var=0.0, poisson_ratio_var=0.0):
        super().__init__(vertices, fixed_vertices_idx, trajectory, boundary_idx)
        tri = Delaunay(vertices)
        self.triangles = tri.simplices
        self.links = []
        for tri in self.triangles:
            for i in range(3):
                v1 = tri[i]
                v2 = tri[(i + 1) % 3]
                if (v1, v2) not in self.links and (v2, v1) not in self.links:
                    self.links.append((v1, v2))

        self.edges = []
        self.rest_lengths = []
        self.compute_edges()
        self.compute_rest_points()
        self.young_modulus = torch.tensor(np.ones(len(self.triangles)), dtype=torch.float32,
                                          requires_grad=False) * default_young_modulus
        self.young_modulus += torch.rand_like(self.young_modulus) * 2 * young_modulus_var - young_modulus_var
        self.poisson_ratio = torch.tensor(np.ones(len(self.triangles)), dtype=torch.float32,
                                          requires_grad=False) * default_poisson_ratio
        self.poisson_ratio += torch.rand_like(self.poisson_ratio) * 2 * poisson_ratio_var - poisson_ratio_var
        self.lump_triangles = []

    def compute_edges(self):
        self.edges = [self.boundary_idx[i:i + 2] for i in range(len(self.boundary_idx) - 1)] + [
            [self.boundary_idx[-1], self.boundary_idx[0]]]

    def compute_rest_points(self):
        self.original_positions = []
        for v in self.vertices:
            rest_pos = v.clone().detach()
            self.original_positions.append(rest_pos)
        self.original_positions = torch.stack(self.original_positions)

    def add_lump(self, lump_center, lump_radius, young_modulus, poisson_ratio):
        # Find all the triangles that are within the lump
        lump_triangles = []
        for i, tri in enumerate(self.triangles):
            tri_vertices = self.vertices[tri]
            centroid = torch.mean(tri_vertices, dim=0).detach().numpy()
            if np.linalg.norm(centroid - lump_center) <= lump_radius:
                lump_triangles.append(i)
                self.lump_triangles.append(i)

        # Modify the young modulus and poisson ratio of the lump triangles
        self.young_modulus[lump_triangles] = young_modulus
        self.poisson_ratio[lump_triangles] = poisson_ratio

    def update_positions(self, dt):
        '''Update the positions of the fixed vertices based on the their velocities and the time step.'''
        velocity = torch.tensor(self.trajectory.step(dt), dtype=torch.float32, requires_grad=False)
        with torch.no_grad():
            self.vertices[self.fixed_vertices_idx] = self.vertices[self.fixed_vertices_idx] + dt * velocity

    def internal_energy(self):
        '''Compute the strain energy of the object.'''
        return self.compute_strain_energy_pytorch()

    def compute_strain_energy_pytorch(self):
        # Calculate the B and D matrices for each triangle
        energy = 0
        for tri, young_modulus, poisson_ratio in zip(self.triangles, self.young_modulus, self.poisson_ratio):
            # Get the vertices of the current triangle
            tri_vertices = self.vertices[tri]
            orig_tri_vertices = self.original_positions[tri]

            # Compute the area of the triangle using the original positions
            area = 0.5 * torch.abs(torch.linalg.det(torch.tensor([
                [1, orig_tri_vertices[0, 0], orig_tri_vertices[0, 1]],
                [1, orig_tri_vertices[1, 0], orig_tri_vertices[1, 1]],
                [1, orig_tri_vertices[2, 0], orig_tri_vertices[2, 1]]
            ])))

            # Calculate the B matrix
            B = torch.zeros((3, 6))
            for i in range(3):
                j = (i + 1) % 3
                k = (i + 2) % 3
                B[0, 2 * i] = orig_tri_vertices[j, 1] - orig_tri_vertices[k, 1]
                B[1, 2 * i + 1] = orig_tri_vertices[k, 0] - orig_tri_vertices[j, 0]
                B[2, 2 * i] = B[1, 2 * i + 1]
                B[2, 2 * i + 1] = B[0, 2 * i]
            B /= (2 * area)

            # Constitutive matrix D
            D = (young_modulus / (1 - poisson_ratio ** 2)) * torch.tensor([
                [1, poisson_ratio, 0],
                [poisson_ratio, 1, 0],
                [0, 0, (1 - poisson_ratio) / 2]
            ])

            # Compute displacements for the current triangle
            displacements = tri_vertices.flatten() - orig_tri_vertices.flatten()

            # Strain and stress calculations
            strain = B @ displacements
            stress = D @ strain

            # Strain energy calculation
            strain_energy = 0.5 * torch.dot(stress,
                                            strain) * 2 * area  # multiply by 2*area for the integral over the area
            energy += strain_energy

        return energy


def create_tetrahedron():
    vertices = [(0, 0), (1, 0), (0.5, np.sqrt(0.75)), (0.5, 0.5 * np.sqrt(0.75))]  # A simple tetrahedron shape
    velocities = [(0, 0), (0, 0), (0, 0), (0, 0)]  # No velocities
    links = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]  # Define links
    boundary_vertices = [0, 1, 2]  # Define boundary vertices
    fixed_indices = [0, 1]  # Fix the first two vertices
    return SoftShapeSpringMass(vertices, fixed_indices, velocities, links, None, boundary_vertices)


def create_point_probe():
    vertices = [(0.8, 0.85)]
    trajectory = FixedVelocityTrajectory(1, 0.8, 0.85, -0.1, 0)
    links = []  # Define links
    boundary_vertices = [0]  # Define boundary vertices
    fixed_indices = [0]  # Fix the first vertex
    return SoftShapeSpringMass(vertices, fixed_indices, trajectory, links, None, boundary_vertices)


def create_circular_probe(radius, num_points, x_center, y_center):
    vertices = []
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        x = radius * np.cos(angle) + x_center
        y = radius * np.sin(angle) + y_center
        vertices.append((x, y))
    trajectory = FixedVelocityTrajectory(1, x_center, y_center, -0.1, 0)
    links = []  # Define links
    boundary_vertices = [i for i in range(num_points)]  # Define boundary vertices
    fixed_indices = [i for i in range(num_points)]  # Fix all vertices
    return SoftShapeSpringMass(vertices, fixed_indices, trajectory, links, None, boundary_vertices)
