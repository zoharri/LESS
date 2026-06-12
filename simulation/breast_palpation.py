import numpy as np

from configs import BreastModelConfig


# Simulation of breast palpation

def breast_model(config: BreastModelConfig):
    # Extract parameters from config
    grid_type = config.grid_type
    R = config.radius
    z = config.grid_size
    z_perimeter = config.perimeter_grid_size
    N = config.num_inner_points

    # A semi-circular breast model
    def generate_semi_circle_with_grid(radius, grid_size, perimeter_grid_size):
        # Generate vertices on the semi-circular perimeter
        angles = np.linspace(0, np.pi, int(np.pi * radius / perimeter_grid_size))
        perimeter_vertices = [(radius * np.cos(angle), radius * np.sin(angle)) for angle in angles]

        # Generate vertices inside the semi-circle on a square grid
        internal_vertices = []
        for x in np.arange(-radius, radius + grid_size, grid_size):
            for y in np.arange(0, radius + grid_size, grid_size):
                if x ** 2 + y ** 2 <= radius ** 2:
                    internal_vertices.append((x + np.random.randn() * config.random_vertices_std,
                                              y + np.random.randn() * config.random_vertices_std))

        return perimeter_vertices, internal_vertices

    def hammersley_points(n, base=2):
        def phi(n, base):
            result = 0
            f = 1.0 / base
            while n > 0:
                result += f * (n % base)
                n //= base
                f /= base
            return result

        return [(i / n, phi(i, base)) for i in range(n)]

    def map_to_semicircle(points, radius):
        return [(radius * np.sqrt(p[1]) * np.cos(np.pi * p[0]), radius * np.sqrt(p[1]) * np.sin(np.pi * p[0])) for p in
                points]

    def generate_semi_circle_with_hammersley(radius, num_inner_points, grid_size):
        # Generate Hammersley points in the unit square
        hammersley_points_unit_square = hammersley_points(num_inner_points)

        # Map the points to the semi-circle
        internal_vertices = map_to_semicircle(hammersley_points_unit_square, radius)
        angles = np.linspace(0, np.pi, int(np.pi * radius / grid_size))
        perimeter_vertices = [(radius * np.cos(angle), radius * np.sin(angle)) for angle in angles]

        return perimeter_vertices, internal_vertices

    def generate_semi_circle_with_grid_and_center_points(radius, grid_size):
        # Generate vertices on the semi-circular perimeter
        angles = np.linspace(0, np.pi, int(np.pi * radius / grid_size))
        perimeter_vertices = [(radius * np.cos(angle), radius * np.sin(angle)) for angle in angles]

        # Generate vertices inside the semi-circle on a square grid
        internal_vertices = []
        center_vertices = []
        for x in np.arange(-radius, radius + grid_size, grid_size):
            for y in np.arange(0, radius + grid_size, grid_size):
                if x ** 2 + y ** 2 <= radius ** 2:
                    internal_vertices.append((x, y))
                    # Calculate and add the center point of the current square
                    center_x = x + grid_size / 2
                    center_y = y + grid_size / 2
                    if center_x ** 2 + center_y ** 2 <= radius ** 2:
                        center_vertices.append((center_x, center_y))

        return perimeter_vertices, internal_vertices + center_vertices

    # Generate vertices based on grid type
    if grid_type == 'square':
        perimeter_vertices, internal_vertices = generate_semi_circle_with_grid(R, z, z_perimeter)
    elif grid_type == 'square_with_middle_point':
        perimeter_vertices, internal_vertices = generate_semi_circle_with_grid_and_center_points(R, z)
    elif grid_type == 'hammersley':
        perimeter_vertices, internal_vertices = generate_semi_circle_with_hammersley(R, N, z)
    else:
        raise ValueError("Invalid grid type. Please choose 'square', 'square_with_middle_point', or 'hammersley'.")

    return perimeter_vertices, internal_vertices
