import os
import pickle

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from configs import SimulationConfig
from shapes import *


# Define a class for the soft object simulation
class SoftObjectSimulation:
    def __init__(self, shapes, config: SimulationConfig):
        """
        shapes: List of SoftShape objects
        """
        self.shapes = shapes
        self.dt = config.dt  # Time step for the velocity update
        self.config = config
        self.collision_spring_constant = config.collision_spring_constant
        self.plot_diff = False

    def reset_positions(self):
        for shape in self.shapes:
            shape.reset_positions()

    def point_is_inside_polygon(self, point, polygon):
        """
        Determines if a point is inside a polygon.

        Args:
        - point: A tuple (x, y) representing the point to check.
        - polygon: A list of tuples [(x1, y1), (x2, y2), ..., (xn, yn)] representing the vertices of the polygon.

        Returns:
        - True if the point is inside the polygon, False otherwise.
        """
        x, y = point
        inside = False

        n = len(polygon)
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xints:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def distance_point_to_segment(self, point, segment):
        """Calculate the distance from a point to a line segment.
        
        Args:
        - point: Tuple (px, py), the point from which the distance is measured.
        - segment: Tuple ((x1, y1), (x2, y2)), the endpoints of the line segment.
        
        Returns:
        - The minimum distance from the point to the line segment.
        - The distance vector from the point to the closest point on the segment.
        """
        (x1, y1), (x2, y2) = segment
        px, py = point
        dx, dy = x2 - x1, y2 - y1
        if dx == dy == 0:  # The segment's endpoints are the same
            return torch.hypot(px - x1, py - y1), torch.nn.functional.normalize(torch.tensor([(px - x1), (py - y1)]),
                                                                                dim=0)
        # Calculate the projection of the point onto the line defined by the segment
        t = ((px - x1) * dx + (py - y1) * dy) / (dx ** 2 + dy ** 2)
        t = torch.max(torch.tensor(0), torch.min(torch.tensor(1), t))  # Clamp t to the range [0, 1]
        closest_point = (x1 + t * dx, y1 + t * dy)
        return torch.hypot(px - closest_point[0], py - closest_point[1]), torch.tensor(
            [(closest_point[0] - px), (closest_point[1] - py)])

    def find_closest_edge(self, point, polygon):
        """Find the edge closest to the given point.
        
        Args:
        - point: Tuple (px, py), the point.
        - edges: List of tuples [((x1, y1), (x2, y2)), ...], the edges.
        
        Returns:
        - The edge closest to the point and the distance to this edge and the vector from the point to the closest point on the edge.
        """
        closest_distance = float('inf')
        closest_edge = None
        closest_vector = None
        n = len(polygon)
        for i in range(n):
            segment = (polygon[i], polygon[(i + 1) % n])
            distance, vector = self.distance_point_to_segment(point, segment)
            if distance < closest_distance:
                closest_distance = distance
                closest_vector = vector
                closest_edge = segment
        return closest_edge, closest_distance, closest_vector

    def get_state_observation(self):
        probe_vertices = self.shapes[1].vertices.detach().numpy()
        forces = self.get_force()

        # Check for NaNs in probe_vertices
        if np.isnan(probe_vertices).any():
            print("Warning: NaNs found in probe_vertices. Replacing with zeros.")
            probe_vertices = np.nan_to_num(probe_vertices, nan=0.0)

        # Check for NaNs in forces
        if np.isnan(forces).any():
            print("Warning: NaNs found in forces. Replacing with zeros.")
            forces = np.nan_to_num(forces, nan=0.0)

        return np.concatenate((probe_vertices, forces), axis=0)

    def get_force(self):
        # Calculate the force acting on the probe
        forces = []
        for vertex in self.shapes[1].vertices[self.shapes[1].boundary_idx]:
            collision = self.point_is_inside_polygon(vertex, self.shapes[0].vertices[self.shapes[0].boundary_idx])
            vector = torch.tensor([0.0, 0.0])
            if collision:
                _, _, vector = self.find_closest_edge(vertex, self.shapes[0].vertices[self.shapes[0].boundary_idx])
            forces.append(vector.detach().numpy() + np.random.randn() * self.config.probe_force_noise_std)
        return forces

    def minimize_energy(self, steps=100):
        optimizer = torch.optim.Adam([shape.vertices for shape in self.shapes],
                                     lr=self.config.learning_rate,
                                     betas=(self.config.adam_beta_1, self.config.adam_beta_2))

        def closure():
            optimizer.zero_grad()
            energy = 0
            for shape in self.shapes:
                energy += shape.internal_energy()
            # Check for collisions
            forces = []
            for vertex in self.shapes[1].vertices[self.shapes[1].boundary_idx]:
                collision = self.point_is_inside_polygon(vertex, self.shapes[0].vertices[self.shapes[0].boundary_idx])
                distance = 0
                if collision:
                    _, distance, _ = self.find_closest_edge(vertex,
                                                            self.shapes[0].vertices[self.shapes[0].boundary_idx])
                energy += self.collision_spring_constant * distance * distance  # Add a penalty for collisions (20)
            energy.backward()

            # Apply the mask to freeze certain elements by setting their gradients to 0.
            # Make sure the mask is used to select elements without gradients computation.
            with torch.no_grad():
                for shape in self.shapes:
                    if shape.vertices.grad is not None:
                        shape.vertices.grad[torch.isnan(shape.vertices.grad)] = 0.0
                        shape.vertices.grad *= shape.frozen_mask
            return energy

        # Early stopping parameters
        best_energy = 1e10
        patience = 5
        epochs_no_improve = 0
        improvement_percentage = 0.001  # 0.1% 0.001
        early_stop = False

        if self.config.warmup:
            # Perform 5 warmup steps with a twice larger learning rate
            optimizer.param_groups[0]['lr'] = self.config.learning_rate * 2.0
            for _ in range(5):
                energy = optimizer.step(closure)
            optimizer.param_groups[0]['lr'] = self.config.learning_rate

        for s in range(steps):  # Number of optimization steps
            energy = optimizer.step(closure)

            #  Calculate the improvement threshold as 0.1% of the current best loss
            threshold = best_energy * improvement_percentage

            # Check if the improvement is significant
            improvement = best_energy - energy.item()
            # print(f"Energy: {energy.item()}, Best Energy: {best_energy}, Improvement: {improvement}")
            if improvement > threshold:
                best_energy = energy.item()
                # print(f"Step: {s}/{steps} Energy: {energy.item()}, Best Energy: {best_energy}, Improvement: {improvement}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                # print(f"Stopping early due to plateau in improvement.")
                early_stop = True
                break
        if not early_stop:
            print('Did not converge.')

    def update(self):
        # Update the positions of the vertices
        for shape in self.shapes:
            shape.update_positions(self.dt)
        self.minimize_energy(steps=self.config.steps)
        # Extract current positions of the particles
        object_vertices = self.shapes[0].vertices.detach().numpy()
        probe_vertices = self.shapes[1].vertices.detach().numpy()
        x_object, y_object = object_vertices[:, 0], object_vertices[:, 1]
        x_probe, y_probe = probe_vertices[:, 0], probe_vertices[:, 1]
        return x_object, y_object, x_probe, y_probe

    def run_sim(self, save_folder=None):
        # Set up the plot for animation
        fig, ax = plt.subplots()
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-0.5, 2.5)
        # disable axes
        ax.axis('off')
        xdata, ydata = [], []
        if not self.config.opaque_model:  # For transparent model
            ln, = ax.plot([], [], 'ro', markersize=3, alpha=0.2)  # For drawing points
            ln_probe, = ax.plot([], [], 'go', markersize=3)  # For drawing points
            line_objs = [ax.plot([], [], 'b-', alpha=0.2)[0] for _ in self.shapes[0].edges]  # For drawing lines
            link_objs = [ax.plot([], [], 'm-', alpha=0.2)[0] for _ in self.shapes[0].links]  # For drawing links
            # draw force vectors
            # Add an inset axes at the top right corner
            axins = inset_axes(ax, width="30%", height="30%", loc='upper right')
            # set x and y labels for axins
            axins.set_xlabel('Sensor Index')
            axins.set_ylabel('Force')
            # set y to be only positive
            # Sample x data for the bar plot (categories/indices for the bars)
            indices = list(range(16))  # Assuming there are 16 bars
            # Initial empty bar plot
            bars = axins.bar(indices, [0] * len(indices), color='blue')  # Heights are initially zero
            # plot the force vectors at the probe vertices as arrows
            arrows = ax.quiver([0] * len(indices), [0] * len(indices), [0] * len(indices), [0] * len(indices),
                               scale=0.5, color='blue', width=0.002)
            if self.plot_diff:
                axins.set_ylim(-5e-3, 5e-3)
            else:
                # axins.set_ylim(-2, 2)
                axins.set_ylim(0, 5e-2)
            self.last_heights = [0] * len(indices)
        else:  # For opaque model
            ln, = plt.plot([], [], marker='o', markerfacecolor='none', markeredgecolor='none', linestyle='None')
            ln_probe, = plt.plot([], [], 'ro')  # For drawing points
            line_objs = [plt.plot([], [], 'b-')[0] for _ in self.shapes[0].edges]  # For drawing lines
            link_objs = [plt.plot([], [], 'm-', alpha=0.0)[0] for _ in self.shapes[0].links]  # For drawing links
            bars = []
        # Plotting triangles based on young_modulus
        triangles = self.shapes[0].triangles
        young_modulus = self.shapes[0].young_modulus
        if self.config.opaque_model:
            young_modulus_norm = torch.ones_like(young_modulus)
        else:
            young_modulus_norm = (young_modulus - young_modulus.min()) / (young_modulus.max() - young_modulus.min())
        colors = plt.cm.gray(young_modulus_norm)
        tri_objs = []
        for triangle, color in zip(triangles, colors):
            x = self.shapes[0].vertices[triangle][:, 0].clone().detach().numpy()
            y = self.shapes[0].vertices[triangle][:, 1].clone().detach().numpy()
            tri_obj = ax.fill(x, y, color=color, alpha=0.2)
            tri_objs.append(tri_obj)

        state_observations = []

        def init():
            if self.config.opaque_model:
                return [ln] + [ln_probe] + line_objs + link_objs + [obj for sublist in tri_objs for obj in sublist]
            else:
                return [ln] + [ln_probe] + line_objs + link_objs + [obj for sublist in tri_objs for obj in sublist] + [
                    bars] + [arrows]

        def update_vid(frame):
            print(f"Frame: {frame}/{self.config.frames}")
            x, y, x_probe, y_probe = self.update()
            ln.set_data(x, y)
            ln_probe.set_data(x_probe, y_probe)
            for line_obj, (i, j) in zip(line_objs, self.shapes[0].edges):
                # Update each line to connect the specified pair of particles
                line_obj.set_data([x[i], x[j]], [y[i], y[j]])
            for link_obj, (i, j) in zip(link_objs, self.shapes[0].links):
                # Update each line to connect the specified pair of particles
                link_obj.set_data([x[i], x[j]], [y[i], y[j]])
            for tri_obj, triangle in zip(tri_objs, self.shapes[0].triangles):
                x = self.shapes[0].vertices[triangle][:, 0].clone().detach().numpy()
                y = self.shapes[0].vertices[triangle][:, 1].clone().detach().numpy()
                tri_obj[0].set_xy(np.column_stack((x, y)))
            if self.config.save_vectors:
                state_observations.append(self.get_state_observation())
            if self.config.opaque_model:
                if self.config.save_images:
                    plt.savefig(os.path.join(save_folder, f"im{frame}.png"))
                return [ln] + [ln_probe] + line_objs + link_objs + [obj for sublist in tri_objs for obj in sublist]
            else:
                for i, item in enumerate(zip(bars, self.get_state_observation()[16:])):
                    bar, new_height = item
                    new_height = np.sqrt(new_height[0] ** 2 + new_height[1] ** 2)
                    if self.plot_diff:
                        bar.set_height(new_height - self.last_heights[i])  # Update the height of each bar
                    else:
                        bar.set_height(new_height)
                    self.last_heights[i] = new_height
                arrows.set_offsets(np.column_stack((np.array(x_probe), np.array(y_probe))))
                arrows.set_UVC(np.array(self.get_state_observation()[16:, 0]),
                               np.array(self.get_state_observation()[16:, 1]))
                if self.config.save_images:
                    # set tight
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_folder, f"im{frame}.png"), dpi=400)

                return [ln] + [ln_probe] + line_objs + link_objs + [obj for sublist in tri_objs for obj in sublist] + [
                    bars] + [arrows]

        def update(frame):
            _, _, _, _ = self.update()
            if self.config.save_images:
                plt.savefig(os.path.join(save_folder, f"im{frame}.png"))
            if self.config.save_vectors:
                state_observations.append(self.get_state_observation())

        if save_folder:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
        print(self.config.frames)
        if self.config.save_video:
            ani = FuncAnimation(fig, update_vid, frames=self.config.frames, init_func=init,
                                blit=True if self.config.opaque_model else False, repeat=False)
            ani.save(os.path.join(save_folder, 'animation.gif'), writer='pillow', fps=30)
        else:
            for frame in range(self.config.frames):
                update(frame)
        if self.config.save_vectors and save_folder:
            with open(os.path.join(save_folder, 'vectors.pk'), 'wb') as f:
                pickle.dump(state_observations, f)
        plt.close()

    def draw_model(self, save_folder=None):
        # Draw a non-opaque model
        fig, ax = plt.subplots()
        x_object, y_object = self.shapes[0].vertices[:, 0].detach().numpy(), self.shapes[0].vertices[:,
                                                                             1].detach().numpy()
        # ax.plot(x_object, y_object, 'ro')
        for (i, j) in self.shapes[0].edges:
            ax.plot([x_object[i], x_object[j]], [y_object[i], y_object[j]], 'b-', alpha=0.2)
        # for (i, j) in self.shapes[0].links:
        #     ax.plot([x_object[i], x_object[j]], [y_object[i], y_object[j]], 'm-')
        triangles = self.shapes[0].triangles
        young_modulus = self.shapes[0].young_modulus
        colors = plt.cm.viridis(young_modulus * 50.0)
        for triangle, color in zip(triangles, colors):
            x = self.shapes[0].vertices[triangle][:, 0].clone().detach().numpy()
            y = self.shapes[0].vertices[triangle][:, 1].clone().detach().numpy()
            ax.fill(x, y, color=color, alpha=1.0, edgecolor=color)
        plt.axis('equal')  # Make the axis equal
        plt.axis('off')  # Remove the axes from the figure
        if save_folder:
            plt.savefig(os.path.join(save_folder, 'model_imaging.png'))
        else:
            plt.show()
        plt.close()
