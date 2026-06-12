import os
from pathlib import Path
from typing import Tuple, Sequence, Optional

import pyvista as pv
import numpy as np
import torch

# Type aliases for readability
Vec3 = Tuple[float, float, float]
CameraPosition = Sequence[Vec3]


class visualize_inserts_3d:

    def __init__(self, window_size: (int, int) = (900, 900), off_screen: bool = True, name: str = "",
                 output_path: str = "screenshots", debug: bool = False, show_text: bool = False, look_up = False,
                 should_smooth_mesh = True):

        self.off_screen = off_screen

        self.plotter = pv.Plotter(window_size=window_size, off_screen=self.off_screen)
        self.plotter.set_background('paraviewbackground')
        # removes an error that doesn't affect us
        self.plotter.left_button_down = lambda *args, **kwargs: None
        self.should_smooth_mesh = should_smooth_mesh

        self.gray_actor = None
        self.original_gray_opacity = 0.3
        self.current_lump_opacity = 0.95
        self.current_pillar_opacity = 0.95
        self.current_gray_opacity = self.original_gray_opacity
        self.history = []  # keep track of arrays
        self.index = -1  # current array index
        self.name = name
        self.screenshot_counter = 0
        self.start_flag = False

        self.big_inserts = ["bst12d30", "bst12d15"]
        self.is_big = []
        # Setup output directory
        self.output_path = Path(output_path)
        self.output_dir = self.output_path / self.name
        self.showcase_path = self.output_dir / "showcase"
        self.html_path = self.output_dir / "HTML"

        self.debug = debug
        if self.debug:
            self.names = []  # keep track of names

            os.makedirs(self.output_path, exist_ok=True)
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(self.showcase_path, exist_ok=True)
            os.makedirs(self.html_path, exist_ok=True)

        # Default camera pos
        self.default_cpos = [(100.12834218186312, 150, 21.032991409301758),
                  (2.6999998092651367, 31.271705389022827, 31.032991409301758),
                  (1, 0, 0)]

        # cpos to look up
        if look_up:
            self.default_cpos = [(150.12834218186312, 31.271705389022827, 31.032991409301758),
                (2.6999998092651367, 31.271705389022827, 31.032991409301758),
                (0, -1, 0)]


        # for showcase
        self.showcased = set()  # track which indices already got a showcase screenshot
        self.showcase_cpos = self.default_cpos

        # Key bindings
        self.plotter.add_key_event("q", self.quit_program)
        self.plotter.add_key_event("g", self.toggle_gray)
        self.plotter.add_key_event("n", self.next_array)
        self.plotter.add_key_event("b", self.prev_array)
        self.plotter.add_key_event("r", self.reset_camera)
        self.plotter.add_key_event("s", self.save_screenshot)
        self.plotter.add_key_event("p", self.print_camera)
        self.plotter.add_key_event("h", self.export_to_html)

        self.show_text = show_text

    def get_name(self):
        return self.name

    def _make_meshes(self, image_stack: np.ndarray):
        """
        Convert a 3D image stack into PyVista meshes for different regions of interest
        (lump, insert, and pillar).

        Parameters
        ----------
        image_stack : np.ndarray
            A 3D NumPy array (k × h × w) representing the volumetric image stack.

        Returns
        -------
        tuple of pyvista.DataSet
            - lump_mesh : pyvista.UnstructuredGrid
                Mesh extracted by thresholding voxels with value of 255.
            - insert_mesh : pyvista.UnstructuredGrid
                Mesh extracted by thresholding voxels with value of 127.
            - pillar_mesh : pyvista.UnstructuredGrid
                Mesh extracted by thresholding voxels with value of 200.
        """

        k, h, w = image_stack.shape
        grid = pv.ImageData()
        grid.dimensions = image_stack.shape

        original_size = 80

        new_size = w
        new_spacing_y = 0.763889 * (original_size / new_size)
        new_spacing_x = 0.763889 * (original_size / new_size)
        new_spacing_z = 0.9

        if self.is_big[self.index]:
            new_spacing_z /= 2

        grid.spacing = (new_spacing_z, new_spacing_y, new_spacing_x)
        grid['values'] = image_stack.flatten(order='F')

        lump_mesh = grid.threshold(value=254)
        insert_mesh = grid.threshold(value=(126, 128))
        pillar_mesh = grid.threshold(value=(199, 201))

        return lump_mesh, insert_mesh, pillar_mesh

    def _reconstruct_tensor_to_np(self, model_images: torch.Tensor) -> np.ndarray:
        """
        Convert a 3D model_images tensor back to a NumPy array with original pixel values.

        Args:
            model_images: torch.Tensor of shape (C, K, H, W) or (K, H, W)
        Returns:
            np.ndarray with shape (K, H, W) with values:
                background = 0
                insert = 127
                pillar = 200
                lump = 255
        """
        if model_images.ndim != 4 and model_images.ndim != 3:
            raise ValueError(f"Expected a 3D tensor (C,K,H,W), got shape {model_images.shape}")

        if model_images.ndim == 4:
            C, K, H, W = model_images.shape

            reconstructed = np.zeros((K, H, W), dtype=np.uint8)  # background = 0

            reconstructed[model_images[1] == 0] = 127
            reconstructed[model_images[2] == 0] = 200
            reconstructed[model_images[3] == 0] = 255
        else:

            K, H, W = model_images.shape
            reconstructed = np.zeros((K, H, W), dtype=np.uint8)  # background = 0

            reconstructed[model_images == 1] = 127
            reconstructed[model_images == 2] = 200
            reconstructed[model_images == 3] = 255

        return reconstructed

    def load_array(self, image_stack, name: str = "", is_big_phantom: bool = False):
        """
          Load a 3D np array or 4D tensor array into history and optionally track its name in debug mode.

          Parameters
          ----------
          image_stack :
              3D array to store can be np.ndarray or a 3d or 4d tensor array.
          name : str, optional.
              Optional label for the array (default "").
        """
        if torch.is_tensor(image_stack):
            image_stack = self._reconstruct_tensor_to_np(image_stack)

        # Ensure 3D
        if len(image_stack.shape) != 3:
            print(f"Error: Array must be 3D. Current shape: {image_stack.shape}")
            return

        self.history.append(image_stack)
        self.is_big.append(is_big_phantom)

        if self.debug:
            self.names.append(name)

    def load_array_from_path(self, image_stack_path, name=""):
        """
        Load a 3D array from a .npy file and append it to history.

        Parameters
        ----------
        image_stack_path : str or Path.
            Path to the .npy file.
        name : str, optional.
            Optional label for the array (default "").
        """

        try:
            image_stack = np.load(image_stack_path)
        except FileNotFoundError:
            print(f"Error: The file '{image_stack_path}' was not found.")
            return

        self.load_array(image_stack, name=name)

    def load_folder(self, data_path):
        """
        Recursively load all .npy arrays from a folder into history.

        Parameters
        ----------
        data_path : str or Path
            Directory containing .npy files.

        Notes
        -----
        Automatically prints debug info for each loaded file if debug mode is enabled.
        """

        i = 1
        for (dirpath, dirnames, filenames) in os.walk(data_path):
            for f in filenames:
                if f.endswith(".npy"):
                    name = (f.split("_")[0])
                    self.load_array_from_path(os.path.join(dirpath, f), name)

                    if self.debug:
                        print(f"[DEBUG] {i : 3}. Found file: {os.path.join(dirpath, f)}")

                    i += 1

    def smooth_mesh(self, mesh, n_iter=100):
        if mesh is None or mesh.n_points == 0:
            return mesh

        # FIX: Convert UnstructuredGrid to PolyData (surface)
        if not isinstance(mesh, pv.PolyData):
            mesh = mesh.extract_surface()

        # Now you can smooth it
        return mesh.smooth(n_iter=n_iter, feature_smoothing=False)

    def _update_scene(self):
        """
        Clear the plotter and render the current array from history.
        """

        if self.index < 0 or self.index >= len(self.history):
            return

        image_stack = self.history[self.index]
        lump_mesh, insert_mesh, pillar_mesh = self._make_meshes(image_stack)

        self.plotter.clear()  # clear old actors

        if self.should_smooth_mesh:
            lump_mesh = self.smooth_mesh(lump_mesh, n_iter=300)
            insert_mesh = self.smooth_mesh(insert_mesh, n_iter=300)
            pillar_mesh = self.smooth_mesh(pillar_mesh, n_iter=300)

        if lump_mesh.n_points > 0:
            self.plotter.add_mesh(
                lump_mesh, color='red', opacity=self.current_lump_opacity, smooth_shading=True
            )

        if insert_mesh.n_points > 0:
            self.gray_actor = self.plotter.add_mesh(
                insert_mesh, color='gray', opacity=self.current_gray_opacity, smooth_shading=True, show_edges=True
            )

        if pillar_mesh.n_points > 0:
            self.plotter.add_mesh(pillar_mesh, color=(240, 240, 240), opacity=self.current_pillar_opacity, smooth_shading=True)

        # Info text
        if self.show_text:
            self.plotter.add_text(f"3D Visualization - {self.name}", font_size=15)
            self.plotter.add_text("Use the mouse to rotate, scroll to zoom, right-click to pan",
                                  font_size=8, position=(0.005, 0.92), viewport=True)
            self.plotter.add_text("Press 'n' for next array, 'b' for previous array",
                                  font_size=8, position=(0.005, 0.89), viewport=True)
            self.plotter.add_text("Press 'r' to reset camera",
                                  font_size=8, position=(0.005, 0.86), viewport=True)
            self.plotter.add_text("Press 'g' to toggle gray mesh opacity",
                                  font_size=8, position=(0.005, 0.83), viewport=True)
            self.plotter.add_text("Press 's' to save screenshot",
                                  font_size=8, position=(0.005, 0.80), viewport=True)
            self.plotter.add_text("Press 'h' to save the visualization as HTML",
                                  font_size=8, position=(0.005, 0.77), viewport=True)
            self.plotter.add_text("Press 'p' to print the camera location",
                                  font_size=8, position=(0.005, 0.74), viewport=True)
            self.plotter.add_text("Press 'q' to quit",
                                  font_size=8, position=(0.005, 0.71), viewport=True)

            # Array progress
            if self.debug:
                self.plotter.add_text(f"Array {self.index + 1}/{len(self.history)} - {self.names[self.index]}",
                                      font_size=12, position=(0.01, 0.95))
            else:
                self.plotter.add_text(f"Array {self.index + 1}/{len(self.history)}",
                                      font_size=12, position=(0.01, 0.95))

        self.plotter.render()

        if self.start_flag and self.index not in self.showcased:
            img = self._save_showcase()
            # HTML exports required an interactive GUI
            if not self.off_screen and self.debug:
                self.export_to_html()

            return img

    def reset_history(self):
        self.history = []
        self.index = -1
        self.showcased = set()

        if self.debug:
            self.names = []

    def toggle_gray(self):
        """
        Cycle the gray mesh opacity through preset values: 0.45 → 0.9 → 1 → 0.3 → 0.45.

        Notes
        -----
        Only affects the gray (insert) mesh. Automatically re-renders the scene.
        """

        if not self.gray_actor:
            return

        current_opacity = self.gray_actor.GetProperty().GetOpacity()
        if current_opacity == self.original_gray_opacity:
            self.gray_actor.GetProperty().SetOpacity(0.9)
        elif current_opacity == 0.9:
            self.gray_actor.GetProperty().SetOpacity(1)
        elif current_opacity == 1:
            self.gray_actor.GetProperty().SetOpacity(0.3)
        else:
            self.gray_actor.GetProperty().SetOpacity(self.original_gray_opacity)

        self.current_gray_opacity = self.gray_actor.GetProperty().GetOpacity()
        self.plotter.render()

    def next_array(self):
        """Go forward in history."""
        if self.index not in self.showcased and self.index == 0:
            self._save_showcase()
            self.export_to_html()

        if self.index < len(self.history) - 1:
            self.index += 1
            self._update_scene()

    def prev_array(self):
        """Go backward in history."""
        if self.index > 0:
            self.index -= 1
            self._update_scene()

    def _save_showcase(self):
        """Save a "showcase" screenshot of the current array using a fixed camera position."""
        self.showcased.add(self.index)

        # temporarily switch camera to showcase position
        old_cpos = self.plotter.camera_position
        self.plotter.camera_position = self.showcase_cpos
        self.plotter.render()  # force apply the camera

        # save screenshot at showcase view
        img = None
        if self.debug:

            if self.index % 2 == 0:
                idx = int(self.index / 2)
                filename = self.showcase_path / f"{self.name}_showcase_gt_{idx}.png"
            else:
                idx = int(self.index / 2)
                filename = self.showcase_path / f"{self.name}_showcase_pred_{idx}.png"

            self.plotter.screenshot(str(filename), return_img=False)
            print(f"[DEBUG] Saved showcase screenshot: {filename}")
        else:
            img = self.plotter.screenshot(return_img=True)

        # restore old camera
        self.plotter.camera_position = old_cpos
        self.plotter.render()

        return img

    def reset_camera(self):
        """Reset the camera to the default position."""
        self.plotter.camera_position = self.default_cpos

    def save_screenshot(self):
        """Save a PNG screenshot of the current scene."""
        self.screenshot_counter += 1
        filename = self.output_dir / f"{self.name}_screenshot_{self.screenshot_counter:03d}.png"
        self.plotter.screenshot(str(filename))

        if self.debug:
            print(f"[DEBUG] Saved screenshot: {filename}")

    def export_to_html(self, filename=None):
        """
        Export the current 3D scene to an interactive HTML file.

        Parameters
        ----------
        filename : str or Path, optional
            Specific filename for the HTML file. If None, a default name is used.
        """

        if filename is None:
            if self.index % 2 == 0:
                idx = int(self.index / 2)
                filename = self.html_path / f"{self.name}_html_gt_{idx}.html"
            else:
                idx = int(self.index / 2)
                filename = self.html_path / f"{self.name}_html_pred_{idx}.html"
        else:
            filename = Path(filename)
            filename = self.html_path / filename.with_suffix(".html")

        # Export the HTML visualization
        self.plotter.export_html(str(filename))

        # Add the array index to the generated HTML file
        if self.show_text:
            try:
                with open(filename, 'r') as file:
                    content = file.read()

                new_content = content.replace(
                    "<body>",
                    f"<body>\n    <p style='position:absolute; top: 10px; left: 10px; "
                    f"z-index: 10; font-family:sans-serif; font-size: 20px;'>"
                    f"Array {self.index + 1}</p>"
                )

                with open(filename, 'w') as file:
                    file.write(new_content)

                if self.debug:
                    print(f"[DEBUG] Saved HTML visualization with text to: {filename}")
            except FileNotFoundError:
                print(f"Error: The HTML file '{filename}' was not found after export.")
        else:
            if self.debug:
                print(f"[DEBUG] Saved HTML visualization without overlay to: {filename}")

    def print_camera(self):
        """Print the current camera position to the console."""
        print("\nCamera position:", self.plotter.camera_position)

    def quit_program(self):
        """Close the visualization window"""
        self.plotter.close()

        if self.debug:
            print("[DEBUG] Finished visualization")

    def show(self, cpos: Optional[CameraPosition] = None):
        """
        Start the interactive GUI visualization loop.

        Parameters
        ----------
        cpos : tuple, optional
            Camera position to use when starting. Defaults to `self.default_cpos`.

        Raises
        ------
        RuntimeError
            If no arrays have been loaded into history.
        """
        if not self.off_screen:
            if len(self.history) == 0:
                raise RuntimeError("No array to render")

            # Start at the first frame
            self.index = 0
            self._update_scene()
            self.start_flag = True

            self.plotter.show(auto_close=False, cpos=cpos or self.default_cpos)
        else:
            imgs = self.render_all()
            return imgs

    def render_all(self):
        """Loop through history and save screenshots + HTML for each array."""
        if len(self.history) == 0:
            return

        imgs = []
        self.start_flag = True
        for i in range(len(self.history)):
            self.index = i
            img = self._update_scene()
            imgs.append(img)

        return imgs

    def save_rotation_animation(self, filename: str = None, n_frames: int = 60, fps: int = 10, axis: str = 'z'):
        """
        Generates a 360-degree rotation GIF orbiting around a specific global axis.

        Parameters
        ----------
        filename : str, optional
            Output filename.
        n_frames : int
            Number of frames for a full 360 rotation.
        fps : int
            Frames per second.
        axis : str
            The global axis to rotate around: 'x', 'y', or 'z'.
        """
        if self.index < 0 or self.index >= len(self.history):
            print("Error: No array selected.")
            return

        self._update_scene()

        # --- 1. Setup Filename ---
        if filename is None:
            name_suffix = f"_{self.index}"
            if self.debug and self.index < len(self.names):
                name_suffix = f"_{self.names[self.index]}"
            os.makedirs(self.output_dir, exist_ok=True)
            filename = self.output_dir / f"{self.name}_rot_{axis}{name_suffix}.gif"
        else:
            filename = Path(filename)
            if not filename.parent.name:
                filename = self.output_dir / filename

        print(f"Generating {axis}-axis animation: {filename}...")

        # --- 2. Setup Rotation Math ---
        # Get current camera state
        original_cpos = self.plotter.camera_position
        pos = np.array(self.plotter.camera.position)
        focal = np.array(self.plotter.camera.focal_point)

        # CORRECTED: Use 'up' instead of 'view_up'
        view_up = np.array(self.plotter.camera.up)

        # Define the rotation axis vector
        axis = axis.lower()
        if axis == 'x':
            k = np.array([1.0, 0.0, 0.0])
        elif axis == 'y':
            k = np.array([0.0, 1.0, 0.0])
        else:  # z
            k = np.array([0.0, 0.0, 1.0])

        # Pre-calculate Rotation Matrix (Rodrigues' formula)
        theta = np.radians(360.0 / n_frames)
        K = np.array([
            [0.0, -k[2], k[1]],
            [k[2], 0.0, -k[0]],
            [-k[1], k[0], 0.0]
        ])
        # R = I + sin(theta)K + (1-cos(theta))K^2
        R = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)

        # Vector from focal point to camera position
        vec_pos = pos - focal

        # --- 3. Render Loop ---
        self.plotter.open_gif(str(filename), fps=fps)

        for _ in range(n_frames):
            # Rotate position vector and view_up vector
            vec_pos = R @ vec_pos
            view_up = R @ view_up

            # Apply new camera coordinates
            self.plotter.camera.position = focal + vec_pos

            # CORRECTED: Use 'up' property
            self.plotter.camera.up = view_up

            self.plotter.reset_camera_clipping_range()
            self.plotter.write_frame()

        self.plotter.close()

        # Restore original camera
        self.plotter.camera_position = original_cpos
        print("Animation saved.")

    def save_defined_orbit(self,
                           focal_point: Sequence[float],
                           distance: float,
                           axis: Sequence[float],
                           filename: str = None,
                           n_frames: int = 60,
                           fps: int = 20):
        """
        Creates a video of the camera orbiting a specific point at a specific distance
        around a specific axis.

        Parameters
        ----------
        focal_point : sequence
            (x, y, z) coordinates to look at.
        distance : float
            Radius of the orbit (distance from camera to focal_point).
        axis : sequence
            (x, y, z) vector defining the axis of rotation.
        filename : str
            Output filename (e.g. 'orbit.mp4' or 'orbit.gif').
        """

        # 1. Validation & Setup
        if self.index < 0 and len(self.history) > 0:
            self.index = 0

        self._update_scene()

        # Normalize the rotation axis
        axis = np.array(axis)
        axis = axis / np.linalg.norm(axis)
        focal_point = np.array(focal_point)

        # 2. Calculate Start Position
        # We want to start the orbit at the angle closest to the current camera position
        # so the transition isn't jarring.
        current_pos = np.array(self.plotter.camera.position)

        # Vector from focus to current camera
        vec_cam = current_pos - focal_point

        # Project this vector onto the plane defined by the axis
        # (remove the component parallel to the axis)
        vec_parallel = np.dot(vec_cam, axis) * axis
        vec_plane = vec_cam - vec_parallel

        # Normalize direction and scale to desired distance
        if np.linalg.norm(vec_plane) < 1e-6:
            # Fallback: if camera is exactly on the axis, pick an arbitrary start
            # (Create a vector perpendicular to axis)
            arbitrary = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            vec_plane = np.cross(axis, arbitrary)

        start_direction = vec_plane / np.linalg.norm(vec_plane)
        start_pos = focal_point + (start_direction * distance)

        # 3. Setup Rotation Matrix (Rodrigues' formula)
        theta = np.radians(360.0 / n_frames)
        K = np.array([
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0]
        ])
        R = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)

        # 4. Handle Filename
        if filename is None:
            filename = self.output_dir / f"{self.name}_orbit_custom.mp4"
        else:
            filename = Path(filename)
            if not filename.parent.name:
                filename = self.output_dir / filename

        print(f"Generating orbit video: {filename}...")

        # Save state
        old_cpos = self.plotter.camera_position

        # 5. Animation Loop
        # Determine strict file type (.gif or .mp4)
        if str(filename).endswith('.gif'):
            self.plotter.open_gif(str(filename), fps=fps)
        else:
            self.plotter.open_movie(str(filename), framerate=fps)

        # Current vector from focus to camera
        current_vec = start_pos - focal_point

        for _ in range(n_frames):
            # Rotate the vector
            current_vec = R @ current_vec

            # Update Camera
            self.plotter.camera.position = focal_point + current_vec
            self.plotter.camera.focal_point = focal_point

            # Align 'up' with rotation axis for stability
            self.plotter.camera.up = axis

            self.plotter.reset_camera_clipping_range()
            self.plotter.write_frame()

        self.plotter.close()

        # Restore
        self.plotter.camera_position = old_cpos
        print("Video saved.")

    def save_x_orbit_default(self, filename: str = None, n_frames: int = 60, fps: int = 10):
        """
        Creates an animation orbiting the X-axis using the focal point and distance
        defined in self.default_cpos.
        """
        if self.index < 0:
            print("Error: No array selected.")
            return

        self._update_scene()

        # --- 1. Extract Geometry from Defaults ---
        # default_cpos is a list of tuples: [position, focal_point, view_up]
        def_pos = np.array(self.default_cpos[0])
        def_focus = np.array(self.default_cpos[1])

        # Calculate the zoom level (distance) from the default position
        distance = np.linalg.norm(def_pos - def_focus)

        # Set the axis strictly to X
        axis = np.array([1.0, 0.0, 0.0])

        # --- 2. Setup Start Position ---
        # We start at the default position to ensure the video begins comfortably
        # projected onto the perfect circular orbit path.
        current_vec = def_pos - def_focus

        # --- 3. Setup Rotation Matrix (Rodrigues) ---
        theta = np.radians(360.0 / n_frames)
        # Cross product matrix for X-axis (1,0,0)
        K = np.array([
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0]
        ])
        # Rotation matrix
        R = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)

        # --- 4. Filename & Writer ---
        if filename is None:
            name_suffix = f"_{self.index}" if not self.debug else f"_{self.names[self.index]}"
            filename = self.output_dir / f"{self.name}_orbit_X{name_suffix}.mp4"

        print(f"Generating X-axis orbit around {def_focus} at dist {distance:.1f}...")

        # Save state
        old_cpos = self.plotter.camera_position

        # Determine writer (GIF vs MP4)
        if str(filename).endswith('.gif'):
            self.plotter.open_gif(str(filename), fps=fps)
        else:
            self.plotter.open_movie(str(filename), framerate=fps)

        # --- 5. Render Loop ---
        for _ in range(n_frames):
            # Rotate vector
            current_vec = R @ current_vec

            # Update camera
            self.plotter.camera.position = def_focus + current_vec
            self.plotter.camera.focal_point = def_focus

            # Vital: Lock 'up' to X-axis to prevent camera rolling/jittering
            self.plotter.camera.up = axis

            self.plotter.reset_camera_clipping_range()
            self.plotter.write_frame()

        self.plotter.close()

        # Restore old camera view
        self.plotter.camera_position = old_cpos
        print(f"Saved: {filename}")

    def save_evolving_rotation_video(self, filename="timelapse.mp4", frames_per_array=15, fps=30, rotate=False):
        """
        Creates a timelapse animation where the camera orbits the X-axis (using default_cpos logic).
        The array (object) changes every 'frames_per_array' frames, while the camera rotation
        remains continuous.
        """
        if not self.history:
            print("Error: No history loaded.")
            return

        # --- 1. Extract Geometry from Defaults (Your working logic) ---
        # default_cpos is [position, focal_point, view_up]
        def_pos = np.array(self.default_cpos[0])
        def_focus = np.array(self.default_cpos[1])

        # Set the axis strictly to X
        axis = np.array([1.0, 0.0, 0.0])

        # Setup Start Vector relative to focus
        current_vec = def_pos - def_focus

        # --- 2. Calculate Rotation Parameters ---
        # We want one full 360 orbit over the entire duration of the history
        total_frames = len(self.history) * frames_per_array
        theta = np.radians(360.0 / total_frames)

        # Cross product matrix for X-axis (1,0,0) - Rodrigues' rotation formula
        K = np.array([
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0]
        ])
        # Rotation matrix for a single step
        R = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)

        # --- 3. Setup Video Writer ---
        filepath = self.output_dir / filename

        # Save current camera state to restore later
        old_cpos = self.plotter.camera_position

        # Initialize the movie
        print(f"Generating evolving X-orbit ({len(self.history)} arrays, {total_frames} frames) to {filepath}...")
        self.plotter.open_movie(str(filepath), framerate=fps)

        # --- 4. Main Batch Loop ---
        for i in range(len(self.history)):
            # A. Switch the Data
            self.index = i
            self._update_scene()

            # B. Render frames for this specific array
            for _ in range(frames_per_array):
                # Apply Rotation
                if rotate:
                    current_vec = R @ current_vec

                # Update Camera
                self.plotter.camera.position = def_focus + current_vec
                self.plotter.camera.focal_point = def_focus
                self.plotter.camera.up = axis

                # Ensure clipping is correct so mesh doesn't disappear
                self.plotter.reset_camera_clipping_range()

                # Write Frame
                self.plotter.write_frame()

        # --- 5. Cleanup ---
        self.plotter.close()
        self.plotter.camera_position = old_cpos
        print("Done.")

    def get_plotly_traces(self, arr, prefix: str = "", is_big: bool = False) -> list:
        """
        Convert a 3D label array to Plotly Mesh3d traces using the same PyVista
        mesh pipeline as _make_meshes.

        Parameters
        ----------
        arr : np.ndarray or torch.Tensor
            Shape (K, H, W). Either class indices (0=bg,1=insert,2=pillar,3=lump)
            or pixel values (0, 127, 200, 255).
        prefix : str
            Legend label prefix ("GT" / "Pred").
        is_big : bool
            Whether this is a big phantom (affects z-spacing in _make_meshes).
        """
        import plotly.graph_objects as go

        if torch.is_tensor(arr):
            arr = self._reconstruct_tensor_to_np(arr)
        elif isinstance(arr, np.ndarray) and arr.max() <= 10:
            arr = self._reconstruct_tensor_to_np(arr)

        old_history, old_is_big, old_index = self.history, self.is_big, self.index
        self.history = [arr]
        self.is_big  = [is_big]
        self.index   = 0
        try:
            lump_mesh, insert_mesh, pillar_mesh = self._make_meshes(arr)
        finally:
            self.history, self.is_big, self.index = old_history, old_is_big, old_index

        traces = []
        for mesh, color, name, opacity in [
            (lump_mesh,   'red',     'Lump',   0.9),
            (insert_mesh, '#888888', 'Insert', 0.3),
            (pillar_mesh, '#e0e0e0', 'Pillar', 0.9),
        ]:
            if mesh is None or mesh.n_points == 0:
                continue
            surf  = mesh.extract_surface(algorithm='dataset_surface').triangulate()
            pts   = np.array(surf.points)
            faces = surf.faces.reshape(-1, 4)[:, 1:]
            lbl   = f"{prefix} {name}".strip()
            traces.append(go.Mesh3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color=color, opacity=opacity,
                name=lbl, showlegend=True, legendgroup=lbl,
            ))
        return traces


