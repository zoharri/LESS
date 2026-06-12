from dataclasses import dataclass, field
from typing import List


@dataclass
class TrajectoriesConfig:
    type: List[
        str]  # Types of trajectories (FixedVelocity, Return, PiecewiseLinear, FourPoint, ThreePoint, TwoPoint, Static)
    frames: List[int]  # Number of frames in the trajectories
    params: List[dict]  # Parameters defining the trajectories


@dataclass
class SimulationConfig:
    collision_spring_constant: float  # Spring constant for collision
    steps: int  # Number of simulation steps
    observation_noise: float  # Noise in observations
    dt: float  # Time step for the simulation
    save_folder: str  # Folder to save simulation results
    opaque_model: bool  # Flag indicating if the model is opaque
    probe_force_noise_std: float  # Standard deviation of the noise in probe force
    save_vectors: bool  # Flag indicating if vectors are saved
    save_images: bool  # Flag indicating if images are saved
    save_video: bool  # Flag indicating if video is saved
    frames: int  # Number of frames to run the simulation for, will be overwritten by the number of frames in the trajectory
    learning_rate: float  # Learning rate for the optimizer
    adam_beta_1: float  # Beta 1 parameter for Adam optimizer
    adam_beta_2: float  # Beta 2 parameter for Adam optimizer
    warmup: bool  # Flag indicating if the optimizer has a warmup of 5 steps with a larger learning rate


@dataclass
class LumpConfig:
    include_lump: bool = False  # Flag indicating if the lump is included
    radius: float = None  # Radius of the lump
    young_modulus: float = None  # Young's modulus of the lump
    poisson_ratio: float = None  # Poisson's ratio of the lump
    center: List[float] = field(default_factory=list)  # Center coordinates of the lump


@dataclass
class BreastModelConfig:
    grid_type: str  # Type of grid (square, square_with_middle_point, or hammersley)
    radius: int  # Radius of the breast model
    grid_size: float  # Size of the grid cells
    perimeter_grid_size: float  # Size of the perimeter grid cells
    num_inner_points: int  # Number of inner points in the grid
    poisson_ratio: float  # Poisson's ratio of the breast tissue
    poisson_ratio_var: float  # Variation in Poisson's ratio
    young_modulus: float  # Young's modulus of the breast tissue
    young_modulus_var: float  # Variation in Young's modulus
    random_vertices_std: float  # Standard deviation of the noise in vertices location
    lump: LumpConfig  # Configuration for the lump in the breast model


@dataclass
class ProbeConfig:
    num_points: int  # Number of points for the probe model
    radius: float  # Radius of the probe
    center: List[float]  # Center coordinates of the probe
    trajectories: TrajectoriesConfig  # Configuration for probe trajectories


@dataclass
class PalpConfig:
    breast_model: BreastModelConfig  # Configuration for the breast model
    probe: ProbeConfig  # Configuration for the probe
    simulation: SimulationConfig  # Configuration for the simulation


@dataclass
class EmbeddingsConfig:
    # Type of positional embedding to use (basic/ffn)
    type: str = field(default='basic')
    # Number of indices to encode
    num_idxs: int = field(default=3)
    # Encoding levels
    enc_levels: int = field(default=40)
    # Base num
    base: float = field(default=0.76)
    # Embedding fusion mode
    fusion_mode: str = field(default='concat')
    # Indices normalization mode, if None don't normalize indices (None/local/global)
    normalization_mode: str = field(default='local')
    # Gaussian kernel scale for ffn (fourier feature network)
    gauss_scale: List[float] = field(default_factory=lambda: [1, 0.1, 0.1])
