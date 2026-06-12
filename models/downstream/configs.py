from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class BaseDownstreamModelConfig:
    dropout: float = 0.0
    input_dropout: float = 0.0


@dataclass
class ImagingDownstreamModelConfig(BaseDownstreamModelConfig):
    num_bins: int = 3
    image_size: int = 128
    num_channels: int = 256
    amount_of_slices: int = 1
    non_background_threshold: float = 0.0
    inference_on_big_phantom: bool = False
    balance_image_classes: bool = False
    loss: str = "cross_entropy"


@dataclass
class TransposedConvImagePredDownstreamModelConfig(ImagingDownstreamModelConfig):
    num_channels: int = 256


@dataclass
class TransposedConvImagePred3DDownstreamModelConfig(ImagingDownstreamModelConfig):
    num_channels: int = 256
    amount_of_slices: int = 1


@dataclass
class FlowMatchingImagePredDownstreamModelConfig(ImagingDownstreamModelConfig):
    num_channels: int = 256
    flow_steps: int = 1000
    use_residual_in_unet_flowmatching: bool = False


@dataclass
class UNetMapRepPredDownstreamModelConfig(ImagingDownstreamModelConfig):
    hidden_size: int = 64


@dataclass
class LocalCNNMapRepPredDownstreamModelConfig(ImagingDownstreamModelConfig):
    cnn_down_multiplier: int = 1
    cnn_usebn: bool = False
    cnn_kernel_size: int = 3
    use1x1: bool = False
    local_cnn_use_aug: bool = False


@dataclass
class CNNMapRepPredDownstreamModelConfig(ImagingDownstreamModelConfig):
    num_channels: int = 256


@dataclass
class AutoCNNMapRepPredDownstreamModelConfig(ImagingDownstreamModelConfig):
    num_channels: int = 256


@dataclass
class TransposedConvParRepPredDownstreamModelConfig(ImagingDownstreamModelConfig):
    num_channels: int = 256
    particle_image_pred_size: int = 16
    add_alpha_channel: bool = False
    particle_addition_mode: str = "sum"
    unlimited_area: bool = False


@dataclass
class TransposedConvParRepPred3DDownstreamModelConfig(ImagingDownstreamModelConfig):
    num_channels: int = 256
    amount_of_slices: int = 1
    particle_image_pred_size: int = 16
    add_alpha_channel: bool = False
    particle_addition_mode: str = "sum"
    unlimited_area: bool = False


@dataclass
class MLPQuantityDownstreamModelConfig(BaseDownstreamModelConfig):
    num_bins: int = 3
    image_size: int = 128
    num_channels: int = 256
    num_layers: int = 1


@dataclass
class DownstreamModelConfig:
    name: str = "TransposedConvImagePred"
    transposed_conv_image_pred: TransposedConvImagePredDownstreamModelConfig = field(
        default_factory=TransposedConvImagePredDownstreamModelConfig
    )
    transposed_conv_image_pred_3d: TransposedConvImagePred3DDownstreamModelConfig = field(
        default_factory=TransposedConvImagePred3DDownstreamModelConfig
    )
    flow_matching_image_pred: FlowMatchingImagePredDownstreamModelConfig = field(
        default_factory=FlowMatchingImagePredDownstreamModelConfig
    )
    unet_map_rep_pred: UNetMapRepPredDownstreamModelConfig = field(
        default_factory=UNetMapRepPredDownstreamModelConfig
    )
    local_cnn_map_rep_pred: LocalCNNMapRepPredDownstreamModelConfig = field(
        default_factory=LocalCNNMapRepPredDownstreamModelConfig
    )
    cnn_map_rep_pred: CNNMapRepPredDownstreamModelConfig = field(
        default_factory=CNNMapRepPredDownstreamModelConfig
    )
    auto_cnn_map_rep_pred: AutoCNNMapRepPredDownstreamModelConfig = field(
        default_factory=AutoCNNMapRepPredDownstreamModelConfig
    )
    transposed_conv_par_rep_pred: TransposedConvParRepPredDownstreamModelConfig = field(
        default_factory=TransposedConvParRepPredDownstreamModelConfig
    )
    transposed_conv_par_rep_pred_3d: TransposedConvParRepPred3DDownstreamModelConfig = field(
        default_factory=TransposedConvParRepPred3DDownstreamModelConfig
    )
    mlp_quantity_model: MLPQuantityDownstreamModelConfig = field(default_factory=MLPQuantityDownstreamModelConfig)

    def _name_to_attr(self) -> Dict[str, str]:
        return {
            "TransposedConvImagePred": "transposed_conv_image_pred",
            "TransposedConvImagePred3D": "transposed_conv_image_pred_3d",
            "FlowMatchingImagePred": "flow_matching_image_pred",
            "UNetMapRepPred": "unet_map_rep_pred",
            "LocalCNNMapRepPred": "local_cnn_map_rep_pred",
            "CNNMapRepPred": "cnn_map_rep_pred",
            "AutoCNNMapRepPred": "auto_cnn_map_rep_pred",
            "TransposedConvParRepPred": "transposed_conv_par_rep_pred",
            "TransposedConvParRepPred3D": "transposed_conv_par_rep_pred_3d",
            "MLPLumpCenterPredictor": "mlp_quantity_model",
            "MLPLumpAreaPredictor": "mlp_quantity_model",
            "MLPPhantomIndexClassifier": "mlp_quantity_model",
            "PhantomIndexClassifier": "mlp_quantity_model",
        }

    def get_active_config(self) -> BaseDownstreamModelConfig:
        mapping = self._name_to_attr()
        if self.name not in mapping:
            raise ValueError(
                f"Unknown downstream model name: {self.name}. "
                f"Supported: {list(mapping.keys())}"
            )
        return deepcopy(getattr(self, mapping[self.name]))
