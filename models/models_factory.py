from models.representation_learning import *
from models.downstream.predictors import *
from models.model_config import ModelConfig

class RepModelFactory:
    """
    Factory for creating representation learning models.
    
    This factory supports the following model types:
    - ReconstructionModel: Neural network models for force reconstruction
    - ForceMapModel: Non-learnable model that generates force maps using KDE
    """

    @staticmethod
    def generate_model(config: ModelConfig) -> RepresentationLearningModel:
        """
        Generate a representation learning model based on the configuration.
        
        Args:
            config: ModelConfig containing the model type and parameters path
            
        Returns:
            A representation learning model instance
            
        Raises:
            ValueError: If the model type is not supported
        """
        model_type = config.name
        params_path = config.params_path

        if model_type == "ReconstructionModel":
            model = ReconstructionModel(params_path)
        elif model_type == "LocalReconstructionModel":
            model = LocalReconstructionModel(params_path)
        elif model_type == "ParticlesReconstructionModel":
            model = ParticlesReconstructionModel(params_path)
        elif model_type == "ForceMapModel":
            model = ForceMapModel(params_path)
        elif model_type == "ParticlesAggregation":
            model = ParticlesAggregation(params_path)
        else:
            raise ValueError(f"Unknown representation learning model type: {model_type}. "
                             f"Supported types: ReconstructionModel, ForceMapModel")

        return model.to(model.device)


class DownstreamModelFactory:
    """
    Factory for creating downstream models.
    """

    @staticmethod
    def generate_model(config: ModelConfig, representation_size: int) -> DownstreamModel:
        """
        Generate a downstream model based on the configuration.
        """
        model_type = config.name
        params_path = config.params_path

        if model_type == "TransposedConvImagePred":
            model = TransposedConvImagePred(params_path, representation_size)
        elif model_type == "TransposedConvImagePred3D":
            model = TransposedConvImagePred3D(params_path, representation_size)
        elif model_type == "FlowMatchingImagePred":
            model = FlowMatchingImagePred(params_path, representation_size)
        elif model_type == "UNetMapRepPred":
            model = UNetMapRepPred(params_path, representation_size)
        elif model_type == "LocalCNNMapRepPred":
            model = LocalCNNMapRepPred(params_path, representation_size)
        elif model_type == "CNNMapRepPred":
            model = CNNMapRepPred(params_path, representation_size)
        elif model_type == "AutoCNNMapRepPred":
            model = AutoCNNMapRepPred(params_path, representation_size)
        elif model_type == "TransposedConvParRepPred":
            model = TransposedConvParRepPred(params_path, representation_size)
        elif model_type == "TransposedConvParRepPred3D":
            model = TransposedConvParRepPred3D(params_path, representation_size)
        elif model_type == "MLPLumpCenterPredictor":
            model = MLPLumpCenterReg(params_path, representation_size)
        elif model_type == "MLPLumpAreaPredictor":
            model = MLPLumpAreaReg(params_path, representation_size)
        elif model_type == "MLPPhantomIndexClassifier":
            model = MLPPhantomIndexClassifier(params_path, representation_size)
        else:
            raise ValueError(f"Unknown downstream model type: {model_type}")
        return model.to(model.device)
