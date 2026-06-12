import torch
from torch import nn

from models.representation_learning.positional_encoding.encoding import FreqEncoder_torch

class VectorLocationEncoder(nn.Module):
    """
    Encodes vectors and locations into a combined representation using positional encoding.

    This module combines force vectors with location information using positional encoding
    to create rich representations that capture both the force magnitude and spatial context.

    Args:
        locations_size (int): Size of the locations vectors
        pe_max_freq_log2 (int): Maximum frequency for positional encoding
        vector_encoding_size (int): Size of the output encoding for vectors
        vector_size (int): Size of the input vectors to be encoded
    """
    """
    Encodes vectors and locations into a combined representation using positional encoding.
    Args:
    locations_size (int): Size of the locations vectors
    pe_max_freq_log2 (int): Maximum frequency for positional encoding.
    vector_encoding_size (int): Size of the output encoding for vectors.
    vector_size (int): Size of the input vectors to be encoded.
    """

    def __init__(self, locations_size: int, pe_max_freq_log2: int, vector_encoding_size: int, vector_size: int):
        """
        Initializes the VectorLocationEncoder.
        Args:
        locations_size (int): Size of the locations vectors.
        pe_max_freq_log2 (int): Maximum frequency for positional encoding.
        vector_encoding_size (int): Size of the output encoding for vectors.
        vector_size (int): Size of the input vectors to be encoded.
        Raises:
        ValueError: If vector_encoding_size is not divisible by locations_size * 2.
        """
        super(VectorLocationEncoder, self).__init__()
        assert vector_encoding_size % (locations_size * 2) == 0, ValueError(
            f"vector size ({vector_encoding_size}) must be divisible by locations_size * 2 ({locations_size * 2})")
        self.vector_encoding_size = vector_encoding_size
        self.pe = FreqEncoder_torch(input_dim=locations_size, max_freq_log2=pe_max_freq_log2,
                                    N_freqs=vector_encoding_size // (locations_size * 2),
                                    log_sampling=True, include_input=False)

        self.vector_encoder = nn.Linear(vector_size, vector_encoding_size)

    def forward(self, vectors: torch.Tensor, locations: torch.Tensor) -> torch.Tensor:
        locations_mean = locations.mean(dim=1)
        pe_out = self.pe(locations_mean)
        force_encodings = self.vector_encoder(vectors)
        return pe_out + force_encodings

    def output_size(self) -> int:
        return self.vector_encoding_size