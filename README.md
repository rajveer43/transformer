# PyTorch Transformer Implementation

A clean and educational implementation of the Transformer architecture based on the ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) paper. This implementation provides a simplified yet fully functional version of the transformer model, similar to PyTorch's official implementation but with added explanations and clarity.

## Overview

This project implements the complete Transformer architecture with both encoder and decoder components. The implementation is designed to be educational and includes detailed comments explaining the key concepts and computations.

### Key Components

- **Transformer**: The main model combining encoder and decoder
- **PositionalEncoding**: Implements sinusoidal positional encodings
- **MultiheadAttention**: Complete implementation of multi-head attention mechanism
- **TransformerEncoder/Decoder**: Stackable encoder and decoder layers

## Architecture Details

### Model Parameters

- `d_model`: Embedding dimension (default: 512)
- `nhead`: Number of attention heads (default: 8)
- `num_encoder_layers`: Number of encoder layers (default: 6)
- `num_decoder_layers`: Number of decoder layers (default: 6)
- `dim_feedforward`: Dimension of feedforward network (default: 2048)
- `dropout`: Dropout probability (default: 0.1)
- `layer_norm_eps`: Layer normalization epsilon (default: 1e-5)
- `batch_first`: Whether to use batch-first format (default: False)

### Key Features

1. **Positional Encoding**
   - Implements sinusoidal position embeddings
   - Supports both batch-first and sequence-first formats
   - Configurable maximum sequence length

2. **Multi-Head Attention**
   - Scaled dot-product attention implementation
   - Support for attention masking and key padding masks
   - Proper scaling of attention scores
   - Configurable number of attention heads

3. **Transformer Layers**
   - Layer normalization
   - Residual connections
   - Feed-forward networks with ReLU activation
   - Support for both pre-norm and post-norm architectures

## Usage Example

```python
# Create a transformer model
transformer = Transformer(
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    dropout=0.1
)

# Create input tensors
src = torch.rand(10, 32, 512)  # (sequence_length, batch_size, d_model)
tgt = torch.rand(20, 32, 512)  # (target_length, batch_size, d_model)

# Optional masks
src_mask = None
tgt_mask = transformer.generate_square_subsequent_mask(tgt.size(0))

# Forward pass
output = transformer(src, tgt, src_mask, tgt_mask)
```

## Implementation Notes

1. **Attention Computation**
   - Uses scaled dot-product attention
   - Proper handling of attention masks
   - Efficient batch matrix multiplications

2. **Layer Organization**
   - Modular design with separate encoder and decoder components
   - Efficient parameter sharing through cloning
   - Support for different normalization strategies

3. **Positional Encoding**
   - Implements the original sinusoidal encoding
   - Supports different sequence lengths
   - Proper dropout implementation

## Requirements

- PyTorch >= 1.0
- Python >= 3.6

## References

1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
2. [PyTorch's Transformer Implementation](https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/transformer.py)

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is provided as-is under the MIT License.
