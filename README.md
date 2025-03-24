# World Model for CoinRun Environment

This project implements a world model for the CoinRun environment using a combination of a video tokenizer, latent action model, and a MaskGIT-based dynamics model. The system can learn to predict future game states based on historical observations and actions.

*Example of CoinRun gameplay and model predictions*

## Architecture

The system consists of three main components:

1. **VQ-VAE (Vector Quantized Variational Autoencoder)**
   - Compresses game frames into discrete latent tokens

2. **Latent Action Model (LAM)**
   - Learns to predict actions from state transitions

3. **MaskGIT-based Dynamics Model**
   - Predicts future states using masked token prediction

## Results

### VQ-VAE Reconstruction Quality
*Comparison of original frames and VQ-VAE reconstructions*

### Single-Step Prediction
*Single-step prediction results with PSNR and SSIM metrics*

### Multi-Step Prediction
*Multi-step prediction showing error accumulation over time*

### Codebook Usage Analysis
*Distribution of codebook token usage*

## Usage

### Training
```bash
python main.py
```

### Evaluation
```bash
python eval.py
```

### Interactive Testing (Play)
```bash
python test.py
```

## Model Configuration

Key hyperparameters can be modified in the `Config` class:

```python
class Config:
    image_size = 64
    vq_embedding_dim = 64
    vq_num_embeddings = 512
    hidden_dim = 256
    num_heads = 4
    num_layers = 4
    history_length = 16
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- Procgen
- OpenCV
- Pygame
- NumPy
- Matplotlib

## Acknowledgments

- Based on Deepmind's Genie paper