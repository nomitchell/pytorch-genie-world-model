# World Model for CoinRun Environment

This project implements a world model for the CoinRun environment using a combination of a video tokenizer, latent action model, and a MaskGIT-based dynamics model. The system can learn to predict future game states based on historical observations and actions.

![Untitledvideo-MadewithClipchamp-ezgif com-video-to-gif-converter (1)](https://github.com/user-attachments/assets/0a6d40fc-f7e7-4bd5-be01-b747d09680fa)
*Example of CoinRun gameplay and model predictions*

## Architecture

The system consists of three main components:

1. **Video Tokenizer**
   - Compresses game frames into discrete latent tokens

2. **Latent Action Model**
   - Learns to predict actions from state transitions

3. **Dynamics Model**
   - Predicts future states using masked token prediction

## Results

### Video Tokenizer Reconstruction Quality
![vqvae_recon_sample_1](https://github.com/user-attachments/assets/048938e6-c894-4eb5-8604-6df49ec837d5)
*Comparison of original frames and VQ-VAE reconstructions*

### Single-Step Dynamics Model Prediction
![single_step_pred_sample_16](https://github.com/user-attachments/assets/97fe91e0-f53f-4e31-b084-69a0cceba98b)
*Single-step prediction results with PSNR and SSIM metrics*

### Multi-Step Dynamics Model Prediction
![multi_step_prediction_frames](https://github.com/user-attachments/assets/ec2ffa00-17da-410b-9f52-630c7e9232e7)
*Multi-step prediction showing error accumulation over time*

### Video Tokenizer Codebook Usage Analysis
![codebook_usage_histogram](https://github.com/user-attachments/assets/9d183f53-b2e4-4491-844c-614802b77fa7)
*Distribution of codebook token usage*

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

```bash
pip install -r requirements.txt
```

## Acknowledgments

- Based on Deepmind's Genie paper
