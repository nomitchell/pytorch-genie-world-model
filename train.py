import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.dyn import MaskGIT
from models.lam import LatentActionModel
from models.vt import VQVAEModel

from config import Config
from utils.dataset import CoinRunDataset
from utils.train_func import train_lam_dynamics, train_vqvae
from utils.util import collect_coinrun_data

# Main function
def main():
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = Config()

    # Set random seeds for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Create data directory
    data_dir = "coinrun_data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Collect data if needed
    if len(os.listdir(data_dir)) == 0:
        print("Collecting CoinRun data...")
        collect_coinrun_data(data_dir)
    
    # Create dataset and dataloader
    dataset = CoinRunDataset(data_dir, image_size=config.image_size, history_length=config.history_length)
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # Initialize models
    vqvae_model = VQVAEModel(config)
    lam_model = LatentActionModel(config)
    dynamics_model = MaskGIT(config)

    if os.path.exists("vqvae_model.pt"):
        print("Loading pre-trained vt model...")
        vqvae_model.load_state_dict(torch.load("vqvae_model.pt"))
    else:
        print("Training VQVAE...")
        vqvae_model = train_vqvae(vqvae_model, train_loader, config)
        torch.save(vqvae_model.state_dict(), "vqvae_model.pt")
    
    # Check if models exist and load them, otherwise train
    if os.path.exists("lam_model.pt") and os.path.exists("dynamics_model.pt"):
        print("Loading pre-trained lam dyn models...")
        lam_model.load_state_dict(torch.load("lam_model.pt"))
        dynamics_model.load_state_dict(torch.load("dynamics_model.pt"))
    else:
        print("Jointly training Latent Action Model and Dynamics Model...")
        lam_model, dynamics_model = train_lam_dynamics(vqvae_model, lam_model, dynamics_model, train_loader, config)
        torch.save(lam_model.state_dict(), "lam_model.pt")
        torch.save(dynamics_model.state_dict(), "dynamics_model.pt")

if __name__ == "__main__":
    main()