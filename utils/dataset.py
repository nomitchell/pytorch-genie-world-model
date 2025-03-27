import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

class CoinRunDataset(Dataset):
    def __init__(self, data_dir, image_size=64, history_length=16, transform=None):
        self.data_dir = data_dir
        self.image_size = image_size
        self.history_length = history_length
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Get all episode directories
        self.episodes = [os.path.join(data_dir, d) for d in os.listdir(data_dir) 
                        if os.path.isdir(os.path.join(data_dir, d))]
        
        # Build index mapping (episode_idx, frame_idx) -> dataset_idx
        self.index_map = []
        for episode_idx, episode_dir in enumerate(self.episodes):
            frame_files = sorted([f for f in os.listdir(episode_dir) if f.endswith('.png')])
            action_file = os.path.join(episode_dir, 'actions.npy')
            
            # Check if we have enough frames and the actions file
            if len(frame_files) >= history_length + 1 and os.path.exists(action_file):
                for i in range(len(frame_files) - history_length):
                    self.index_map.append((episode_idx, i))
    
    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, idx):
        episode_idx, frame_idx = self.index_map[idx]
        episode_dir = self.episodes[episode_idx]
        
        # Load frames
        frames = []
        for i in range(self.history_length + 1):  # +1 for the next frame
            frame_path = os.path.join(episode_dir, f"frame_{frame_idx + i:05d}.png")
            frame = Image.open(frame_path).convert('RGB')
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        
        # Load action
        # In the paper they use the LAM to predict actions, here I train the lam but cheat and use the actions from the dataset for better results, you can toggle this
        actions = np.load(os.path.join(episode_dir, 'actions.npy'))
        action = actions[frame_idx + self.history_length - 1] 
        
        history_frames = torch.stack(frames[:-1])  # (history_length, C, H, W)
        next_frame = frames[-1]  # Last frame is the next frame
        
        return {
            'history_frames': history_frames,  # (history_length, C, H, W)
            'next_frame': next_frame,          # (C, H, W)
            'action': action                   # Integer action ID
        }
