import os
import gym
from PIL import Image
import numpy as np

def collect_coinrun_data(output_dir, num_episodes=5, max_steps=1000):
    os.makedirs(output_dir, exist_ok=True)
    
    env = gym.make('procgen:procgen-coinrun-v0')
    
    for episode in range(num_episodes):
        episode_dir = os.path.join(output_dir, f'episode_{episode:03d}')
        os.makedirs(episode_dir, exist_ok=True)
        
        obs = env.reset()
        actions = []
        
        for step in range(max_steps):
            # Save observation as PNG
            img = Image.fromarray(obs)
            img.save(os.path.join(episode_dir, f'frame_{step:05d}.png'))
            
            # Take random action
            # TODO, repeat action selected for better representation 
            action = env.action_space.sample()
            obs, _, done, _ = env.step(action)
            
            actions.append(action)
            
            if done:
                break
        
        # Save actions
        np.save(os.path.join(episode_dir, 'actions.npy'), np.array(actions))
    
    env.close()