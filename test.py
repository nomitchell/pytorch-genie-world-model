import os
import torch
import numpy as np
import pygame
import cv2
from datetime import datetime

from main import (
    Config, 
    VQVAEModel, 
    MaskGIT, 
    CoinRunDataset
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def test_saved_models():
    config = Config()
    
    data_dir = "coinrun_data"
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} not found. Please run data collection first.")
    
    vqvae_model = VQVAEModel(config)
    dynamics_model = MaskGIT(config)
    
    model_files = {
        "vqvae_model.pt": vqvae_model,
        "dynamics_model.pt": dynamics_model
    }
    
    for file_name, model in model_files.items():
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"Model file {file_name} not found. Please train the models first.")
        
        state_dict = torch.load(file_name, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
    
    print("All models loaded successfully!")
    
    pygame.init()
    
    pygame.display.set_mode((256, 256), pygame.HWSURFACE | pygame.DOUBLEBUF)
    
    pygame.display.set_caption("Testing pygame...")
    pygame.time.wait(500)  # Wait a bit to ensure window is shown
    
    print("Starting interactive evaluation...")
    
    evaluate_models_enhanced(vqvae_model, dynamics_model, config, max_steps=1000)

def evaluate_models_enhanced(vqvae_model, dynamics_model, config, max_steps=1000):
    """Enhanced evaluation with continuous rendering and support for combined actions"""
    import time
    import random
    
    videos_dir = "recorded_videos"
    os.makedirs(videos_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(videos_dir, f"gameplay_{timestamp}.mp4")
    
    render_size = (256, 256)
    fps = 10
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, render_size)
    print(f"Recording video to: {video_path}")
    
    screen = pygame.display.set_mode(render_size)
    pygame.display.set_caption("Enhanced Genie Model Demo")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)
    
    vqvae_model.eval()
    dynamics_model.eval()
    
    dataset = CoinRunDataset("coinrun_data", image_size=config.image_size, 
                            history_length=config.dynamics_context_frames)
    
    sample_idx = random.randint(0, len(dataset)-1)
    sample = dataset[sample_idx]
    initial_frames = sample['history_frames'].to(device)
    
    with torch.no_grad():
        frame_history = []
        for i in range(initial_frames.size(0)):
            tokens = vqvae_model.encode(initial_frames[i:i+1])
            frame_history.append(tokens)
        
        current_tokens = frame_history[-1]
        current_frame = vqvae_model.decode(current_tokens)
    
    coinrun_actions = [
        ("LEFT", "DOWN"),    # 0
        ("LEFT",),           # 1
        ("LEFT", "UP"),      # 2
        ("DOWN",),           # 3
        (),                  # 4 - No action
        ("UP",),             # 5
        ("RIGHT", "DOWN"),   # 6
        ("RIGHT",),          # 7
        ("RIGHT", "UP"),     # 8
        ("D",),              # 9
        ("A",),              # 10
        ("W",),              # 11
        ("S",),              # 12
        ("Q",),              # 13
        ("E",),              # 14
    ]
    
    keys_pressed = {
        "LEFT": False,
        "RIGHT": False,
        "UP": False,
        "DOWN": False,
        "A": False,
        "D": False,
        "W": False,
        "S": False,
        "Q": False,
        "E": False
    }
    
    key_mappings = {
        pygame.K_LEFT: "LEFT",
        pygame.K_RIGHT: "RIGHT",
        pygame.K_UP: "UP",
        pygame.K_DOWN: "DOWN",
        pygame.K_a: "A",
        pygame.K_d: "D",
        pygame.K_w: "W",
        pygame.K_s: "S",
        pygame.K_q: "Q",
        pygame.K_e: "E"
    }
    
    # Display instructions
    print("Interactive Demo Controls:")
    print("Arrow keys: Movement (LEFT, RIGHT, UP, DOWN)")
    print("WASD: Alternative movement")
    print("Q, E: Additional actions")
    print("Press P to pause/unpause")
    print("Press R to reset the frame")
    print("Press ESC to quit")
    
    running = True
    step = 0
    reanchor_every = 3
    
    DEFAULT_ACTION = 4  # Default action
    current_action = DEFAULT_ACTION
    action_text = f"Action: {coinrun_actions[current_action]} (default)"
    paused = False
    
    last_update_time = time.time()
    update_interval = 1.0/fps
    
    def get_current_action_id():
        active_keys = tuple(k for k, v in keys_pressed.items() if v)
        
        for i, action_tuple in enumerate(coinrun_actions):
            if set(active_keys) == set(action_tuple):
                return i
        
        return DEFAULT_ACTION
    
    try:
        while running and step < max_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_p:
                        paused = not paused
                        action_text = "PAUSED" if paused else f"Action: {coinrun_actions[current_action]}"
                    elif event.key == pygame.K_r:
                        sample_idx = random.randint(0, len(dataset)-1)
                        sample = dataset[sample_idx]
                        initial_frames = sample['history_frames'].to(device)
                        with torch.no_grad():
                            frame_history = []
                            for i in range(initial_frames.size(0)):
                                tokens = vqvae_model.encode(initial_frames[i:i+1])
                                frame_history.append(tokens)
                            
                            current_tokens = frame_history[-1]
                            current_frame = vqvae_model.decode(current_tokens)
                        step = 0
                    elif event.key in key_mappings:
                        keys_pressed[key_mappings[event.key]] = True
                        current_action = get_current_action_id()
                        action_text = f"Action: {coinrun_actions[current_action]}"
                
                elif event.type == pygame.KEYUP:
                    if event.key in key_mappings:
                        keys_pressed[key_mappings[event.key]] = False
                        current_action = get_current_action_id()
                        action_text = f"Action: {coinrun_actions[current_action]}"
            
            current_time = time.time()
            if not paused and (current_time - last_update_time) >= update_interval:
                last_update_time = current_time
                
                action_tensor = torch.tensor([current_action], device=device)
                with torch.no_grad():
                    history_tensor = torch.stack(frame_history, dim=0).transpose(0, 1)
                    
                    next_tokens = dynamics_model.generate(
                        current_tokens, 
                        action_tensor, 
                        vqvae_model, 
                        prev_frames_history=history_tensor
                    )
                    
                    next_frame = vqvae_model.decode(next_tokens)
                    
                    if (step+1) % reanchor_every == 0:
                        reencoded_tokens = vqvae_model.encode(next_frame)
                        token_diff = (next_tokens != reencoded_tokens).float().mean().item()
                        
                        if token_diff > 0.2:  # 20% threshold instead of 10%
                            next_tokens = reencoded_tokens
                            print(f"Reanchoring at step {step+1}, diff: {token_diff:.2f}")
                    
                    current_tokens = next_tokens
                    current_frame = next_frame
                    
                    frame_history.append(current_tokens)
                    if len(frame_history) > config.dynamics_context_frames:
                        frame_history.pop(0)
                    
                    step += 1
            
            current_img = current_frame.squeeze(0).cpu()
            current_img = current_img * 0.5 + 0.5
            current_img = current_img.permute(1, 2, 0).numpy()  # Shape: (H, W, C)
            current_img = (current_img * 255).astype(np.uint8)
            
            current_img = np.transpose(current_img, (1, 0, 2))  # Swap width and height
            
            current_img = pygame.surfarray.make_surface(current_img)
            current_img = pygame.transform.scale(current_img, render_size)
            screen.blit(current_img, (0, 0))
            
            text_surface = font.render(action_text, True, (255, 255, 255))
            text_rect = text_surface.get_rect(topleft=(10, 10))
            pygame.draw.rect(screen, (0, 0, 0), text_rect.inflate(10, 10))
            screen.blit(text_surface, text_rect)
            
            step_text = font.render(f"Step: {step}", True, (255, 255, 255))
            step_rect = step_text.get_rect(topleft=(10, 40))
            pygame.draw.rect(screen, (0, 0, 0), step_rect.inflate(10, 10))
            screen.blit(step_text, step_rect)
            
            active_keys = [k for k, v in keys_pressed.items() if v]
            keys_text = f"Keys: {', '.join(active_keys) if active_keys else 'None'}"
            keys_surface = font.render(keys_text, True, (255, 255, 255))
            keys_rect = keys_surface.get_rect(topleft=(10, 70))
            pygame.draw.rect(screen, (0, 0, 0), keys_rect.inflate(10, 10))
            screen.blit(keys_surface, keys_rect)
            
            pygame.display.update()
            
            frame_data = pygame.surfarray.array3d(screen)

            frame_data = frame_data.transpose([1, 0, 2])
            frame_data = cv2.cvtColor(frame_data, cv2.COLOR_RGB2BGR)
            
            # Write to video
            video_writer.write(frame_data)
            
            clock.tick(fps * 2)
    
    finally:
        # Make sure to release the video writer
        video_writer.release()
        print(f"Video saved to: {video_path}")
        pygame.quit()
        print(f"Demo ended after {step} steps")

if __name__ == "__main__":
    torch.manual_seed(15)
    np.random.seed(15)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(15)
    
    test_saved_models()