import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from main import (
    Config, 
    VQVAEModel, 
    LatentActionModel, 
    MaskGIT, 
    CoinRunDataset
)

config = Config()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create output directory for saved plots and images
output_dir = "eval_results"
os.makedirs(output_dir, exist_ok=True)

# Helper functions for testing
def show_image_comparison(original, reconstructed, title="Original vs Reconstructed", filename=None):
    """Save original and reconstructed images side by side to a file"""
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed)
    plt.title("Reconstructed")
    plt.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if filename is None:
        filename = title.replace(" ", "_").replace(":", "").lower() + ".png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close() 

def tensor_to_image(tensor):
    """Convert tensor to numpy image, handling denormalization"""
    img = tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
    img = img * 0.5 + 0.5 
    img = np.clip(img, 0, 1)
    return img

def calculate_metrics(original, reconstructed):
    """Calculate PSNR and SSIM metrics between images"""
    # Convert to 0-1 range if needed
    if original.max() > 1:
        original = original / 255.0
    if reconstructed.max() > 1:
        reconstructed = reconstructed / 255.0
        
    psnr_value = psnr(original, reconstructed)
    
    min_dim = min(original.shape[0], original.shape[1])
    if min_dim < 7:
        win_size = min_dim - (1 if min_dim % 2 == 0 else 0)  # Ensure odd window size
    else:
        win_size = 7
    
    ssim_value = ssim(
        original, 
        reconstructed, 
        win_size=win_size,
        channel_axis=2,
        data_range=1.0
    )
    
    return psnr_value, ssim_value

def test_vqvae_reconstruction(vqvae_model, dataset, num_samples=5):
    print("\nTesting VQ-VAE Reconstruction Quality")
    vqvae_model.eval()
    
    indices = random.sample(range(len(dataset)), num_samples)
    
    metrics = []
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        frame = sample['history_frames'][0].unsqueeze(0).to(device)
        
        with torch.no_grad():
            tokens = vqvae_model.encode(frame)
            reconstructed = vqvae_model.decode(tokens)
        
        original_img = tensor_to_image(frame)
        reconstructed_img = tensor_to_image(reconstructed)
        
        psnr_val, ssim_val = calculate_metrics(original_img, reconstructed_img)
        metrics.append((psnr_val, ssim_val))
        
        print(f"Sample {idx}: PSNR = {psnr_val:.2f}dB, SSIM = {ssim_val:.4f}")
        
        filename = f"vqvae_recon_sample_{i+1}.png"
        show_image_comparison(original_img, reconstructed_img, 
                             f"VQ-VAE Reconstruction (PSNR: {psnr_val:.2f}dB, SSIM: {ssim_val:.4f})",
                             filename=filename)
    
    avg_psnr = sum([m[0] for m in metrics]) / len(metrics)
    avg_ssim = sum([m[1] for m in metrics]) / len(metrics)
    print(f"Average: PSNR = {avg_psnr:.2f}dB, SSIM = {avg_ssim:.4f}")
    
    test_codebook_usage(vqvae_model, dataset)
    
    return avg_psnr, avg_ssim

def test_codebook_usage(vqvae_model, dataset, num_samples=100):
    print("\nTesting VQ-VAE Codebook Usage")
    vqvae_model.eval()
    
    code_usage = {}
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    for idx in tqdm(indices, desc="Checking codebook usage"):
        sample = dataset[idx]
        frames = torch.cat([
            sample['history_frames'].reshape(-1, 3, config.image_size, config.image_size),
            sample['next_frame'].unsqueeze(0)
        ], dim=0).to(device)
        
        with torch.no_grad():
            for frame in frames:
                tokens = vqvae_model.encode(frame.unsqueeze(0))
                for token in tokens.flatten().cpu().numpy():
                    if token in code_usage:
                        code_usage[token] += 1
                    else:
                        code_usage[token] = 1
    
    total_codes = vqvae_model.vq.num_embeddings
    used_codes = len(code_usage)
    usage_percentage = (used_codes / total_codes) * 100
    
    print(f"Codebook usage: {used_codes} out of {total_codes} codes used ({usage_percentage:.2f}%)")
    
    plt.figure(figsize=(10, 5))
    plt.hist(list(code_usage.values()), bins=30, log=True)
    plt.title("Codebook Usage Distribution (Log Scale)")
    plt.xlabel("Frequency")
    plt.ylabel("Number of Codes")
    plt.savefig(os.path.join(output_dir, "codebook_usage_histogram.png"))
    plt.close()
    
    return usage_percentage

def test_single_step_prediction(vqvae_model, dynamics_model, dataset, num_samples=5):
    print("\nTesting Single-Step Dynamics Prediction")
    vqvae_model.eval()
    dynamics_model.eval()
    
    indices = random.sample(range(len(dataset)), num_samples)
    
    metrics = []
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        history_frames = sample['history_frames'].to(device)  # Get full history
        frame = history_frames[-1].unsqueeze(0).to(device)  # Last frame in history
        next_frame = sample['next_frame'].unsqueeze(0).to(device)
        action = torch.tensor([sample['action']], device=device)
        
        with torch.no_grad():
            history_tokens = []
            for i in range(history_frames.size(0)):
                tokens = vqvae_model.encode(history_frames[i:i+1])
                history_tokens.append(tokens)
            
            history_tokens = torch.stack(history_tokens, dim=1)
            
            frame_tokens = vqvae_model.encode(frame)
            next_frame_tokens_gt = vqvae_model.encode(next_frame)
            
            predicted_tokens = dynamics_model.generate(
                frame_tokens, 
                action, 
                vqvae_model, 
                prev_frames_history=history_tokens
            )
            predicted_frame = vqvae_model.decode(predicted_tokens)
        
        original_img = tensor_to_image(next_frame)
        predicted_img = tensor_to_image(predicted_frame)
        
        psnr_val, ssim_val = calculate_metrics(original_img, predicted_img)
        metrics.append((psnr_val, ssim_val))
        
        print(f"Sample {idx} (Action {sample['action']}): PSNR = {psnr_val:.2f}dB, SSIM = {ssim_val:.4f}")
        
        filename = f"single_step_pred_sample_{i+1}.png"
        show_image_comparison(original_img, predicted_img, 
                             f"Single Step Prediction (Action {sample['action']}, PSNR: {psnr_val:.2f}dB)",
                             filename=filename)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.hist(next_frame_tokens_gt.cpu().numpy().flatten(), bins=30, alpha=0.5, label='Ground Truth')
        plt.title("Ground Truth Token Distribution")
        plt.subplot(1, 2, 2)
        plt.hist(predicted_tokens.cpu().numpy().flatten(), bins=30, alpha=0.5, label='Predicted')
        plt.title("Predicted Token Distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"token_dist_sample_{i+1}.png"))
        plt.close()
    
    avg_psnr = sum([m[0] for m in metrics]) / len(metrics)
    avg_ssim = sum([m[1] for m in metrics]) / len(metrics)
    print(f"Average: PSNR = {avg_psnr:.2f}dB, SSIM = {avg_ssim:.4f}")
    
    return avg_psnr, avg_ssim

def test_multi_step_prediction(vqvae_model, dynamics_model, dataset, num_steps=5):
    print("\nTesting Multi-Step Prediction (Quality Degradation)")
    vqvae_model.eval()
    dynamics_model.eval()
    
    idx = random.randint(0, len(dataset)-1)
    sample = dataset[idx]
    
    history_frames = sample['history_frames'].to(device)
    
    frame_history = []
    with torch.no_grad():
        for i in range(history_frames.size(0)):
            tokens = vqvae_model.encode(history_frames[i:i+1])
            frame_history.append(tokens)
    
    current_frame = history_frames[-1:].to(device)
    current_tokens = frame_history[-1]
    
    actions = torch.tensor([random.randint(0, 6) for _ in range(num_steps)], device=device)
    
    frames = [tensor_to_image(current_frame)]
    metrics = []
    
    for step in range(num_steps):
        action = actions[step].unsqueeze(0)
        
        with torch.no_grad():
            # Create history tensor in format (batch, seq_len, num_tokens)
            history_tensor = torch.stack(frame_history[-config.dynamics_context_frames:], dim=1)
            
            next_tokens = dynamics_model.generate(
                current_tokens, 
                action, 
                vqvae_model,
                prev_frames_history=history_tensor
            )
            next_frame = vqvae_model.decode(next_tokens)
            
            frames.append(tensor_to_image(next_frame))
            
            reencoded_tokens = vqvae_model.encode(next_frame)
            
            token_diff = (next_tokens != reencoded_tokens).float().mean().item() * 100
            metrics.append(token_diff)
            
            print(f"Step {step+1} (Action {action.item()}): Token difference: {token_diff:.2f}%")
            
            frame_history.append(next_tokens.clone())
            if len(frame_history) > config.dynamics_context_frames:
                frame_history.pop(0)
            
            current_tokens = next_tokens.clone()
    
    plt.figure(figsize=(15, 5))
    for i, frame in enumerate(frames):
        plt.subplot(1, len(frames), i+1)
        plt.imshow(frame)
        if i == 0:
            plt.title("Initial")
        else:
            plt.title(f"Step {i}\nAction: {actions[i-1].item()}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"multi_step_prediction_frames.png"))
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_steps+1), metrics)
    plt.title("Token Difference Over Multiple Steps")
    plt.xlabel("Step")
    plt.ylabel("Token Difference (%)")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "token_difference_plot.png"))
    plt.close()
    
    return metrics

def test_latent_action_model(vqvae_model, lam_model, dataset, num_samples=5):
    print("\nTesting Latent Action Model Prediction")
    vqvae_model.eval()
    lam_model.eval()
    
    indices = random.sample(range(len(dataset)), num_samples)
    print(indices)
    metrics = []
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        history_frames = sample['history_frames'].unsqueeze(0).to(device)
        next_frame = sample['next_frame'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            history_tokens = []
            for x in range(history_frames.size(1)):
                tokens = vqvae_model.encode(history_frames[:, x])
                history_tokens.append(tokens)
            history_tokens = torch.stack(history_tokens, dim=1)
            
            next_tokens = vqvae_model.encode(next_frame)
            
            latent_actions = lam_model.encode_actions(history_tokens, next_tokens)
            predicted_tokens = lam_model.predict_next_frame(history_tokens, latent_actions)
            predicted_frame = vqvae_model.decode(predicted_tokens)
        
        original_img = tensor_to_image(next_frame)
        predicted_img = tensor_to_image(predicted_frame)
        
        psnr_val, ssim_val = calculate_metrics(original_img, predicted_img)
        metrics.append((psnr_val, ssim_val))
        
        print(f"Sample {idx}: PSNR = {psnr_val:.2f}dB, SSIM = {ssim_val:.4f}")
        
        print(i+1)
        filename = f"lam_prediction_sample_{i+1}.png"
        show_image_comparison(original_img, predicted_img, 
                             f"LAM Prediction (PSNR: {psnr_val:.2f}dB, SSIM: {ssim_val:.4f})",
                             filename=filename)
    
    avg_psnr = sum([m[0] for m in metrics]) / len(metrics)
    avg_ssim = sum([m[1] for m in metrics]) / len(metrics)
    print(f"Average: PSNR = {avg_psnr:.2f}dB, SSIM = {avg_ssim:.4f}")
    
    return avg_psnr, avg_ssim

def main():
    config = Config()
    
    data_dir = "coinrun_data"
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} not found. Please run data collection first.")
    
    vqvae_model = VQVAEModel(config)
    lam_model = LatentActionModel(config)
    dynamics_model = MaskGIT(config)
    
    model_files = {
        "vqvae_model.pt": vqvae_model,
        "lam_model.pt": lam_model,
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
    
    dataset = CoinRunDataset(data_dir, image_size=config.image_size, history_length=config.history_length)
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving evaluation results to: {output_dir}")
    
    print("\n===== STARTING DIAGNOSTIC TESTS =====")
    
    vqvae_metrics = test_vqvae_reconstruction(vqvae_model, dataset)
    
    lam_metrics = test_latent_action_model(vqvae_model, lam_model, dataset)
    
    dynamics_single_metrics = test_single_step_prediction(vqvae_model, dynamics_model, dataset)
    
    dynamics_multi_metrics = test_multi_step_prediction(vqvae_model, dynamics_model, dataset, num_steps=10)
    
    with open(os.path.join(output_dir, "evaluation_summary.txt"), "w") as f:
        f.write("===== DIAGNOSTIC SUMMARY =====\n")
        f.write(f"VQ-VAE Reconstruction: PSNR = {vqvae_metrics[0]:.2f}dB, SSIM = {vqvae_metrics[1]:.4f}\n")
        f.write(f"Latent Action Model: PSNR = {lam_metrics[0]:.2f}dB, SSIM = {lam_metrics[1]:.4f}\n")
        f.write(f"Dynamics Model (Single Step): PSNR = {dynamics_single_metrics[0]:.2f}dB, SSIM = {dynamics_single_metrics[1]:.4f}\n")
        f.write(f"Dynamics Model (Multi Step): Error accumulation trend - {'Increasing' if dynamics_multi_metrics[-1] > dynamics_multi_metrics[0] else 'Stable'}\n")
    
    # Print summary
    print("\n===== DIAGNOSTIC SUMMARY =====")
    print(f"VQ-VAE Reconstruction: PSNR = {vqvae_metrics[0]:.2f}dB, SSIM = {vqvae_metrics[1]:.4f}")
    print(f"Latent Action Model: PSNR = {lam_metrics[0]:.2f}dB, SSIM = {lam_metrics[1]:.4f}")
    print(f"Dynamics Model (Single Step): PSNR = {dynamics_single_metrics[0]:.2f}dB, SSIM = {dynamics_single_metrics[1]:.4f}")
    print(f"Dynamics Model (Multi Step): Error accumulation trend - {'Increasing' if dynamics_multi_metrics[-1] > dynamics_multi_metrics[0] else 'Stable'}")
    print(f"\nEvaluation results saved to: {output_dir}")

if __name__ == "__main__":
    torch.manual_seed(15)
    np.random.seed(15)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(15)
    
    main()