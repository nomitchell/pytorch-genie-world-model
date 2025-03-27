# Configuration
class Config:
    # General
    seed = 42
    image_size = 64
    batch_size = 16
    gradient_accumulation_steps = 4
    num_workers = 4
    
    # Video Tokenizer
    vq_embedding_dim = 64 
    vq_num_embeddings = 512 
    vq_commitment_cost = 0.25
    
    # Spatiotemporal Transformer, Reduced from original paper
    hidden_dim = 256  # 512
    num_heads = 4  # 8
    num_layers = 4  # 12? 
    
    # Latent Action Model
    num_actions = 14 
    history_length = 16 
    
    # Dynamics Model
    dynamics_context_frames = 16 # TODO make naming more consiste
    
    # Training
    epochs = 1
    lr = 3e-4
    lr_min = 1e-5
    weight_decay = 1e-5
    warmup_epochs = 5