import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import os
import numpy as np
import time
import math
from torch.cuda.amp import GradScaler, autocast
from transformer2D import build_optimized_transformer2d
from data import OptimizedTransformerDataset, PDEDataset

def train_epoch(model, loader, criterion, optimizer, device, scaler=None, use_amp=False):
    model.train()
    total_loss = 0
    start_time = time.time()
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        # Clear gradients
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        
        if use_amp:
            # Mixed precision training
            with autocast():
                output = model(x)
                loss = criterion(output, y)
            
            # Scale gradients and optimize
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Regular training
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
    
    epoch_time = time.time() - start_time
    return total_loss / len(loader), epoch_time

def validate(model, loader, criterion, device, use_amp=False):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            
            if use_amp:
                with autocast():
                    output = model(x)
                    loss = criterion(output, y)
            else:
                output = model(x)
                loss = criterion(output, y)
            
            total_loss += loss.item()
    
    return total_loss / len(loader)

def compute_metrics(model, test_loader, device, use_amp=False):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            
            if use_amp:
                with autocast():
                    preds = model(x)
            else:
                preds = model(x)
            
            all_preds.append(preds.reshape(-1).cpu())
            all_targets.append(y.reshape(-1).cpu())
    
    # Concatenate all predictions and targets
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    # Compute MSE
    mse = nn.MSELoss()(all_preds, all_targets).item()
    
    # Compute R²
    ss_res = torch.sum((all_targets - all_preds) ** 2)
    ss_tot = torch.sum((all_targets - torch.mean(all_targets)) ** 2)
    r2 = (1.0 - ss_res / ss_tot).item()
    
    return {"mse": mse, "r2": r2}

def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def load_checkpoint(model, optimizer, path):
    """Load a checkpoint to resume training"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

def load_checkpoint_with_compatibility(model, optimizer, path):
    """Load a checkpoint with compatibility for PE layer name and shape changes"""
    checkpoint = torch.load(path)
    state_dict = checkpoint['model_state_dict']
    model_dict = model.state_dict()
    
    # Check for key renames
    if 'pos_enc.pos_embed' in state_dict and 'pos_enc.pe' in model_dict:
        state_dict['pos_enc.pe'] = state_dict.pop('pos_enc.pos_embed')
    elif 'pos_enc.pe' in state_dict and 'pos_enc.pos_embed' in model_dict:
        state_dict['pos_enc.pos_embed'] = state_dict.pop('pos_enc.pe')
    
    # Handle shape mismatches by only loading matching parameters
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                compatible_state_dict[k] = v
            else:
                print(f"Skipping parameter '{k}' due to shape mismatch: checkpoint={v.shape}, model={model_dict[k].shape}")
        else:
            print(f"Skipping unexpected key in checkpoint: {k}")
    
    # Add missing parameters from the model
    missing_keys = [k for k in model_dict.keys() if k not in compatible_state_dict]
    print(f"Parameters in model but not in checkpoint: {missing_keys}")
    
    # Load the compatible state dict
    model.load_state_dict(compatible_state_dict, strict=False)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    return model, optimizer, epoch, loss

def main():
    # Configuration
    train_path = "/home/ys460/Desktop/Inverse_Problem/transformer2.0/Transformer2.0Test100/transformer2.0_test100.npz"
    test_path = "/home/ys460/Desktop/Inverse_Problem/transformer2.0/Transformer2.0Train1000/transformer2.0_train1000.npz"
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training parameters
    batch_size = 4 
    learning_rate = 1e-3
    weight_decay = 1e-5 
    epochs = 10000
    save_interval = 10  # Save every 10 epochs
    use_amp = True  # Use mixed precision training

    # Model parameters
    in_channel = 1
    out_channel = 1
    d_model = 128
    N = 2  # Number of encoder/decoder blocks
    dropout = 0.1
    h = 4  # Number of attention heads
    d_ff = 256  # Feed-forward dimension
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and prepare data
    print("Loading data...")
    dataset = OptimizedTransformerDataset()
    dataset.load_data(train_path, test_path)
    D_train, u_train, D_test, u_test = dataset.get_raw_data()
    
    # Visualize some samples
    dataset.visualize_samples(num_samples=4)
    
    # Get input shape from data
    HEIGHT, WIDTH = D_train.shape[1], D_train.shape[2]
    PATCH_SIZE = (1, 1)
    input_shape = (HEIGHT, WIDTH)
    
    # Create data loaders with optimized settings
    train_dataset = PDEDataset(D_train, u_train)
    test_dataset = PDEDataset(D_test, u_test)
    
    # Adjust number of workers based on CPU cores (0 for Windows if experiencing issues)
    num_workers = min(4, os.cpu_count() or 1)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Check for existing checkpoints
    resume_training = False
    latest_checkpoint = None
    
    if os.path.exists(checkpoint_dir):
        epoch_checkpoints = [f for f in os.listdir(checkpoint_dir) 
                            if f.endswith('.pth') and f.startswith('model_epoch_')]
        if epoch_checkpoints:
            latest_checkpoint = os.path.join(
                checkpoint_dir, 
                sorted(epoch_checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
            )
            resume_training = True
    
    # If resuming, extract model dimensions from checkpoint
    if resume_training and latest_checkpoint:
        print(f"Examining checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint)
        if 'model_state_dict' in checkpoint:
            # Try to determine dimensions from the checkpoint
            if 'pos_enc.pos_embed' in checkpoint['model_state_dict']:
                pos_embed_shape = checkpoint['model_state_dict']['pos_enc.pos_embed'].shape
                if len(pos_embed_shape) == 3:  # [1, H*W, d_model]
                    expected_d_model = pos_embed_shape[2]
                    expected_hw = pos_embed_shape[1]
                    # Assuming square input and 1:1 patch size for simplicity
                    expected_h = expected_w = int(math.sqrt(expected_hw))
                    print(f"Detected dimensions from checkpoint: H={expected_h}, W={expected_w}, d_model={expected_d_model}")
                    # Override model parameters with checkpoint values
                    d_model = expected_d_model
                    # Set input_shape based on patch size and expected output size
                    input_shape = (expected_h * PATCH_SIZE[0], expected_w * PATCH_SIZE[1])
                    print(f"Setting input_shape to {input_shape}")
            elif 'pos_enc.pe' in checkpoint['model_state_dict']:
                pos_embed_shape = checkpoint['model_state_dict']['pos_enc.pe'].shape
                expected_d_model = pos_embed_shape[1]
                expected_hw = pos_embed_shape[0]
                # Assuming square input for simplicity
                expected_h = expected_w = int(math.sqrt(expected_hw))
                print(f"Detected dimensions from checkpoint: H={expected_h}, W={expected_w}, d_model={expected_d_model}")
                # Override model parameters
                d_model = expected_d_model
                input_shape = (expected_h * PATCH_SIZE[0], expected_w * PATCH_SIZE[1])
                print(f"Setting input_shape to {input_shape}")
    
    # Build the optimized model
    print("Building model...")
    model = build_optimized_transformer2d(
        in_channels=in_channel,
        out_channels=out_channel,
        input_shape=input_shape,
        patch_size=PATCH_SIZE,
        d_model=d_model,
        N=N,
        dropout=dropout,
        h=h,
        d_ff=d_ff
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Set up loss and optimizer
    criterion = nn.MSELoss()
    
    # Use AdamW instead of Adam for better weight decay handling
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler (removed verbose parameter)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=20
    )
    
    # Set up mixed precision training (updated to new format)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    # Resume training if checkpoint exists
    start_epoch = 0
    if resume_training and latest_checkpoint:
        print(f"Resuming from checkpoint: {latest_checkpoint}")
        try:
            # First try to load checkpoint directly
            model, optimizer, start_epoch, _ = load_checkpoint(model, optimizer, latest_checkpoint)
            print("Successfully loaded checkpoint with direct loading")
        except RuntimeError as e:
            print(f"Direct loading failed: {e}")
            print("Trying compatibility loading...")
            # If direct loading fails, try compatibility loading
            model, optimizer, start_epoch, _ = load_checkpoint_with_compatibility(model, optimizer, latest_checkpoint)
            print("Successfully loaded checkpoint with compatibility loading")
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print("Starting training...")
    for epoch in range(start_epoch, epochs):
        # Train for one epoch
        train_loss, epoch_time = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler, use_amp
        )
        
        # Validate
        val_loss = validate(model, test_loader, criterion, device, use_amp)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        # Print current learning rate
        print(f"Current learning rate: {scheduler.optimizer.param_groups[0]['lr']}")
        
        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs} | Time: {epoch_time:.2f}s | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # Save checkpoint based on interval
        if (epoch + 1) % save_interval == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
            )
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(checkpoint_dir, "best_model.pth")
            )
            print(f"Saved new best model with validation loss: {val_loss:.6f}")
    
    print("Training complete!")
    
    # Calculate final metrics
    print("Computing final metrics...")
    metrics = compute_metrics(model, test_loader, device, use_amp)
    mse = metrics["mse"]
    r2 = metrics["r2"]
    
    print(f"Final Test MSE: {mse:.6f}")
    print(f"Final Test R²: {r2:.6f}")
    
    # Plot training and validation loss and add R² and MSE
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    
    # Plot some example predictions
    plt.subplot(1, 2, 2)
    model.eval()
    with torch.no_grad():
        sample_inputs, sample_targets = next(iter(test_loader))
        sample_inputs = sample_inputs.to(device)
        sample_outputs = model(sample_inputs).cpu()
    
    # Plot a random example
    idx = np.random.randint(0, sample_inputs.size(0))
    
    plt.subplot(1, 2, 2)
    plt.imshow(sample_outputs[idx, 0].numpy(), cmap='viridis')
    plt.colorbar()
    plt.title('Example Prediction')
    
    # Add metrics text
    plt.figtext(0.5, 0.01, f"Test MSE: {mse:.6f} | Test R²: {r2:.6f}", 
                ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.close()
    
    print("Results saved to 'training_results.png'")
    
    # Save the final model
    torch.save(model.state_dict(), "optimized_transformer2d_trained.pth")
    print("Final model saved to 'optimized_transformer2d_trained.pth'")

if __name__ == "__main__":
    main()