import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from transformer2D import build_optimized_transformer2d
from data import OptimizedTransformerDataset, PDEDataset
from torch.utils.data import DataLoader

def load_model(model_path, model_config):
    """Load a trained model from checkpoint"""
    # Build model with the same configuration
    model = build_optimized_transformer2d(
        in_channels=model_config['in_channels'],
        out_channels=model_config['out_channels'],
        input_shape=model_config['input_shape'],
        patch_size=model_config['patch_size'],
        d_model=model_config['d_model'],
        N=model_config['N'],
        dropout=model_config['dropout'],
        h=model_config['h'],
        d_ff=model_config['d_ff']
    )
    
    # Load state dict
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def compute_metrics(model, dataloader):
    """Compute MSE and R² metrics"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
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

def visualize_predictions(model, dataloader, num_samples=5, save_path='prediction_results.png'):
    """Visualize model predictions vs ground truth"""
    model.eval()
    
    # Get a batch of data
    inputs, targets = next(iter(dataloader))
    
    # Generate predictions
    with torch.no_grad():
        inputs = inputs.to(device)
        predictions = model(inputs).cpu()
    
    # Determine number of samples to visualize (min of batch size and num_samples)
    n = min(inputs.size(0), num_samples)
    
    # Create a figure
    fig, axes = plt.subplots(n, 3, figsize=(15, 4*n))
    
    # If there's only one sample, make axes indexable
    if n == 1:
        axes = axes.reshape(1, -1)
    
    # Plot each sample
    for i in range(n):
        # Plot input
        im0 = axes[i, 0].imshow(inputs[i, 0].cpu().numpy(), cmap='viridis')
        axes[i, 0].set_title(f"Input D[{i}]")
        plt.colorbar(im0, ax=axes[i, 0])
        
        # Plot ground truth
        im1 = axes[i, 1].imshow(targets[i, 0].numpy(), cmap='viridis')
        axes[i, 1].set_title(f"Ground Truth u[{i}]")
        plt.colorbar(im1, ax=axes[i, 1])
        
        # Plot prediction
        im2 = axes[i, 2].imshow(predictions[i, 0].numpy(), cmap='viridis')
        axes[i, 2].set_title(f"Prediction")
        plt.colorbar(im2, ax=axes[i, 2])
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Visualization saved to {save_path}")

def compute_error_map(model, dataloader, num_samples=5, save_path='error_analysis.png'):
    """Compute and visualize error maps between predictions and ground truth"""
    model.eval()
    
    # Get a batch of data
    inputs, targets = next(iter(dataloader))
    
    # Generate predictions
    with torch.no_grad():
        inputs = inputs.to(device)
        predictions = model(inputs).cpu()
    
    # Determine number of samples to visualize
    n = min(inputs.size(0), num_samples)
    
    # Create a figure
    fig, axes = plt.subplots(n, 4, figsize=(20, 4*n))
    
    # If there's only one sample, make axes indexable
    if n == 1:
        axes = axes.reshape(1, -1)
    
    # Plot each sample
    for i in range(n):
        # Plot input
        im0 = axes[i, 0].imshow(inputs[i, 0].cpu().numpy(), cmap='viridis')
        axes[i, 0].set_title(f"Input D[{i}]")
        plt.colorbar(im0, ax=axes[i, 0])
        
        # Plot ground truth
        im1 = axes[i, 1].imshow(targets[i, 0].numpy(), cmap='viridis')
        axes[i, 1].set_title(f"Ground Truth u[{i}]")
        plt.colorbar(im1, ax=axes[i, 1])
        
        # Plot prediction
        im2 = axes[i, 2].imshow(predictions[i, 0].numpy(), cmap='viridis')
        axes[i, 2].set_title(f"Prediction")
        plt.colorbar(im2, ax=axes[i, 2])
        
        # Plot error map (absolute difference)
        error_map = np.abs(targets[i, 0].numpy() - predictions[i, 0].numpy())
        im3 = axes[i, 3].imshow(error_map, cmap='hot')
        axes[i, 3].set_title(f"Error Map")
        plt.colorbar(im3, ax=axes[i, 3])
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Error analysis saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Test Transformer 2D model for PDE problems')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth',
                      help='Path to the trained model')
    parser.add_argument('--test_data', type=str, 
                      default="/home/ys460/Desktop/Inverse_Problem/transformer2.0/Transformer2.0Test100/transformer2.0_test100.npz",
                      help='Path to test data')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for testing')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize')
    args = parser.parse_args()
    
    # Define model configuration (must match the trained model)
    model_config = {
        'in_channels': 1,
        'out_channels': 1,
        'input_shape': (64, 64),  # Adjust based on your data
        'patch_size': (1, 1),
        'd_model': 128,
        'N': 2,
        'dropout': 0.1,
        'h': 4,
        'd_ff': 256
    }
    
    # Load test data
    print(f"Loading test data from {args.test_data}")
    dataset = OptimizedTransformerDataset()
    try:
        # For test-only, we load the test data as train data to reuse the load_data method
        dataset.load_data(args.test_data, args.test_data)
        D_test, u_test = dataset.D_test, dataset.u_test
    except FileNotFoundError:
        print(f"Error: Could not find test data at {args.test_data}")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Create test dataset and dataloader
    test_dataset = PDEDataset(D_test, u_test)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=True
    )
    
    # Load the model
    print(f"Loading model from {args.model_path}")
    try:
        model = load_model(args.model_path, model_config)
        print("Model loaded successfully")
    except FileNotFoundError:
        print(f"Error: Could not find model at {args.model_path}")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Compute metrics
    print("Computing metrics...")
    metrics = compute_metrics(model, test_loader)
    print(f"Test MSE: {metrics['mse']:.6f}")
    print(f"Test R²: {metrics['r2']:.6f}")
    
    # Visualize predictions
    print(f"Visualizing {args.num_samples} sample predictions...")
    visualize_predictions(model, test_loader, num_samples=args.num_samples)
    
    # Compute error maps
    print("Computing error maps...")
    compute_error_map(model, test_loader, num_samples=args.num_samples)
    
    # Check specific failure cases where error is high
    print("Analyzing high-error cases...")
    
    # Create a summary figure
    plt.figure(figsize=(10, 6))
    
    # Add test metrics
    plt.text(0.5, 0.8, f"Test Results Summary", 
             ha='center', va='center', fontsize=16, fontweight='bold')
    plt.text(0.5, 0.6, f"Mean Squared Error (MSE): {metrics['mse']:.6f}", 
             ha='center', va='center', fontsize=14)
    plt.text(0.5, 0.4, f"R² Score: {metrics['r2']:.6f}", 
             ha='center', va='center', fontsize=14)
    
    # Add note about visualizations
    plt.text(0.5, 0.2, "See prediction_results.png and error_analysis.png for visualizations", 
             ha='center', va='center', fontsize=12, style='italic')
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('test_summary.png')
    plt.close()
    print("Test summary saved to test_summary.png")

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    main()