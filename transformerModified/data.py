import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class PDEDataset(Dataset):
    def __init__(self, D_array, u_array):
        # Directly convert to tensors with proper shapes
        self.D = torch.tensor(D_array, dtype=torch.float32).unsqueeze(1)  # [N, 1, H, W]
        self.u = torch.tensor(u_array, dtype=torch.float32).unsqueeze(1)  # [N, 1, H, W]
    
    def __len__(self):
        return self.D.shape[0]
    
    def __getitem__(self, idx):
        return self.D[idx], self.u[idx]

class OptimizedTransformerDataset:
    def __init__(self, random_seed=42):
        self.D_train = None
        self.u_train = None
        self.D_test = None
        self.u_test = None
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
    
    def load_data(self, train_path, test_path):
        # Use mmap_mode for potentially large files - prevents loading all data into memory
        train_data = np.load(train_path, mmap_mode='r')
        self.D_train = train_data['D']
        self.u_train = train_data['u']
        
        test_data = np.load(test_path, mmap_mode='r')
        self.D_test = test_data['D']
        self.u_test = test_data['u']
        
        # Print data shapes for debugging
        print(f"Training data shapes: D={self.D_train.shape}, u={self.u_train.shape}")
        print(f"Testing data shapes: D={self.D_test.shape}, u={self.u_test.shape}")
        
        return self
    
    def create_dataloaders(self, batch_size=4, num_workers=4):
        """Create DataLoaders with optimized settings"""
        # Create datasets
        train_dataset = PDEDataset(self.D_train, self.u_train)
        test_dataset = PDEDataset(self.D_test, self.u_test)
        
        # Create data loaders with performance optimizations
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,  # Parallel data loading
            pin_memory=True,         # Faster data transfer to GPU
            prefetch_factor=2,       # Pre-fetch batches
            persistent_workers=True  # Keep workers alive between epochs
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )
        
        return train_loader, test_loader
    
    def get_raw_data(self):
        return self.D_train, self.u_train, self.D_test, self.u_test
    
    def visualize_samples(self, num_samples=4):
        """Visualize random samples from the dataset"""
        if self.D_train is None or self.u_train is None:
            print("Data not loaded yet")
            return
        
        indices = np.random.choice(len(self.D_train), num_samples, replace=False)
        
        fig, axes = plt.subplots(num_samples, 2, figsize=(10, 3*num_samples))
        
        for i, idx in enumerate(indices):
            # Plot input
            im0 = axes[i, 0].imshow(self.D_train[idx], cmap='viridis')
            axes[i, 0].set_title(f"Input D[{idx}]")
            plt.colorbar(im0, ax=axes[i, 0])
            
            # Plot output
            im1 = axes[i, 1].imshow(self.u_train[idx], cmap='viridis')
            axes[i, 1].set_title(f"Output u[{idx}]")
            plt.colorbar(im1, ax=axes[i, 1])
        
        plt.tight_layout()
        plt.savefig('sample_visualization.png')
        plt.close()
        print("Sample visualization saved to 'sample_visualization.png'")