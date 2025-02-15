# main.py
import torch
import random
import numpy as np
import os
from model import create_model
from dataset import load_robust_data
from trainer import train_model


def set_seed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    np.random.default_rng(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

CONFIG = {
    # Data parameters
    'data_dir': './carbonate_1',
    'batch_size': 512,
    
    # Model parameter
    'model_name': 'resnet18',  # Options: resnet18/resnet50/vgg16/densenet121/mobilenet_v2/efficientnet_b0
    'num_classes': 22,         # Lithological number
    
    # Train parameter
    'epochs': 50,
    'learning_rate': 1e-5,
    'seed': 42,             # Seed
    
    # Log parameter
    'save_dir': './experiments',
    'experiment_name': 'exp1' # Experiment name
}

def setup_experiment():
    """Create experiment directory"""
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    print(f"Experiment directory created at: {os.path.abspath(CONFIG['save_dir'])}")

def print_data_stats(train_set, val_set, test_set):
    """Print data statistics"""
    print("\n=== Dataset statistic ===")
    print(f"Train samples: {len(train_set)} (Valid samples: {len(train_set.valid_indices)})")
    print(f"Validation samples: {len(val_set)} (Valid samples: {len(val_set.valid_indices)})")
    print(f"Test samples: {len(test_set)} (Valid samples: {len(test_set.valid_indices)})")
    
    print("\nClass distribution in train set:")
    train_set.analyze_distribution()
    
    print("\nBronken files statistics:")
    print(f"Train: {len(train_set.bad_files)}")
    print(f"Val: {len(val_set.bad_files)}")
    print(f"Test: {len(test_set.bad_files)}")

# main function
def main():
    # Initialize seed and experiment
    set_seed(CONFIG['seed'])
    setup_experiment()
    
    try:
        # Load data
        print("\nDataset is loading...")
        train_loader, val_loader, test_loader = load_robust_data(
            CONFIG['data_dir'], 
            CONFIG['batch_size']
        )
        
        # Display dataset information
        print_data_stats(train_loader.dataset.dataset, val_loader.dataset.dataset, test_loader.dataset.dataset)
        
        # Initial model
        print(f"\nThe {CONFIG['model_name']} model is initializing...")
        model = create_model(
            model_name=CONFIG['model_name'],
            num_classes=CONFIG['num_classes']
        )
        
        # Training and Validation
        print("\nStarted training...")
        train_model(
            model=model,
            loaders=(train_loader, val_loader, test_loader),
            num_classes=CONFIG['num_classes'],
            epochs=CONFIG['epochs'],
            lr=CONFIG['learning_rate'],
            early_stopping_patience=5  # Set early stop patience value
        )
        
    except Exception as e:
        print(f"Run error: {str(e)}")
        raise

if __name__ == "__main__":
    main()