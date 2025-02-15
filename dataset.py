# dataset.py
import torch
import os
import random
import imghdr
import cv2
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image, ImageFile
from collections import Counter

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

class RobustImageFolder(ImageFolder):
    """Enhanced robustness dataset class (supporting format detection and error recovery)"""
    def __init__(self, root, transform=None, max_retry=3):
        # Call the parent class constructor to initialize the root directory and transform of the dataset
        super().__init__(root, transform=transform)
        # Count the sample size for each category
        self.class_dist = Counter([label for _, label in self.samples])
        # Set maximum retry count
        self.max_retry = max_retry
        # Initialize the collection of bad files
        self.bad_files = set()
        # Initialize the list of valid indexes
        self.valid_indices = [i for i in range(len(self.samples))]

    def __getitem__(self, index):
        # Save original index
        original_index = index
        # Obtain sample path and label
        path, target = self.samples[index]
        
        # If it is already a known bad file, return virtual data directly
        if path in self.bad_files:
            return self._get_dummy_item()
            
        # Attempt to load image, with a maximum of x_retry attempts
        for _ in range(self.max_retry):
            try:
                # Using enhanced loading methods to load images
                img = self._enhanced_loader(path)
                # If there is a transformation, apply the transformation
                if self.transform:
                    img = self.transform(img)
                # Return image and label
                return img, target
            except Exception as e:
                print(f"Load Fail ({_+1}/{self.max_retry}): {path} - {str(e)}")
                # Randomly select a new index
                index = random.choice(self.valid_indices)
                path, target = self.samples[index]
        
        # Record bad files
        self.bad_files.add(self.samples[original_index][0])
        return self._get_dummy_item()

    def _enhanced_loader(self, path):
        """Enhanced image loading method"""
        # Method 1: Attempt PIL native loading
        try:
            with open(path, 'rb') as f:
                img = Image.open(f)
                img.load()  # Force data loading
                return self._convert_image(img)
        except:
            pass
            
        # Method 2: Try loading OpenCV
        try:
            cv_img = cv2.imread(path)
            if cv_img is not None:
                return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        except:
            pass
            
        # Method 3: Real format detection through binary detection
        try:
            with open(path, 'rb') as f:
                data = f.read()
                detected_format = imghdr.what(None, h=data)
                if detected_format:
                    return Image.open(path).convert('RGB')
        except:
            pass
            
        raise RuntimeError(f"All loading methods failed: {path}")

    def _convert_image(self, img):
        """Unified image format processing"""
        if img.mode != 'RGB':
            img = img.convert('L')  # Convert to grayscale
            img = Image.merge('RGB', (img, img, img))  # Copy as 3 channels
        return img

    def _get_dummy_item(self):
        """Generate virtual data"""
        return torch.zeros(3, 224, 224), -1  # Using -1 as an invalid tag

    def analyze_distribution(self):
        """Analyze the statistical information of the dataset"""
        print("Category distribution:", self.class_dist)
        print(f"Find {len(self.bad_files)} broken files")
        
    def get_valid_loader(self, batch_size):
        """Retrieve the filtered valid data loader"""
        valid_indices = [i for i in range(len(self)) if self[i][1] != -1]
        return DataLoader(
            self,
            batch_size=batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(valid_indices)
        )

def get_transform(train=False):
    """Augmentation"""
    basic = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
    
    if train:
        basic = [
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ] + basic
        
    return transforms.Compose(basic)

def load_robust_data(data_dir, batch_size):
    """Robust data loading"""
    train_set = RobustImageFolder(
        root=os.path.join(data_dir, 'train'),
        transform=get_transform(train=True),
        max_retry=3
    )
    
    val_set = RobustImageFolder(
        root=os.path.join(data_dir, 'val'),
        transform=get_transform(),
        max_retry=2
    )
    
    test_set = RobustImageFolder(
        root=os.path.join(data_dir, 'test'),
        transform=get_transform(),
        max_retry=2
    )

    # 数据集分析
    print("\n=== Training set statistics ===")
    train_set.analyze_distribution()
    print("\n=== Validation set statistics ===")
    val_set.analyze_distribution()
    
    return (
        train_set.get_valid_loader(batch_size),
        val_set.get_valid_loader(batch_size),
        test_set.get_valid_loader(batch_size)
    )