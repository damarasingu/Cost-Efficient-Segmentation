import os
import json
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Mean and std for RGB normalization
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

def load_and_preprocess_image(image_path):
    """
    Loads an image from file, converts to RGB, and resizes to 256x256 with cv2.INTER_AREA.
    Returns:
        np_rgb: uint8 array of shape (256, 256, 3)
        tensor: torch.FloatTensor of shape (3, 256, 256) normalized
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized_rgb = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_AREA)
    
    # Normalized tensor (3, 256, 256)
    norm = ((resized_rgb.astype(np.float32) / 255.0) - MEAN) / STD
    tensor = torch.from_numpy(norm).permute(2, 0, 1).float()
    return resized_rgb, tensor

def load_and_preprocess_mask(mask_path):
    """
    Loads a mask from file, resizes to 256x256 with cv2.INTER_NEAREST.
    A mask pixel is object if its value is >= 128.
    Returns:
        bin_mask: uint8 array of shape (256, 256) with values in {0, 1}
        tensor: torch.FloatTensor of shape (1, 256, 256) with values in {0.0, 1.0}
    """
    mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_gray is None:
        raise FileNotFoundError(f"Could not load mask: {mask_path}")
    resized_mask = cv2.resize(mask_gray, (256, 256), interpolation=cv2.INTER_NEAREST)
    bin_mask = (resized_mask >= 128).astype(np.uint8)
    tensor = torch.from_numpy(bin_mask).unsqueeze(0).float()
    return bin_mask, tensor

class DUTSDataset(Dataset):
    def __init__(self, image_paths, mask_paths, augment=False):
        assert len(image_paths) == len(mask_paths)
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        _, img_tensor = load_and_preprocess_image(img_path)
        _, mask_tensor = load_and_preprocess_mask(mask_path)
        
        # Simple data augmentations (random horizontal flip)
        if self.augment and random.random() > 0.5:
            img_tensor = torch.flip(img_tensor, dims=[2])
            mask_tensor = torch.flip(mask_tensor, dims=[2])
            
        return img_tensor, mask_tensor, img_path

def get_duts_tr_splits(data_root="data/DUTS-TR", val_ratio=0.1, seed=42):
    """
    Splits DUTS-TR (10,553 images) into train and validation sets deterministically.
    Never uses DUTS-TE.
    """
    img_dir = os.path.join(data_root, "DUTS-TR-Image")
    mask_dir = os.path.join(data_root, "DUTS-TR-Mask")
    
    img_names = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))])
    
    # Pair images and masks by basename
    paired = []
    for f in img_names:
        base = os.path.splitext(f)[0]
        mask_name = base + ".png"
        mask_path = os.path.join(mask_dir, mask_name)
        if os.path.exists(mask_path):
            paired.append((os.path.join(img_dir, f), mask_path))
            
    # Shuffle deterministically
    random.seed(seed)
    random.shuffle(paired)
    
    val_count = int(len(paired) * val_ratio)
    val_pairs = paired[:val_count]
    train_pairs = paired[val_count:]
    
    train_imgs, train_masks = zip(*train_pairs)
    val_imgs, val_masks = zip(*val_pairs)
    
    return list(train_imgs), list(train_masks), list(val_imgs), list(val_masks)

def get_duts_te_data(data_root="data/DUTS-TE"):
    img_dir = os.path.join(data_root, "DUTS-TE-Image")
    mask_dir = os.path.join(data_root, "DUTS-TE-Mask")
    
    img_names = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))])
    paired_imgs, paired_masks = [], []
    for f in img_names:
        base = os.path.splitext(f)[0]
        mask_name = base + ".png"
        mask_path = os.path.join(mask_dir, mask_name)
        if os.path.exists(mask_path):
            paired_imgs.append(os.path.join(img_dir, f))
            paired_masks.append(mask_path)
            
    return paired_imgs, paired_masks

if __name__ == "__main__":
    train_imgs, train_masks, val_imgs, val_masks = get_duts_tr_splits()
    print(f"DUTS-TR split: {len(train_imgs)} train images, {len(val_imgs)} val images.")
    te_imgs, te_masks = get_duts_te_data()
    print(f"DUTS-TE: {len(te_imgs)} test images.")
    
    # Test loading one sample
    train_ds = DUTSDataset(train_imgs[:10], train_masks[:10])
    img_t, mask_t, path = train_ds[0]
    print(f"Sample tensor shapes: image {img_t.shape}, mask {mask_t.shape}")
