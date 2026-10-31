import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import DUTSDataset, get_duts_tr_splits
from model import FastSegNet, DiceBCELoss
from metrics import MetricEvaluator, compute_ces

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, masks, _ in loader:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    evaluator = MetricEvaluator()
    for images, masks, _ in loader:
        images = images.to(device)
        logits = model(images)
        preds = (torch.sigmoid(logits) >= 0.5).to(torch.uint8)
        
        preds_np = preds.squeeze(1).cpu().numpy()
        masks_np = masks.squeeze(1).cpu().numpy()
        
        for p, m in zip(preds_np, masks_np):
            evaluator.update(p, m)
            
    return evaluator.compute()

def main():
    parser = argparse.ArgumentParser(description="Train FastSegNet on DUTS-TR")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--train-samples", type=int, default=4000, help="Subset of training samples for rapid convergence")
    parser.add_argument("--val-samples", type=int, default=500, help="Subset of validation samples for evaluation during training")
    parser.add_argument("--save-path", type=str, default="weights.pt", help="Path to save model weights")
    args = parser.parse_args()
    
    # Use full CPU capability during training
    num_threads = max(1, os.cpu_count() - 2) if os.cpu_count() else 8
    torch.set_num_threads(num_threads)
    print(f"Training using {num_threads} CPU threads...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading DUTS-TR splits...")
    train_imgs, train_masks, val_imgs, val_masks = get_duts_tr_splits()
    print(f"Total DUTS-TR pool: {len(train_imgs)} train, {len(val_imgs)} validation.")
    
    # Subsample for training efficiency if specified
    if args.train_samples and args.train_samples < len(train_imgs):
        train_imgs = train_imgs[:args.train_samples]
        train_masks = train_masks[:args.train_samples]
    if args.val_samples and args.val_samples < len(val_imgs):
        val_imgs = val_imgs[:args.val_samples]
        val_masks = val_masks[:args.val_samples]
        
    print(f"Active training set: {len(train_imgs)} images.")
    print(f"Active validation set: {len(val_imgs)} images.")
    
    train_dataset = DUTSDataset(train_imgs, train_masks, augment=True)
    val_dataset = DUTSDataset(val_imgs, val_masks, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    model = FastSegNet().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model architecture: FastSegNet with {total_params:,} parameters.")
    
    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    
    best_q = -1.0
    start_total_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        val_metrics = validate(model, val_loader, device)
        t_epoch = time.time() - t0
        
        q = val_metrics["q"]
        miou = val_metrics["miou"]
        bf1 = val_metrics["bf1"]
        
        print(f"Epoch {epoch:02d}/{args.epochs:02d} [{t_epoch:.1f}s] - Train Loss: {train_loss:.4f} | Val mIoU: {miou*100:.2f}% | Val BF1: {bf1*100:.2f}% | Val Q: {q*100:.2f}%")
        
        if q > best_q:
            best_q = q
            # Save weights (convert to float16 to optimize S_disk)
            state_dict = model.state_dict()
            half_state_dict = {k: v.half() if v.is_floating_point() else v for k, v in state_dict.items()}
            torch.save(half_state_dict, args.save_path)
            s_disk = os.path.getsize(args.save_path)
            print(f"  -> Best model saved! (Val Q: {q*100:.2f}%, S_disk: {s_disk} bytes / {s_disk/1024:.1f} KB)")
            
    total_time = time.time() - start_total_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes. Best Val Q: {best_q*100:.2f}%")

if __name__ == "__main__":
    main()
