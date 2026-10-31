import os
import sys

# Section 2b: Enforce single-thread CPU execution BEFORE importing libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import time
import json
import argparse
import cv2
import numpy as np
import torch

cv2.setNumThreads(1)
torch.set_num_threads(1)

from model import FastSegNet
from predict import load_pipeline, segment_photo_pipeline
from dataset import get_duts_tr_splits, get_duts_te_data, load_and_preprocess_mask
from metrics import MetricEvaluator, compute_ces

def evaluate_subset(image_paths, mask_paths, model, measure_timing=False, desc="Subset"):
    evaluator = MetricEvaluator()
    total_cpu_time = 0.0
    num_samples = len(image_paths)
    
    print(f"Evaluating {desc} ({num_samples} samples)...")
    
    for i in range(num_samples):
        img_path = image_paths[i]
        mask_path = mask_paths[i]
        
        # Untimed: read image and resize to 256x256 with cv2.INTER_AREA
        orig_img = cv2.imread(img_path)
        if orig_img is None:
            continue
        photo_256 = cv2.resize(orig_img, (256, 256), interpolation=cv2.INTER_AREA)
        
        # Ground truth mask (256x256, values in {0, 1})
        gt_mask_01, _ = load_and_preprocess_mask(mask_path)
        
        # Timed pipeline
        if measure_timing:
            t0 = time.process_time()
            pred_mask_01 = segment_photo_pipeline(photo_256, model)
            t1 = time.process_time()
            total_cpu_time += (t1 - t0)
        else:
            pred_mask_01 = segment_photo_pipeline(photo_256, model)
            
        evaluator.update(pred_mask_01, gt_mask_01)
        
        if (i + 1) % 500 == 0 or (i + 1) == num_samples:
            sys.stdout.write(f"\rProgress: {i + 1}/{num_samples}")
            sys.stdout.flush()
    print()
    
    results = evaluator.compute()
    avg_t_ms = (total_cpu_time / num_samples * 1000.0) if (measure_timing and num_samples > 0) else None
    results["t_cpu_ms"] = avg_t_ms
    return results

def get_captured_data(captured_dir="captured"):
    img_dir = os.path.join(captured_dir, "images_256")
    mask_dir = os.path.join(captured_dir, "masks_256")
    
    if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
        return [], []
        
    imgs, masks = [], []
    for f in sorted(os.listdir(img_dir)):
        if f.lower().endswith(('.png', '.jpg')):
            img_path = os.path.join(img_dir, f)
            mask_name = os.path.splitext(f)[0] + ".png"
            mask_path = os.path.join(mask_dir, mask_name)
            if os.path.exists(mask_path):
                imgs.append(img_path)
                masks.append(mask_path)
    return imgs, masks

def main():
    parser = argparse.ArgumentParser(description="Evaluate FastSegNet on Val, DUTS-TE, and Captured Photos")
    parser.add_argument("--weights", type=str, default="weights.pt", help="Path to weights file")
    parser.add_argument("--val-limit", type=int, default=None, help="Limit val evaluation samples (default all)")
    parser.add_argument("--te-limit", type=int, default=None, help="Limit test evaluation samples (default all 5019)")
    args = parser.parse_args()
    
    model = load_pipeline(args.weights)
    s_disk_bytes = os.path.getsize(args.weights)
    print(f"Loaded weights from {args.weights} ({s_disk_bytes} bytes / {s_disk_bytes/1024:.2f} KB)")
    
    # Warmup
    dummy = np.zeros((256, 256, 3), dtype=np.uint8)
    _ = segment_photo_pipeline(dummy, model)
    
    # 1. Validation Set
    _, _, val_imgs, val_masks = get_duts_tr_splits()
    if args.val_limit:
        val_imgs = val_imgs[:args.val_limit]
        val_masks = val_masks[:args.val_limit]
    val_res = evaluate_subset(val_imgs, val_masks, model, measure_timing=False, desc="Validation Set")
    
    # 2. DUTS-TE
    te_imgs, te_masks = get_duts_te_data()
    if args.te_limit:
        te_imgs = te_imgs[:args.te_limit]
        te_masks = te_masks[:args.te_limit]
    te_res = evaluate_subset(te_imgs, te_masks, model, measure_timing=True, desc="DUTS-TE")
    
    # 3. Captured Photos
    cap_imgs, cap_masks = get_captured_data()
    cap_res = None
    if cap_imgs:
        cap_res = evaluate_subset(cap_imgs, cap_masks, model, measure_timing=False, desc="Captured Photos")
        
    # Compute CES on DUTS-TE
    t_cpu = te_res["t_cpu_ms"]
    ces = compute_ces(te_res["q"], t_cpu, s_disk_bytes)
    te_res["ces"] = ces
    
    # Summary Table
    print("\n" + "="*80)
    print("FINAL EVALUATION METRICS SUMMARY TABLE")
    print("="*80)
    print(f"{'Dataset':<18} | {'mIoU (%)':<10} | {'BF1 (%)':<10} | {'Q':<8} | {'T_CPU (ms)':<10} | {'S_disk (B)':<10} | {'CES':<8}")
    print("-" * 80)
    print(f"{'Validation':<18} | {val_res['miou']*100:<10.2f} | {val_res['bf1']*100:<10.2f} | {val_res['q']:<8.4f} | {'-':<10} | {s_disk_bytes:<10} | {'-'}")
    print(f"{'DUTS-TE':<18} | {te_res['miou']*100:<10.2f} | {te_res['bf1']*100:<10.2f} | {te_res['q']:<8.4f} | {t_cpu:<10.2f} | {s_disk_bytes:<10} | {ces:<8.4f}")
    if cap_res:
        print(f"{'Captured':<18} | {cap_res['miou']*100:<10.2f} | {cap_res['bf1']*100:<10.2f} | {cap_res['q']:<8.4f} | {'-':<10} | {s_disk_bytes:<10} | {'-'}")
    print("="*80)
    
    out_record = {
        "s_disk_bytes": s_disk_bytes,
        "val": val_res,
        "duts_te": te_res,
        "captured": cap_res
    }
    with open("eval_results.json", "w") as f:
        json.dump(out_record, f, indent=2)
    print("Saved evaluation results to eval_results.json")

if __name__ == "__main__":
    main()
