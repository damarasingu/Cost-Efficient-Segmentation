import os
import sys

# Section 2b: Enforce single-thread CPU execution BEFORE importing libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import time
import argparse
import cv2
import numpy as np
import torch

cv2.setNumThreads(1)
torch.set_num_threads(1)

from model import FastSegNet

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

def load_pipeline(weights_path="weights.pt"):
    model = FastSegNet()
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model parameters file not found: {weights_path}")
        
    state_dict = torch.load(weights_path, map_location="cpu")
    # Cast float16 weights back to float32 for CPU inference execution
    state_dict_f32 = {k: v.float() if v.is_floating_point() else v for k, v in state_dict.items()}
    model.load_state_dict(state_dict_f32)
    model.eval()
    return model

def segment_photo_pipeline(photo_256_bgr, model):
    """
    The timed pipeline starts with a 256x256 photo in memory and ends with its mask,
    a 256x256 array of 0s and 1s, in memory.
    Every step in between counts, including colour conversion and tensor conversion.
    """
    # 1. Colour conversion (BGR to RGB)
    rgb = cv2.cvtColor(photo_256_bgr, cv2.COLOR_BGR2RGB)
    
    # 2. Normalization
    norm = ((rgb.astype(np.float32) / 255.0) - MEAN) / STD
    
    # 3. Tensor layout conversion (H, W, C) -> (1, C, H, W)
    inp = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0).float()
    
    # 4. Neural network forward pass
    with torch.no_grad():
        logits = model(inp)
        
    # 5. Thresholding to binary 0 and 1 mask in memory (256x256 array of 0s and 1s)
    # Logit >= 0 is equivalent to Sigmoid(logit) >= 0.5
    mask_01 = (logits[0, 0] >= 0.0).numpy().astype(np.uint8)
    return mask_01

def main():
    parser = argparse.ArgumentParser(description="Segment an image and measure single-core CPU latency")
    parser.add_argument("--image", type=str, required=True, help="Path to input photo")
    parser.add_argument("--out", type=str, required=True, help="Path to save output mask PNG")
    parser.add_argument("--weights", type=str, default="weights.pt", help="Path to saved model parameters")
    args = parser.parse_args()
    
    # Untimed: load model parameters
    model = load_pipeline(args.weights)
    s_disk_bytes = os.path.getsize(args.weights)
    
    # Untimed: read image and resize original photo to 256x256 with cv2.INTER_AREA (Section 1b)
    orig_img = cv2.imread(args.image)
    if orig_img is None:
        raise FileNotFoundError(f"Could not load image at {args.image}")
    photo_256 = cv2.resize(orig_img, (256, 256), interpolation=cv2.INTER_AREA)
    
    # Untimed: Warmup run (first run is slow while libraries initialize)
    _ = segment_photo_pipeline(photo_256, model)
    
    # Run once to get the final output mask
    mask_01 = segment_photo_pipeline(photo_256, model)
    
    # Untimed: write output mask as 0 and 255 PNG
    mask_255 = (mask_01 * 255).astype(np.uint8)
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(args.out, mask_255)
    
    # Benchmark: time 20 more runs with time.process_time()
    num_runs = 20
    times_ms = []
    for _ in range(num_runs):
        t0 = time.process_time()
        _ = segment_photo_pipeline(photo_256, model)
        t1 = time.process_time()
        times_ms.append((t1 - t0) * 1000.0)
        
    avg_t_ms = sum(times_ms) / len(times_ms)
    
    # Output format specified in PDF:
    # T_CPU_ms: 12.34
    # S_disk_bytes: 425767
    print(f"T_CPU_ms: {avg_t_ms:.2f}")
    print(f"S_disk_bytes: {s_disk_bytes}")

if __name__ == "__main__":
    main()
