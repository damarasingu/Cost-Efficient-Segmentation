import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from predict import load_pipeline, segment_photo_pipeline

def create_overlay(image_bgr, mask_01, color_bgr=(0, 255, 0), alpha=0.5):
    """
    Overlays a binary mask onto an image with semi-transparent colored alpha blending.
    """
    overlay = image_bgr.copy()
    colored_mask = np.zeros_like(image_bgr)
    colored_mask[mask_01 == 1] = color_bgr
    
    cv2.addWeighted(colored_mask, alpha, overlay, 1 - alpha, 0, overlay)
    
    # Draw boundary contour for sharp visualization
    contours, _ = cv2.findContours(mask_01, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color_bgr, 2)
    return overlay

def generate_report_visualizations(weights_path="weights.pt", out_dir="report_figures"):
    os.makedirs(out_dir, exist_ok=True)
    model = load_pipeline(weights_path)
    
    captured_img_dir = "captured/images_256"
    captured_mask_dir = "captured/masks_256"
    
    items = sorted([f for f in os.listdir(captured_img_dir) if f.endswith('.png')])
    
    print(f"Generating side-by-side overlay figures for {len(items)} captured photos...")
    
    for item in items:
        name = os.path.splitext(item)[0]
        img_path = os.path.join(captured_img_dir, item)
        mask_path = os.path.join(captured_mask_dir, item)
        
        photo_256_bgr = cv2.imread(img_path)
        gt_mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt_mask_01 = (gt_mask_gray >= 128).astype(np.uint8)
        
        # Predicted mask
        pred_mask_01 = segment_photo_pipeline(photo_256_bgr, model)
        
        # Ground truth overlay (Green)
        gt_overlay = create_overlay(photo_256_bgr, gt_mask_01, color_bgr=(0, 255, 0), alpha=0.45)
        # Prediction overlay (Cyan / Sky Blue)
        pred_overlay = create_overlay(photo_256_bgr, pred_mask_01, color_bgr=(255, 128, 0), alpha=0.45)
        
        # Convert BGR to RGB for matplotlib / saving
        rgb_orig = cv2.cvtColor(photo_256_bgr, cv2.COLOR_BGR2RGB)
        rgb_gt = cv2.cvtColor(gt_overlay, cv2.COLOR_BGR2RGB)
        rgb_pred = cv2.cvtColor(pred_overlay, cv2.COLOR_BGR2RGB)
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
        axes[0].imshow(rgb_orig)
        axes[0].set_title(f"Input Photo: {name.capitalize()}", fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(rgb_gt)
        axes[1].set_title("Ground Truth Mask Overlay", fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        axes[2].imshow(rgb_pred)
        axes[2].set_title("Predicted Mask Overlay", fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(out_dir, f"overlay_{name}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved overlay figure: {save_path}")

if __name__ == "__main__":
    generate_report_visualizations()
