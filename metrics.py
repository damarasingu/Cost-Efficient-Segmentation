import torch
import torch.nn.functional as F
import numpy as np

def extract_boundary(mask_tensor):
    """
    Extract boundary pixels for a binary mask (H, W).
    Rule: A boundary pixel is an object pixel (1) with a background pixel (0)
    directly above, below, left, or right of it. Pixels outside the image count as background.
    """
    # mask_tensor: shape (H, W), dtype float or bool (1 for object, 0 for background)
    if not isinstance(mask_tensor, torch.Tensor):
        mask_tensor = torch.from_numpy(mask_tensor)
    
    mask = (mask_tensor > 0.5).float()
    H, W = mask.shape
    
    # Pad by 1 with 0 (outside counts as background)
    padded = F.pad(mask.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="constant", value=0).squeeze()
    
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    
    # Interior pixel has all 4 neighbors equal to 1
    interior = (mask == 1) & (up == 1) & (down == 1) & (left == 1) & (right == 1)
    boundary = (mask == 1) & (~interior)
    return boundary

def compute_boundary_f1(pred_mask, true_mask):
    """
    Computes Boundary F1 (BF1) for a single image according to assignment specifications:
    1. A boundary pixel is an object pixel with a background pixel directly above, below, left or right.
       Pixels outside the image count as background.
    2. A predicted boundary pixel is correct if a true boundary pixel lies within its 5x5 window.
       A true boundary pixel is found if a predicted boundary pixel lies within its 5x5 window.
    3. Precision P = correct_pred / total_pred.
       Recall R = found_true / total_true.
       BF1 = 2*P*R / (P + R), or 0 if P + R == 0.
    4. If neither mask has boundary pixels, BF1 is 1. If only one does, it is 0.
    """
    if not isinstance(pred_mask, torch.Tensor):
        pred_mask = torch.from_numpy(pred_mask)
    if not isinstance(true_mask, torch.Tensor):
        true_mask = torch.from_numpy(true_mask)
        
    b_pred = extract_boundary(pred_mask)
    b_true = extract_boundary(true_mask)
    
    n_pred = b_pred.sum().item()
    n_true = b_true.sum().item()
    
    if n_pred == 0 and n_true == 0:
        return 1.0
    if n_pred == 0 or n_true == 0:
        return 0.0
        
    # 5x5 window dilation using F.max_pool2d
    b_pred_4d = b_pred.float().unsqueeze(0).unsqueeze(0)
    b_true_4d = b_true.float().unsqueeze(0).unsqueeze(0)
    
    d_pred = (F.max_pool2d(b_pred_4d, kernel_size=5, stride=1, padding=2) > 0.5).squeeze()
    d_true = (F.max_pool2d(b_true_4d, kernel_size=5, stride=1, padding=2) > 0.5).squeeze()
    
    correct_pred = (b_pred & d_true).sum().item()
    found_true = (b_true & d_pred).sum().item()
    
    prec = correct_pred / n_pred
    rec = found_true / n_true
    
    if prec + rec == 0:
        return 0.0
    return (2.0 * prec * rec) / (prec + rec)

class MetricEvaluator:
    """
    Accumulates confusion matrix for dataset-level mIoU and per-image BF1 scores.
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        # 2x2 confusion matrix: [true_class, pred_class]
        self.confusion_matrix = np.zeros((2, 2), dtype=np.int64)
        self.bf1_scores = []
        
    def update(self, pred_mask, true_mask):
        """
        pred_mask: 2D array or Tensor of 0s and 1s (256x256)
        true_mask: 2D array or Tensor of 0s and 1s (256x256)
        """
        if isinstance(pred_mask, torch.Tensor):
            pred_arr = pred_mask.detach().cpu().numpy().astype(np.uint8)
        else:
            pred_arr = np.asarray(pred_mask, dtype=np.uint8)
            
        if isinstance(true_mask, torch.Tensor):
            true_arr = true_mask.detach().cpu().numpy().astype(np.uint8)
        else:
            true_arr = np.asarray(true_mask, dtype=np.uint8)
            
        pred_bin = (pred_arr >= 1).astype(np.int64)
        true_bin = (true_arr >= 1).astype(np.int64)
        
        # Accumulate confusion matrix
        # 0: bg, 1: fg
        c00 = np.sum((true_bin == 0) & (pred_bin == 0))
        c01 = np.sum((true_bin == 0) & (pred_bin == 1))
        c10 = np.sum((true_bin == 1) & (pred_bin == 0))
        c11 = np.sum((true_bin == 1) & (pred_bin == 1))
        
        self.confusion_matrix[0, 0] += c00
        self.confusion_matrix[0, 1] += c01
        self.confusion_matrix[1, 0] += c10
        self.confusion_matrix[1, 1] += c11
        
        # BF1 per image
        bf1 = compute_boundary_f1(pred_bin, true_bin)
        self.bf1_scores.append(bf1)
        return bf1
        
    def compute(self):
        cm = self.confusion_matrix
        # Background IoU
        bg_denom = cm[0, 0] + cm[0, 1] + cm[1, 0]
        bg_iou = cm[0, 0] / bg_denom if bg_denom > 0 else 0.0
        
        # Object IoU
        obj_denom = cm[1, 1] + cm[0, 1] + cm[1, 0]
        obj_iou = cm[1, 1] / obj_denom if obj_denom > 0 else 0.0
        
        miou = (bg_iou + obj_iou) / 2.0
        mean_bf1 = float(np.mean(self.bf1_scores)) if len(self.bf1_scores) > 0 else 0.0
        q = 0.70 * miou + 0.30 * mean_bf1
        
        return {
            "miou": float(miou),
            "bg_iou": float(bg_iou),
            "obj_iou": float(obj_iou),
            "bf1": float(mean_bf1),
            "q": float(q),
            "num_samples": len(self.bf1_scores)
        }

def compute_ces(q, t_cpu_ms, s_disk_bytes):
    """
    CES = Q * [1 - 0.20 * (TCPU / 50 ms) - 0.10 * min(1, Sdisk / 4 MiB)] if TCPU <= 50 ms else 0
    where 4 MiB = 4,194,304 bytes.
    """
    if t_cpu_ms > 50.0:
        return 0.0
    four_mib = 4194304.0
    ces = q * (1.0 - 0.20 * (t_cpu_ms / 50.0) - 0.10 * min(1.0, s_disk_bytes / four_mib))
    return max(0.0, float(ces))

if __name__ == "__main__":
    # Unit tests for metrics
    # Test case 1: identical square
    m1 = np.zeros((256, 256), dtype=np.uint8)
    m1[100:150, 100:150] = 1
    evaluator = MetricEvaluator()
    evaluator.update(m1, m1)
    res = evaluator.compute()
    print("Test 1 (Identical masks):", res)
    assert abs(res["miou"] - 1.0) < 1e-5
    assert abs(res["bf1"] - 1.0) < 1e-5
    assert abs(res["q"] - 1.0) < 1e-5
    
    # Test case 2: 1 pixel shift
    m2 = np.zeros((256, 256), dtype=np.uint8)
    m2[101:151, 101:151] = 1
    evaluator.reset()
    evaluator.update(m2, m1)
    res2 = evaluator.compute()
    print("Test 2 (1-pixel shift within 5x5 window):", res2)
    # Since boundary shift is 1 pixel (well within 5x5 window), BF1 should be 1.0
    assert abs(res2["bf1"] - 1.0) < 1e-5
    
    # Test CES calculation from PDF example:
    # Q = 0.70 at 20 ms with a 0.5 MiB file gives 0.70 * (1 - 0.08 - 0.0125) = 0.63525
    ces_example = compute_ces(0.70, 20.0, 0.5 * 1024 * 1024)
    print("CES PDF example test:", ces_example)
    assert abs(ces_example - 0.63525) < 1e-4
    print("All metric tests passed successfully!")
