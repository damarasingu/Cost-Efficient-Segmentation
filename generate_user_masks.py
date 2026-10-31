import cv2
import numpy as np
import os

def create_masks():
    os.makedirs("captured/masks_256", exist_ok=True)
    
    # 1. cat.png
    # Let's inspect cat.png using GrabCut initialized with a bounding box around the central cat
    img_cat = cv2.imread("captured/images_256/cat.png")
    mask = np.zeros(img_cat.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    # Bounding box around cat
    rect_cat = (30, 20, 195, 220)
    cv2.grabCut(img_cat, mask, rect_cat, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask_cat = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
    mask_cat = cv2.morphologyEx(mask_cat, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cv2.imwrite("captured/masks_256/cat.png", mask_cat)
    
    # 2. cluttered_background.png
    img_clutter = cv2.imread("captured/images_256/cluttered_background.png")
    mask = np.zeros(img_clutter.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect_clutter = (25, 25, 205, 210)
    cv2.grabCut(img_clutter, mask, rect_clutter, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask_clutter = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
    mask_clutter = cv2.morphologyEx(mask_clutter, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cv2.imwrite("captured/masks_256/cluttered_background.png", mask_clutter)

    # 3. cycle.png
    img_cycle = cv2.imread("captured/images_256/cycle.png")
    mask = np.zeros(img_cycle.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect_cycle = (15, 30, 225, 200)
    cv2.grabCut(img_cycle, mask, rect_cycle, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask_cycle = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
    mask_cycle = cv2.morphologyEx(mask_cycle, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cv2.imwrite("captured/masks_256/cycle.png", mask_cycle)

    # 4. dove.png
    img_dove = cv2.imread("captured/images_256/dove.png")
    mask = np.zeros(img_dove.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect_dove = (20, 30, 215, 200)
    cv2.grabCut(img_dove, mask, rect_dove, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask_dove = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
    mask_dove = cv2.morphologyEx(mask_dove, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cv2.imwrite("captured/masks_256/dove.png", mask_dove)

    # 5. toy.png
    img_toy = cv2.imread("captured/images_256/toy.png")
    mask = np.zeros(img_toy.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect_toy = (35, 25, 185, 215)
    cv2.grabCut(img_toy, mask, rect_toy, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask_toy = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
    mask_toy = cv2.morphologyEx(mask_toy, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cv2.imwrite("captured/masks_256/toy.png", mask_toy)

    # 6. tree_outdoor.png
    img_tree = cv2.imread("captured/images_256/tree_outdoor.png")
    mask = np.zeros(img_tree.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect_tree = (20, 10, 215, 235)
    cv2.grabCut(img_tree, mask, rect_tree, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask_tree = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
    mask_tree = cv2.morphologyEx(mask_tree, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cv2.imwrite("captured/masks_256/tree_outdoor.png", mask_tree)

    print("Generated ground truth masks for user photos:")
    for f in sorted(os.listdir("captured/masks_256")):
        m = cv2.imread(os.path.join("captured/masks_256", f), 0)
        print(f, f"{(m == 255).mean()*100:.1f}% object")

if __name__ == "__main__":
    create_masks()
