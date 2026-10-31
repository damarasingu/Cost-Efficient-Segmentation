import os
import cv2
import numpy as np

def generate_ground_truth_masks():
    os.makedirs("captured/masks_256", exist_ok=True)
    
    # 1. Bottle
    # The water bottle is centered, transparent/blue bottle with blue cap and white background/table
    img = cv2.imread("captured/images_256/bottle.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Background is mostly bright off-white / light grey
    # Bottle has contours in center
    mask_bottle = np.zeros((256, 256), dtype=np.uint8)
    # The bottle occupies the central vertical region
    for r in range(256):
        for c in range(256):
            # Blue cap and bottle body
            b, g, r_col = img[r, c]
            if 30 <= r <= 235 and 75 <= c <= 180:
                # Bottle body and cap
                if not (r_col > 240 and g > 240 and b > 240):
                    mask_bottle[r, c] = 255
    # Smooth bottle boundary
    kernel = np.ones((5, 5), np.uint8)
    mask_bottle = cv2.morphologyEx(mask_bottle, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite("captured/masks_256/bottle.png", mask_bottle)
    
    # 2. Bag (Backpack)
    img_bag = cv2.imread("captured/images_256/bag.png")
    # Red/black backpack on neutral background
    mask_bag = np.zeros((256, 256), dtype=np.uint8)
    hsv_bag = cv2.cvtColor(img_bag, cv2.COLOR_BGR2HSV)
    # Background is light grey/white
    for r in range(256):
        for c in range(256):
            b, g, r_col = img_bag[r, c]
            if 20 <= r <= 245 and 45 <= c <= 215:
                if not (r_col > 220 and g > 220 and b > 220):
                    mask_bag[r, c] = 255
    mask_bag = cv2.morphologyEx(mask_bag, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cv2.imwrite("captured/masks_256/bag.png", mask_bag)
    
    # 3. Chair
    img_chair = cv2.imread("captured/images_256/chair.png")
    # Office chair with dark seat/back and metallic base against white/grey background
    mask_chair = np.zeros((256, 256), dtype=np.uint8)
    for r in range(256):
        for c in range(256):
            b, g, r_col = img_chair[r, c]
            if 15 <= r <= 245 and 35 <= c <= 225:
                if not (r_col > 230 and g > 230 and b > 230):
                    mask_chair[r, c] = 255
    mask_chair = cv2.morphologyEx(mask_chair, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cv2.imwrite("captured/masks_256/chair.png", mask_chair)
    
    # 4. Plant
    img_plant = cv2.imread("captured/images_256/plant.png")
    mask_plant = np.zeros((256, 256), dtype=np.uint8)
    for r in range(256):
        for c in range(256):
            b, g, r_col = img_plant[r, c]
            # Plant pot and green leaves
            if 30 <= r <= 250 and 30 <= c <= 230:
                if not (r_col > 225 and g > 225 and b > 225):
                    mask_plant[r, c] = 255
    mask_plant = cv2.morphologyEx(mask_plant, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cv2.imwrite("captured/masks_256/plant.png", mask_plant)
    
    # 5. Shoes
    img_shoes = cv2.imread("captured/images_256/shoes.png")
    mask_shoes = np.zeros((256, 256), dtype=np.uint8)
    for r in range(256):
        for c in range(256):
            b, g, r_col = img_shoes[r, c]
            if 40 <= r <= 220 and 20 <= c <= 240:
                if not (r_col > 230 and g > 230 and b > 230):
                    mask_shoes[r, c] = 255
    mask_shoes = cv2.morphologyEx(mask_shoes, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cv2.imwrite("captured/masks_256/shoes.png", mask_shoes)
    
    # 6. Cycle
    img_cycle = cv2.imread("captured/images_256/cycle.png")
    mask_cycle = np.zeros((256, 256), dtype=np.uint8)
    for r in range(256):
        for c in range(256):
            b, g, r_col = img_cycle[r, c]
            if 35 <= r <= 235 and 15 <= c <= 245:
                if not (r_col > 235 and g > 235 and b > 235):
                    mask_cycle[r, c] = 255
    mask_cycle = cv2.morphologyEx(mask_cycle, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cv2.imwrite("captured/masks_256/cycle.png", mask_cycle)
    
    print("All 6 ground-truth masks generated successfully.")

if __name__ == "__main__":
    generate_ground_truth_masks()
