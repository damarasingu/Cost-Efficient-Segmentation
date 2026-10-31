import cv2
import numpy as np

def update_cluttered_mask():
    img = cv2.imread("captured/images_256/cluttered_background.png")
    mask = np.zeros((256, 256), dtype=np.uint8)
    
    # 1. White paper plate with green dumplings (Top)
    cv2.circle(mask, (126, 68), 58, 255, -1)
    
    # 2. Red Buldak Ramen packet (Right)
    ramen_pts = [
        (130, 118), (170, 72), (210, 52), (255, 68), 
        (255, 145), (225, 170), (205, 182), (135, 130)
    ]
    cv2.fillPoly(mask, [np.array(ramen_pts, dtype=np.int32)], 255)
    
    # 3. Shredded cabbage zip-lock pouch (Center)
    cabbage_pts = [
        (65, 135), (105, 108), (145, 118), (140, 168), (95, 168), (68, 145)
    ]
    cv2.fillPoly(mask, [np.array(cabbage_pts, dtype=np.int32)], 255)
    
    # 4. Diced chicken meat pouch (Middle-left) - accurately included
    chicken_pts = [
        (30, 145), (115, 145), (115, 205), (30, 205)
    ]
    cv2.fillPoly(mask, [np.array(chicken_pts, dtype=np.int32)], 255)
    
    # 5. Sweet corn zip-lock pouch (Middle-right)
    corn_pts = [
        (120, 165), (145, 138), (185, 145), (185, 195), (135, 195), (120, 175)
    ]
    cv2.fillPoly(mask, [np.array(corn_pts, dtype=np.int32)], 255)
    
    # 6. Shredded carrot zip-lock pouch (Bottom-right)
    carrot_pts = [
        (130, 205), (160, 175), (225, 165), (225, 220), (175, 240), (135, 225)
    ]
    cv2.fillPoly(mask, [np.array(carrot_pts, dtype=np.int32)], 255)
    
    # 7. Square cheese slice (Bottom-left)
    cheese_pts = [
        (10, 225), (85, 188), (118, 215), (42, 255), (10, 255)
    ]
    cv2.fillPoly(mask, [np.array(cheese_pts, dtype=np.int32)], 255)
    
    # 8. Chopsticks & wooden fork (Left side)
    utensils_pts = [
        (0, 140), (38, 98), (92, 150), (32, 200), (0, 185)
    ]
    cv2.fillPoly(mask, [np.array(utensils_pts, dtype=np.int32)], 255)
    
    # 9. Top-left chopsticks in wrapper
    top_chopsticks = [
        (0, 35), (25, 10), (120, 2), (125, 45), (65, 60), (0, 50)
    ]
    cv2.fillPoly(mask, [np.array(top_chopsticks, dtype=np.int32)], 255)
    
    # Morphological close to create smooth contiguous coverage of the meal items
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    cv2.imwrite("captured/masks_256/cluttered_background.png", mask)
    print(f"Updated cluttered_background.png mask: {(mask == 255).mean()*100:.1f}% foreground")

if __name__ == "__main__":
    update_cluttered_mask()
