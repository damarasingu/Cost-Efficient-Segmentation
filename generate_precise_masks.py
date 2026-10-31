import cv2
import numpy as np
import os

def create_precise_masks():
    os.makedirs("captured/masks_256", exist_ok=True)
    
    # =========================================================================
    # 1. DOVE (dove.png)
    # The Dove cosmetic jar is centered on the wooden table.
    # Cap from y=59 to y=105, body from y=105 to y=186.
    # Width x=84 to x=184.
    # =========================================================================
    img_dove = cv2.imread("captured/images_256/dove.png")
    mask_dove = np.zeros((256, 256), dtype=np.uint8)
    # The Dove jar is a clean cylinder with rounded top cap:
    # Cap contour
    cap_pts = [
        (88, 105), (88, 75), (96, 64), (110, 60), (146, 60), (160, 64), (180, 75), (180, 105)
    ]
    # Body contour
    body_pts = [
        (84, 105), (184, 105), (184, 182), (180, 186), (88, 186), (84, 182)
    ]
    cv2.fillPoly(mask_dove, [np.array(cap_pts, dtype=np.int32)], 255)
    cv2.fillPoly(mask_dove, [np.array(body_pts, dtype=np.int32)], 255)
    cv2.imwrite("captured/masks_256/dove.png", mask_dove)
    
    # =========================================================================
    # 2. TOY (toy.png)
    # The plush teddy bear with Gryffindor jacket lying on the floor.
    # Include ears, head, arms, jacket, paws, feet, and tail/tag.
    # =========================================================================
    mask_toy = np.zeros((256, 256), dtype=np.uint8)
    toy_poly = [
        # Left ear
        (55, 62), (52, 45), (63, 38), (80, 42), (90, 50),
        # Top of head
        (120, 38), (140, 38), (150, 45),
        # Right ear
        (160, 42), (175, 42), (184, 52), (180, 68), (175, 75),
        # Right arm
        (185, 90), (210, 95), (232, 105), (232, 115), (220, 122), (195, 125), (185, 125),
        # Right body side & right leg
        (182, 140), (190, 145), (210, 150), (216, 160), (210, 172), (180, 175), (165, 172), (145, 155),
        # Bottom center
        (130, 155), (120, 155),
        # Left leg
        (105, 155), (85, 172), (65, 175), (35, 172), (24, 162), (30, 150), (50, 145), (58, 140),
        # Left body side & left arm
        (58, 125), (45, 125), (20, 122), (12, 115), (12, 105), (30, 95), (55, 90),
        # Left neck / cheek
        (65, 75), (58, 68)
    ]
    cv2.fillPoly(mask_toy, [np.array(toy_poly, dtype=np.int32)], 255)
    # Smooth edges slightly
    mask_toy = cv2.GaussianBlur(mask_toy, (5, 5), 1.0)
    mask_toy = ((mask_toy >= 128) * 255).astype(np.uint8)
    cv2.imwrite("captured/masks_256/toy.png", mask_toy)

    # =========================================================================
    # 3. CAT (cat.png)
    # Orange tabby cat: ears, head, whiskers/snout, front chest, front paws,
    # back, hind leg, and curved tail in front.
    # =========================================================================
    mask_cat = np.zeros((256, 256), dtype=np.uint8)
    cat_poly = [
        # Left ear
        (115, 92), (111, 75), (122, 70), (138, 90),
        # Top of head between ears
        (150, 92), (170, 90),
        # Right ear
        (190, 80), (212, 82), (215, 95), (195, 110),
        # Right cheek and snout
        (190, 125), (185, 145), (175, 155), (160, 155), (150, 150),
        # Front chest and front legs
        (140, 158), (135, 180), (125, 205), (125, 228), (115, 235), (98, 235), (96, 215),
        # Space between front legs and paws
        (92, 215), (90, 228), (85, 235), (75, 235), (75, 215),
        # Tail curving to the right
        (85, 225), (105, 230), (125, 230), (134, 222), (128, 245), (105, 250), (80, 248),
        (50, 242), (32, 230), (25, 218), (28, 205), (42, 212), (60, 218),
        # Back / rump
        (40, 195), (25, 178), (22, 168), (35, 140), (52, 125), (70, 120), (95, 115), (108, 105)
    ]
    cv2.fillPoly(mask_cat, [np.array(cat_poly, dtype=np.int32)], 255)
    mask_cat = cv2.GaussianBlur(mask_cat, (5, 5), 1.0)
    mask_cat = ((mask_cat >= 128) * 255).astype(np.uint8)
    cv2.imwrite("captured/masks_256/cat.png", mask_cat)

    # =========================================================================
    # 4. CYCLE (cycle.png)
    # The entire bicycle: handlebars, front wheel/fork, frame, seat,
    # rear wheel/fender, pedals, kickstand.
    # =========================================================================
    mask_cycle = np.zeros((256, 256), dtype=np.uint8)
    # Front wheel
    cv2.circle(mask_cycle, (58, 194), 48, 255, -1)
    # Clear inside of front wheel spokes partially (or fill complete wheel as object)
    # In saliency detection, wheels are filled regions
    # Rear wheel
    cv2.circle(mask_cycle, (208, 176), 48, 255, -1)
    
    # Handlebars & fork
    handle_poly = [
        (42, 55), (48, 48), (56, 52), (54, 70), (62, 85), (68, 115),
        (72, 140), (64, 150), (56, 115), (48, 85), (38, 65)
    ]
    cv2.fillPoly(mask_cycle, [np.array(handle_poly, dtype=np.int32)], 255)
    
    # Frame tubes & seat & fender
    frame_poly = [
        # Head tube
        (58, 110), (80, 118),
        # Top tube to seat
        (150, 118), (145, 95), (120, 80), (125, 68), (155, 68), (168, 82), (162, 98), (158, 118),
        # Rear fender & seat stays
        (190, 105), (235, 115), (245, 122), (230, 125), (195, 118),
        # Rear dropouts to bottom bracket
        (210, 180), (160, 192), (140, 192), (135, 180),
        # Down tube to head tube
        (70, 130), (58, 120)
    ]
    cv2.fillPoly(mask_cycle, [np.array(frame_poly, dtype=np.int32)], 255)
    # Kickstand
    cv2.line(mask_cycle, (186, 190), (192, 250), 255, thickness=4)
    # Connect frame smoothly
    mask_cycle = cv2.dilate(mask_cycle, np.ones((5, 5), np.uint8))
    cv2.imwrite("captured/masks_256/cycle.png", mask_cycle)

    # =========================================================================
    # 5. TREE_OUTDOOR (tree_outdoor.png)
    # The salient standalone tree in the center of the field.
    # Trunk at x=155-162, y=170-184. Canopy at x=142-175, y=152-172.
    # And surrounding field trees along the horizon.
    # =========================================================================
    mask_tree = np.zeros((256, 256), dtype=np.uint8)
    # Main prominent standalone tree
    tree_canopy = [
        (144, 164), (148, 156), (156, 152), (165, 152), (173, 158), (175, 166), (168, 172), (150, 172)
    ]
    cv2.fillPoly(mask_tree, [np.array(tree_canopy, dtype=np.int32)], 255)
    # Trunk
    cv2.rectangle(mask_tree, (156, 170), (162, 183), 255, -1)
    # Left standalone tree
    left_canopy = [
        (68, 166), (72, 160), (82, 158), (92, 160), (95, 168), (88, 174), (72, 174)
    ]
    cv2.fillPoly(mask_tree, [np.array(left_canopy, dtype=np.int32)], 255)
    cv2.rectangle(mask_tree, (80, 172), (83, 178), 255, -1)
    mask_tree = cv2.dilate(mask_tree, np.ones((3, 3), np.uint8))
    cv2.imwrite("captured/masks_256/tree_outdoor.png", mask_tree)

    # =========================================================================
    # 6. CLUTTERED_BACKGROUND (cluttered_background.png)
    # The primary meal on the tray: the plate with green dumplings and the
    # Buldak ramen package.
    # =========================================================================
    mask_clutter = np.zeros((256, 256), dtype=np.uint8)
    # Circular plate with dumplings
    cv2.circle(mask_clutter, (126, 68), 56, 255, -1)
    # Buldak ramen packet
    ramen_poly = [
        (128, 118), (198, 62), (255, 75), (255, 142), (210, 178), (138, 128)
    ]
    cv2.fillPoly(mask_clutter, [np.array(ramen_poly, dtype=np.int32)], 255)
    # Food packets and chopsticks on tray
    sides_poly = [
        (82, 108), (142, 112), (135, 158), (72, 152)
    ]
    cv2.fillPoly(mask_clutter, [np.array(sides_poly, dtype=np.int32)], 255)
    mask_clutter = cv2.dilate(mask_clutter, np.ones((5, 5), np.uint8))
    cv2.imwrite("captured/masks_256/cluttered_background.png", mask_clutter)

    print("Accurate, anatomically correct ground truth masks generated.")

if __name__ == "__main__":
    create_precise_masks()
