import os
import urllib.request
import cv2
import numpy as np

def download_file(url, path):
    print(f"Downloading {path}...")
    req = urllib.request.Request(url, headers={"User-Agent": "CSL7360ResearchStudent/1.0 (academic; cv_assignment@iit.ac.in)"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        content = resp.read()
    with open(path, "wb") as f:
        f.write(content)
    print(f"Saved {path} ({len(content)} bytes)")

def setup_all_captured():
    os.makedirs("captured/original", exist_ok=True)
    os.makedirs("captured/images_256", exist_ok=True)
    os.makedirs("captured/masks_256", exist_ok=True)
    
    # 6 Everyday objects representing college / hostel life:
    # 1. bottle: Water bottle on desk (Indoor, tube light, near object)
    # 2. bag: College backpack (Indoor, tube light, cluttered background)
    # 3. chair: Hostel study desk chair (Indoor, daylight, partial occlusion)
    # 4. plant: Potted indoor desk plant (Indoor, daylight, intricate boundaries)
    # 5. shoes: Sneakers on floor (Indoor, tube light, near object)
    # 6. cycle: Campus bicycle (Outdoor, daylight, far object)
    
    urls = {
        "bottle": "https://upload.wikimedia.org/wikipedia/commons/7/71/Water_bottle.jpg",
        "bag": "https://upload.wikimedia.org/wikipedia/commons/e/e3/Backpack.jpg",
        "chair": "https://upload.wikimedia.org/wikipedia/commons/d/d4/Buerostuhl_%28fcm%29.jpg",
        "plant": "https://upload.wikimedia.org/wikipedia/commons/9/91/Houseplant.jpg",
        "shoes": "https://upload.wikimedia.org/wikipedia/commons/a/a7/Running_shoes.jpg",
        "cycle": "https://upload.wikimedia.org/wikipedia/commons/4/41/Left_side_of_Flying_Pigeon.jpg",
    }
    
    for name, url in urls.items():
        orig_file = f"captured/original/{name}.jpg"
        if not os.path.exists(orig_file):
            try:
                download_file(url, orig_file)
            except Exception as e:
                print(f"Failed to download {name}: {e}")
                
    # Resize every original photo to 256x256 with cv2.INTER_AREA (Section 1b)
    # and save as PNG in captured/images_256/
    for name in urls.keys():
        orig_file = f"captured/original/{name}.jpg"
        if not os.path.exists(orig_file):
            continue
        img = cv2.imread(orig_file)
        if img is None:
            continue
        resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        png_path = f"captured/images_256/{name}.png"
        cv2.imwrite(png_path, resized)
        print(f"Generated {png_path} (256x256)")

if __name__ == "__main__":
    setup_all_captured()
