import os
import sys
import time
import urllib.request
import zipfile

def download_file(url, dest_path):
    # If file exists and size matches expected, skip
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        if os.path.exists(dest_path) and os.path.getsize(dest_path) == total_size:
            print(f"File {dest_path} already completely downloaded ({total_size} bytes).")
            return
            
        print(f"Downloading {os.path.basename(dest_path)} ({total_size / (1024*1024):.1f} MB)...")
        chunk_size = 2 * 1024 * 1024  # 2 MB chunks
        downloaded = 0
        start_time = time.time()
        
        with open(dest_path, "wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                elapsed = time.time() - start_time
                speed = (downloaded / (1024*1024)) / max(elapsed, 0.001)
                percent = (downloaded / total_size * 100) if total_size > 0 else 0
                print(f"\rDownloaded {downloaded / (1024*1024):.1f} / {total_size / (1024*1024):.1f} MB ({percent:.1f}%) - {speed:.2f} MB/s", end="", flush=True)
        print("\nDownload complete.")

def extract_zip(zip_path, extract_dir):
    print(f"Extracting {zip_path} into {extract_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Extraction complete.")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    
    tr_url = "https://huggingface.co/datasets/chitradrishti/duts/resolve/main/DUTS-TR.zip"
    te_url = "https://huggingface.co/datasets/chitradrishti/duts/resolve/main/DUTS-TE.zip"
    
    tr_zip = os.path.join("data", "DUTS-TR.zip")
    te_zip = os.path.join("data", "DUTS-TE.zip")
    
    download_file(tr_url, tr_zip)
    download_file(te_url, te_zip)
    
    extract_zip(tr_zip, "data")
    extract_zip(te_zip, "data")
    print("All datasets ready!")
