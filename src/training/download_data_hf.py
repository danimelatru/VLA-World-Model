import os
from huggingface_hub import hf_hub_download
import shutil

# --- CONFIGURATION ---
REPO_ID = "amandlek/robomimic"
FILENAME = "v1.5/lift/ph/low_dim_v15.hdf5" # Official Low-Dim Dataset (Lift task)
LOCAL_DIR = "./data"
FINAL_FILENAME = "lift_ph.hdf5"
def main():
    print(f"Starting download from Hugging Face: {REPO_ID}")
    
    os.makedirs(LOCAL_DIR, exist_ok=True)
    
    try:
        # 1. Download the file using the official Hub API
        # This handles caching and secure connection automatically
        downloaded_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            repo_type="dataset",
            cache_dir=os.path.join(LOCAL_DIR, "cache") # Temporary cache
        )
        
        print(f"Downloaded to cache: {downloaded_path}")
        
        # 2. Move and Rename to your data folder
        destination = os.path.join(LOCAL_DIR, FINAL_FILENAME)
        shutil.copy(downloaded_path, destination)
        
        print(f"File saved and renamed to: {destination}")
        print("You are ready to run preprocessing!")
        
        # Optional: Clean up cache if space is tight
        # shutil.rmtree(os.path.join(LOCAL_DIR, "cache"))

    except Exception as e:
        print(f"Error downloading: {e}")
        print("Tip: Check your internet connection or Hugging Face status.")

if __name__ == "__main__":
    main()