import os
from huggingface_hub import hf_hub_download
import shutil

# --- CONFIGURATION ---
# --- CONFIGURATION ---
REPO_ID = "amandlek/robomimic"
TASKS = {
    "lift": "v1.5/lift/ph/low_dim_v15.hdf5",
    "can": "v1.5/can/ph/low_dim_v15.hdf5",
    "square": "v1.5/square/ph/low_dim_v15.hdf5"
}
LOCAL_DIR = "./data"

def main():
    print(f"Starting multi-task download from: {REPO_ID}")
    
    os.makedirs(LOCAL_DIR, exist_ok=True)
    
    try:
        for task_name, filename in TASKS.items():
            print(f"⬇️ Downloading {task_name}...")
            
            # 1. Download
            downloaded_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type="dataset",
                cache_dir=os.path.join(LOCAL_DIR, "cache")
            )
            
            # 2. Rename/Move
            final_name = f"{task_name}_ph.hdf5"
            destination = os.path.join(LOCAL_DIR, final_name)
            shutil.copy(downloaded_path, destination)
            print(f"✅ Saved: {destination}")
            
        print("\nAll datasets ready for preprocessing!")

        
        
        # Optional: Clean up cache if space is tight
        # shutil.rmtree(os.path.join(LOCAL_DIR, "cache"))
        
        # Optional: Clean up cache if space is tight
        # shutil.rmtree(os.path.join(LOCAL_DIR, "cache"))

    except Exception as e:
        print(f"Error downloading: {e}")
        print("Tip: Check your internet connection or Hugging Face status.")

if __name__ == "__main__":
    main()