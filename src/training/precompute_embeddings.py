import h5py
import numpy as np
from tqdm import tqdm
import os
import torch
from transformers import CLIPProcessor, CLIPModel

# --- CONFIGURATION ---
DATASET_PATH = "./data/lift_ph.hdf5"
OUTPUT_PATH = "./data/lift_ph_embeddings.hdf5"
# Data Augmentation: Multiple synonyms for the same task
TEXT_INSTRUCTIONS = [
    "Lift the red cube",
    "Pick up the red cube",
    "Grasp the red object",
    "Raise the red block",
    "Lift the object vertically",
    "Elevate the red cube"
]

def main():
    print(f"Processing Low-Dim Data from: {DATASET_PATH}")
    
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Run download_data_hf.py first.")

    # Load CLIP model for text encoding
    print("Loading CLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Encode ALL text instructions
    print(f"Encoding {len(TEXT_INSTRUCTIONS)} text variants...")
    with torch.no_grad():
        text_inputs = clip_processor(text=TEXT_INSTRUCTIONS, return_tensors="pt", padding=True)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        text_features = clip_model.get_text_features(**text_inputs)
        # Shape: (N_prompts, 512)
        text_embeddings = text_features.cpu().numpy()
    
    print(f"Text embeddings shape: {text_embeddings.shape}")

    f_in = h5py.File(DATASET_PATH, "r")
    demos = list(f_in["data"].keys())
    
    f_out = h5py.File(OUTPUT_PATH, "w")
    grp = f_out.create_group("data")

    print(f"Processing {len(demos)} trajectories...")

    for demo_key in tqdm(demos):
        obs_grp = f_in[f"data/{demo_key}/obs"]
        
        # Build state embedding from robot proprioception
        robot_pos = obs_grp["robot0_eef_pos"][:]  # (T, 3)
        robot_quat = obs_grp["robot0_eef_quat"][:]  # (T, 4)
        gripper_qpos = obs_grp["robot0_gripper_qpos"][:]  # (T, 2)
        object_pos = obs_grp["object"][:]  # (T, 10)
        
        embedding = np.concatenate([robot_pos, robot_quat, gripper_qpos, object_pos], axis=1)
        
        actions = f_in[f"data/{demo_key}/actions"][:]
        
        # Save to output
        demo_grp = grp.create_group(demo_key)
        demo_grp.create_dataset("obs_embedding", data=embedding)
        demo_grp.create_dataset("actions", data=actions)
        # Save ALL variants so datasets/vla_dataset.py can sample randomly
        demo_grp.create_dataset("text_embeddings", data=text_embeddings)
        
        if "num_samples" in f_in[f"data/{demo_key}"].attrs:
            demo_grp.attrs["num_samples"] = f_in[f"data/{demo_key}"].attrs["num_samples"]

    f_in.close()
    f_out.close()
    print(f"Success! Data ready at: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()