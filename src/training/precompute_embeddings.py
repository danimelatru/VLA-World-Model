import h5py
import numpy as np
from tqdm import tqdm
import os
import torch
from transformers import CLIPProcessor, CLIPModel

# --- CONFIGURATION ---
DATA_DIR = "./data"
OUTPUT_PATH = "./data/multi_task_embeddings.hdf5"

# Define tasks and their specific synonymous prompts
TASK_CONFIG = {
    "lift": {
        "filename": "lift_ph.hdf5",
        "prompts": [
            "Lift the red cube", "Pick up the red cube", "Grasp the red object", 
            "Raise the red block", "Lift the object vertically", "Elevate the red cube"
        ]
    },
    "can": {
        "filename": "can_ph.hdf5",
        "prompts": [
            "Pick up the coke can", "Grasp the soda can", "Lift the metal can",
            "Place the can in the bin", "Move the coke can", "Retrieve the soda can"
        ]
    },
    "square": {
        "filename": "square_ph.hdf5",
        "prompts": [
            "Push the square nut", "Insert the square nut onto the peg", "Assemble the nut",
            "Push the object", "Slide the square nut", "Fit the nut on the peg"
        ]
    }
}

def main():
    print(f"⚡ Processing Multi-Task Data into: {OUTPUT_PATH}")
    
    # Check inputs
    for task, cfg in TASK_CONFIG.items():
        path = os.path.join(DATA_DIR, cfg["filename"])
        if not os.path.exists(path):
             raise FileNotFoundError(f"Missing {task} dataset at {path}. Run download logic first.")

    # Load CLIP
    print("Loading CLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    f_out = h5py.File(OUTPUT_PATH, "w")
    grp_out = f_out.create_group("data")
    
    total_processed = 0

    for task_name, cfg in TASK_CONFIG.items():
        print(f"\n--- Processing Task: {task_name.upper()} ---")
        
        # 1. Encode Prompts for this task
        prompts = cfg["prompts"]
        print(f"Encoding {len(prompts)} prompts...")
        with torch.no_grad():
            text_inputs = clip_processor(text=prompts, return_tensors="pt", padding=True)
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            text_features = clip_model.get_text_features(**text_inputs)
            text_embeddings = text_features.cpu().numpy() # (N_prompts, 512)

        # 2. Process trajectories
        input_path = os.path.join(DATA_DIR, cfg["filename"])
        f_in = h5py.File(input_path, "r")
        demos = list(f_in["data"].keys())
        
        print(f"Merging {len(demos)} trajectories...")
        
        for demo_key in tqdm(demos):
            # Unique key: task_demo_0, task_demo_1...
            new_key = f"{task_name}_{demo_key}"
            
            obs_grp = f_in[f"data/{demo_key}/obs"]
            
            # State construction (Standard Robomimic Low-Dim keys)
            # Lift/Can/Square all share these basic keys
            robot_pos = obs_grp["robot0_eef_pos"][:]   # (T, 3)
            robot_quat = obs_grp["robot0_eef_quat"][:] # (T, 4)
            gripper_qpos = obs_grp["robot0_gripper_qpos"][:] # (T, 2)
            object_pos = obs_grp["object"][:]          # (T, X) - varies by task, but we concat
            
            # Note: Square task object dim might differ from Lift/Can.
            # For a proper World Model, state dim should be consistent OR padded.
            # Lift object: 10, Can object: 14, Square object: 14.
            # We will PAD object_pos to max_dim=14 to ensure consistency.
            
            curr_obj_dim = object_pos.shape[1]
            max_obj_dim = 14
            if curr_obj_dim < max_obj_dim:
                padding = np.zeros((object_pos.shape[0], max_obj_dim - curr_obj_dim))
                object_pos = np.concatenate([object_pos, padding], axis=1)
            
            embedding = np.concatenate([robot_pos, robot_quat, gripper_qpos, object_pos], axis=1)
            
            actions = f_in[f"data/{demo_key}/actions"][:]
            
            # Save
            demo_grp = grp_out.create_group(new_key)
            demo_grp.create_dataset("obs_embedding", data=embedding)
            demo_grp.create_dataset("actions", data=actions)
            demo_grp.create_dataset("text_embeddings", data=text_embeddings) # Task-specific embeddings
            
            # Metadata
            demo_grp.attrs["task"] = task_name
            if "num_samples" in f_in[f"data/{demo_key}"].attrs:
                demo_grp.attrs["num_samples"] = f_in[f"data/{demo_key}"].attrs["num_samples"]
            
            total_processed += 1

        f_in.close()

    f_out.close()
    print(f"\n✅ Success! Merged {total_processed} trajectories into {OUTPUT_PATH}")

if __name__ == "__main__":
    main()