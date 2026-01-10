import h5py
import numpy as np
from tqdm import tqdm
import os

# --- CONFIGURATION ---
DATASET_PATH = "./data/lift_ph.hdf5"
OUTPUT_PATH = "./data/lift_ph_embeddings.hdf5"

def main():
    print(f"⚡ Processing Low-Dim Data from: {DATASET_PATH}")
    
    if not os.path.exists(DATASET_PATH):
         raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Run download_data_hf.py first.")

    f_in = h5py.File(DATASET_PATH, "r")
    demos = list(f_in["data"].keys())
    
    f_out = h5py.File(OUTPUT_PATH, "w")
    grp = f_out.create_group("data")

    print(f"Processing {len(demos)} trajectories...")

    for demo_key in tqdm(demos):
        # En datasets Low-Dim, la info está en "obs/robot0_eef_pos", "obs/object", etc.
        # Robomimic concatena esto en "obs/proprio" o similar, pero aquí lo haremos simple.
        
        # Extraemos el estado del robot (proprioception) como "fake embedding"
        # Shape: (T, D) donde D es la dimensión del estado (ej. 10 o 20)
        # Esto permite que tu pipeline fluya igual que si fuera un embedding de imagen.
        
        # 1. Leer estado (obs)
        # Intentamos leer claves comunes en datasets PH
        obs_grp = f_in[f"data/{demo_key}/obs"]
        
        # Concatenamos posición del robot y estado del objeto para hacer un vector rico
        robot_pos = obs_grp["robot0_eef_pos"][:] # (T, 3)
        robot_quat = obs_grp["robot0_eef_quat"][:] # (T, 4)
        gripper_qpos = obs_grp["robot0_gripper_qpos"][:] # (T, 2)
        object_pos = obs_grp["object"][:] # (T, 10) - info del objeto
        
        # Creamos nuestro "embedding" (vector de características)
        embedding = np.concatenate([robot_pos, robot_quat, gripper_qpos, object_pos], axis=1)
        
        # 2. Leer acciones
        actions = f_in[f"data/{demo_key}/actions"][:]
        
        # 3. Guardar
        demo_grp = grp.create_group(demo_key)
        demo_grp.create_dataset("obs_embedding", data=embedding) # Tu modelo creerá que esto vino de SigLIP
        demo_grp.create_dataset("actions", data=actions)
        
        if "num_samples" in f_in[f"data/{demo_key}"].attrs:
            demo_grp.attrs["num_samples"] = f_in[f"data/{demo_key}"].attrs["num_samples"]

    f_in.close()
    f_out.close()
    print(f"Success! Data ready at: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()