import pandas as pd
import numpy as np
import os
import shutil

import h5py
from scipy.spatial import cKDTree

import random
import h5py
import matplotlib.pyplot as plt

import joblib
from sklearn.preprocessing import FunctionTransformer
from functools import partial
from tqdm import tqdm
import warnings

### Filter outliers ###

# Load the FEM CSV file into a DataFrame
df_FEA = pd.read_csv('Your directory/FEM_data.csv')
df_FEA = df_FEA.drop(['Unnamed: 0'], axis=1)

# Drop message
df_FEA = df_FEA.drop(['message'], axis=1)

# Load the mesh CSV file into a DataFrame
df_mesh = pd.read_csv('Your directory/mesh_data.csv')
df_mesh = df_mesh.drop(['Unnamed: 0'], axis=1)

# Merge dataframes
df_FEA_merged = pd.merge(
    df_FEA,
    df_mesh[['file', 'Vertex number', 'Boundary box x (mm)', 'Boundary box y (mm)', 'Boundary box z (mm)']],  # 필요한 컬럼만 선택
    on='file',
    how='left'
)

# Convert inf to nan
df_FEA_cleaned = df_FEA_merged.replace([np.inf, -np.inf], np.nan)

# Remove rows with nan
df_FEA_cleaned = df_FEA_cleaned.dropna()

def copy_variant_files(file_number, source_dir, target_dir):

    file_str = f"{file_number:08d}"

    files_to_copy = [
        f"{file_str}.xdmf",
        f"{file_str}.h5"
    ]

    for file in files_to_copy:
        src = os.path.join(source_dir, file)
        dst = os.path.join(target_dir, file)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Copied {src} to {dst}")
        else:
            print(f"File not found: {src}")

source_dir = "Your directory/Deformed_ABC/FEM"
target_dir = "Your directory/Deformed_ABC/1_Outliers_Filtered"

filtered_data = df_FEA_cleaned[(df_FEA_cleaned['status'] == 'success') & 
                          (df_FEA_cleaned['connected mesh number'] == 1) &
                          (df_FEA_cleaned['max displacement (mm)'] < 1e2) &
                          (df_FEA_cleaned['max displacement (mm)'] > 1e-4)
                          ]

# Filter fine data
for file_num in filtered_data['file']:
    copy_variant_files(file_num, source_dir, target_dir)

### Filter non-linear cases ###

# Checking Non-linearity
def check_geometric_nonlinearity(verts, disp, threshold=0.05):
    # characteristic length
    L = np.max(verts, axis=0) - np.min(verts, axis=0)
    L_char = np.linalg.norm(L) # characteristic lenght = diagnal length in bounding box
    
    # displacement magnitude
    disp_mag = np.linalg.norm(disp, axis=1)
    max_disp = disp_mag.max()
    
    disp_ratio = max_disp / L_char
    status = "Possible geometric nonlinearity" if disp_ratio > threshold else "Linear assumption reasonable"
    return max_disp, L_char, disp_ratio, status

def estimate_strain(verts, disp, k=8):
    tree = cKDTree(verts)
    local_strains = []

    for i, v in enumerate(verts):
        d, idx = tree.query(v, k=k+1)
        neighbors = idx[1:]
        for j in neighbors:
            du = disp[j] - disp[i]
            dx = verts[j] - verts[i]
            strain = np.linalg.norm(du) / (np.linalg.norm(dx) + 1e-12)
            local_strains.append(strain)

    max_strain = np.max(local_strains)
    mean_strain = np.mean(local_strains)
    
    return max_strain, mean_strain

# Check the whole file and copy only linear cases.
def copy_linear_results(src_folder, dst_folder, disp_threshold=0.05, strain_threshold=0.01):
    os.makedirs(dst_folder, exist_ok=True)
    files = sorted([f for f in os.listdir(src_folder) if f.endswith(".h5")])

    n_total = len(files)
    n_linear = 0
    n_nonlinear = 0

    for f in files:
        h5_path = os.path.join(src_folder, f)
        base_name = os.path.splitext(f)[0]
        xdmf_path = os.path.join(src_folder, base_name + ".xdmf")

        try:
            with h5py.File(h5_path, "r") as h5f:
                verts = h5f["/Mesh/mesh/geometry"][:]
                disp = h5f["/Function/Displacement (mm)/0"][:]
        except Exception as e:
            print(f"Failed to read {f}: {e}")
            continue

        _, _, disp_ratio, _ = check_geometric_nonlinearity(verts, disp, threshold=disp_threshold)
        max_strain, _ = estimate_strain(verts, disp)

        if disp_ratio <= disp_threshold and max_strain <= strain_threshold:
            # Copy if the case is linear
            shutil.copy2(h5_path, os.path.join(dst_folder, f))
            if os.path.exists(xdmf_path):
                shutil.copy2(xdmf_path, os.path.join(dst_folder, base_name + ".xdmf"))
            n_linear += 1
            print(f"{f} copied to {dst_folder} (Linear)")
        else:
            n_nonlinear += 1
            print(f"{f} skipped (Non-linear: disp_ratio={disp_ratio:.3f}, strain={max_strain:.3f})")

    # Summarize the results
    print("\n===== SUMMARY =====")
    print(f"Total files checked: {n_total}")
    print(f"Linear results copied: {n_linear}")
    print(f"Non-linear results skipped: {n_nonlinear}")
    print(f"Linear ratio: {n_linear / n_total * 100:.1f}%")

src_folder = "Your directory/Deformed_ABC/1_Outliers_Filtered"
dst_folder = "Your directory/Deformed_ABC/2_Nonlinear_Filtered"
copy_linear_results(src_folder, dst_folder, disp_threshold=0.05, strain_threshold=0.01)

### Scale data ###

# Load the CSV file into a DataFrame
df_material = pd.read_csv('Your directory/FEM_data.csv')
df_material = df_material[["file", "Young's moduli [GPa]", "Poisson's ratios"]]

# Split data to train and test
source_dir = "Your directory/Deformed_ABC/2_Nonlinear_Filtered"
train_dir  = "Your directory/Deformed_ABC/3_Scaled/data-train"
test_dir   = "Your directory/Deformed_ABC/3_Scaled/data-test"

random_seed = 0

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Collect all base name
all_basenames = sorted([
    f.replace(".xdmf", "") for f in os.listdir(source_dir)
    if f.endswith(".xdmf")
])

random.seed(random_seed)
random.shuffle(all_basenames)

n_total = len(all_basenames)
n_train = int(0.8 * n_total)

train_names = all_basenames[:n_train]
test_names  = all_basenames[n_train:]

verts_all = []
disp_all = []
load_all = []
dirichlet_all = []
sdf_all = []
normal_vecs_all = []
material_all = []
youngs_modulus_all = []
poisson_ratio_all = []
shear_modulus_all = []

def collect_statistics_from_h5(basename, src_dir):
    try:
        with h5py.File(os.path.join(src_dir, basename + ".h5"), "r") as f:
            # node coordinates
            verts = f["/Mesh/mesh/geometry"][:]                      # (N, 3)
            
            # Displacement
            disp = f["/Function/Displacement (mm)/0"][:]             # (N, 3)
            
            # Nodal surface load
            load = f["/Function/Nodal surface load (N)/0"][:]          # (N, 3)
            
            # Dirichlet condition
            dirichlet = f["/Function/Dirichlet/0"][:]                 # (N, 1)
            
            # SDF
            sdf = f["/Function/SDF/0"][:]                             # (N, 1)
            
            # Normal Vector
            normal_vecs = f["/Function/Normal Vector/0"][:]           # (N, 3)

        num_id = int(basename[:8])
        material_data = df_material.loc[df_material["file"] == num_id]

        if material_data.empty:
            print("No material data")
            return

        material_data = material_data.drop(columns="file").to_numpy(dtype=np.float64).squeeze()

        youngs_modulus = material_data[0]
        poisson_ratio = material_data[1]

        # Flatten and store
        disp_all.append(disp)
        load_all.append(load)
        
        youngs_modulus_all.append(youngs_modulus)
        poisson_ratio_all.append(poisson_ratio)

        verts_all.append(verts)
        dirichlet_all.append(dirichlet)
        sdf_all.append(sdf)
        normal_vecs_all.append(normal_vecs)

    except:
        pass

for name in train_names: # Use only train for scaling.
    collect_statistics_from_h5(name, source_dir)

# Material related boundary condtions and solution
# Divide norm and direction -> Log the norm -> Standard scaler -> Shift -> Combine it back with the direction
disp_all_np = np.vstack(disp_all)
load_all_np = np.vstack(load_all)

disp_all_magnitude_np = np.linalg.norm(disp_all_np, axis=1)
load_all_magnitude_np = np.linalg.norm(load_all_np, axis=1)

youngs_modulus_all_np = np.vstack(youngs_modulus_all)

youngs_modulus_all_magnitude_np = np.linalg.norm(youngs_modulus_all_np, axis=1)

# No scaling
dirichlet_all_np = np.vstack(dirichlet_all) # 0 or 1
poisson_ratio_all_np = np.vstack(poisson_ratio_all) # 0.01 ~ 0.5
normal_vecs_all_np = np.vstack(normal_vecs_all)

## Saving scalers

def log_transform_displacement(x, epsilon_disp = 1e-16, scale_factor = 1e4):
    x = np.asarray(x, dtype=np.float64)
    
    x = x * scale_factor
    x = x + epsilon_disp
    x = np.log(x)
    
    return x

def log_inverse_displacement(y, epsilon_disp = 1e-16, scale_factor = 1e4):
    y = np.asarray(y, dtype=np.float64)
    
    y = np.exp(y)
    y = y - epsilon_disp
    y = y / scale_factor
    
    return y

log_scaler_disp = FunctionTransformer(func=log_transform_displacement,
                                      inverse_func=log_inverse_displacement,
                                      check_inverse=False)

joblib.dump(log_scaler_disp, "Your directory/Deformed_ABC/3_Scaled/Scalers/log_disp_scaler.pkl")

def shift_generic(x, shift_value):
    return x - shift_value

def inverse_shift_generic(x, shift_value):
    return x + shift_value

shift_value_display = float(np.min(log_scaler_disp.transform(disp_all_magnitude_np.reshape(-1, 1))))

shift_scaler_disp = FunctionTransformer(
    func=partial(shift_generic, shift_value=shift_value_display),
    inverse_func=partial(inverse_shift_generic, shift_value=shift_value_display),
    check_inverse=False
)

import joblib
joblib.dump(shift_scaler_disp, "Your directory/Deformed_ABC/3_Scaled/Scalers/shift_disp_scaler.pkl")

# Load

def log_transform_load(x, epsilon_load = 1e-1):
    x = np.asarray(x, dtype=np.float32)
    return np.log(x + epsilon_load)

def log_inverse_load(y, epsilon_load = 1e-1):
    y = np.asarray(y, dtype=np.float32)
    return np.exp(y) - epsilon_load

log_scaler_load = FunctionTransformer(func=log_transform_load,
                                      inverse_func=log_inverse_load,
                                      check_inverse=False)

joblib.dump(log_scaler_load, "Your directory/Deformed_ABC/3_Scaled/Scalers/log_load_scaler.pkl")

shift_value_load = float(np.min(log_scaler_load.transform(load_all_magnitude_np.reshape(-1, 1))))

shift_scaler_load = FunctionTransformer(
    func=partial(shift_generic, shift_value=shift_value_load),
    inverse_func=partial(inverse_shift_generic, shift_value=shift_value_load),
    check_inverse=False
)

joblib.dump(shift_scaler_load, "Your directory/Deformed_ABC/3_Scaled/Scalers/shift_load_scaler.pkl")

# Material

def log_transform_material(x,epsilon_material = 1e-6):
    x = np.asarray(x, dtype=np.float64)
    return np.log(x + epsilon_material)

def log_inverse_material(y,epsilon_material = 1e-6):
    y = np.asarray(y, dtype=np.float64)
    return np.exp(y) - epsilon_material

log_scaler_material = FunctionTransformer(
    func=log_transform_material, 
    inverse_func=log_inverse_material,
    check_inverse=False)

joblib.dump(log_scaler_material, "Your directory/Deformed_ABC/3_Scaled/Scalers/log_youngs_scaler.pkl")

shift_value_youngs_modulus = float(np.min(log_scaler_material.transform(youngs_modulus_all_magnitude_np.reshape(-1, 1))))

# Generate pickle-safe FunctionTransformer
shift_scaler_youngs_modulus = FunctionTransformer(
    func=partial(shift_generic, shift_value=shift_value_youngs_modulus),
    inverse_func=partial(inverse_shift_generic, shift_value=shift_value_youngs_modulus),
    check_inverse=False
)

joblib.dump(shift_scaler_youngs_modulus, "Your directory/Deformed_ABC/3_Scaled/Scalers/shift_youngs_scaler.pkl")

# Load scalers and scale the data

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.preprocessing._function_transformer")

log_scaler_disp = joblib.load("Your directory/Deformed_ABC/3_Scaled/Scalers/log_disp_scaler.pkl")
shift_scaler_disp = joblib.load("Your directory/Deformed_ABC/3_Scaled/Scalers/shift_disp_scaler.pkl")
log_scaler_load = joblib.load("Your directory/Deformed_ABC/3_Scaled/Scalers/log_load_scaler.pkl")
shift_scaler_load = joblib.load("Your directory/Deformed_ABC/3_Scaled/Scalers/shift_load_scaler.pkl")
log_scaler_material = joblib.load("Your directory/Deformed_ABC/3_Scaled/Scalers/log_youngs_scaler.pkl")
shift_scaler_youngs_modulus = joblib.load("Your directory/Deformed_ABC/3_Scaled/Scalers/shift_youngs_scaler.pkl")

count_success = 0
count_fail = 0
fail_list = []

def scale_vector(vector, log_scaler, shift_scaler):

    vec_norm = np.linalg.norm(vector, axis=1)  # (N,)
    scaled_vec = np.zeros_like(vector)         

    # Select non zero vector
    nonzero_mask = vec_norm > 0
    if np.any(nonzero_mask):
        vec_dir = np.zeros_like(vector)
        vec_dir[nonzero_mask] = vector[nonzero_mask] / vec_norm[nonzero_mask, np.newaxis]

        vec_norm_log = log_scaler.transform(vec_norm[nonzero_mask].reshape(-1, 1)).flatten()
        vec_norm_shifted = shift_scaler.transform(vec_norm_log.reshape(-1, 1)).flatten()

        # Combine with vec_dir
        scaled_vec[nonzero_mask] = vec_dir[nonzero_mask] * vec_norm_shifted[:, np.newaxis]

    return scaled_vec

def inverse_scale_vector(scaled_vector, log_scaler, shift_scaler):
    vec_norm_shifted = np.linalg.norm(scaled_vector, axis=1)  # (N,)
    original_vector = np.zeros_like(scaled_vector)

    # Select non zero vector
    nonzero_mask = vec_norm_shifted > 0
    if np.any(nonzero_mask):
        vec_dir = np.zeros_like(scaled_vector)
        vec_dir[nonzero_mask] = scaled_vector[nonzero_mask] / vec_norm_shifted[nonzero_mask, np.newaxis]

        vec_norm_scaled = shift_scaler.inverse_transform(vec_norm_shifted[nonzero_mask].reshape(-1, 1)).flatten()
        vec_norm = log_scaler.inverse_transform(vec_norm_scaled.reshape(-1, 1)).flatten()

        original_vector[nonzero_mask] = vec_dir[nonzero_mask] * vec_norm[:, np.newaxis]

    return original_vector

def contains_nan_or_inf(*arrays):
    return any(np.isnan(arr).any() or np.isinf(arr).any() for arr in arrays)

def convert_to_npz_from_h5_safe(basename, src_dir, dst_dir):
    global count_success, count_fail, fail_list
    try:
        # Load raw data
        h5_path = os.path.join(src_dir, basename + ".h5")
        with h5py.File(h5_path, "r") as f:
            verts = f["/Mesh/mesh/geometry"][:]
            elems = f["/Mesh/mesh/topology"][:]
            disp = f["/Function/Displacement (mm)/0"][:]
            load = f["/Function/Nodal surface load (N)/0"][:]
            dirichlet = f["/Function/Dirichlet/0"][:]
            sdf = f["/Function/SDF/0"][:]
            normal_vecs = f["/Function/Normal Vector/0"][:]

        # Material data
        num_id = int(basename[:8])
        row = df_material.loc[df_material["file"] == num_id]
        if row.empty:
            raise ValueError(f"No material data (file id={num_id})")
        material = row.iloc[0].drop(labels="file").to_numpy(dtype=np.float32)
        youngs_modulus = np.tile(material[0], (len(verts), 1))
        poisson_ratio = np.tile(material[1], (len(verts), 1))

        # Checking raw data on NaN/Inf
        if contains_nan_or_inf(verts, disp, load, sdf, normal_vecs, dirichlet, youngs_modulus, poisson_ratio):
            raise ValueError("Raw data contains NaN/Inf")

        # Only subtract mean
        mean_only_scaler = FunctionTransformer(
            func=lambda x: x - np.mean(x, axis=0),
            inverse_func=lambda x: x + np.mean(x, axis=0))

        mean_only_scaler.fit(verts)
        verts_scaled = mean_only_scaler.transform(verts)
        
        # Scaling material related data
        disp_scaled = scale_vector(disp, log_scaler_disp, shift_scaler_disp)
        load_scaled = scale_vector(load, log_scaler_load, shift_scaler_load)
        youngs_scaled = scale_vector(youngs_modulus, log_scaler_material, shift_scaler_youngs_modulus)

        # Checking nan/inf in scaled data
        if contains_nan_or_inf(verts_scaled):
            raise ValueError("verts_scaled contains NaN/Inf")
        if contains_nan_or_inf(disp_scaled):
            raise ValueError("disp_scaled contains NaN/Inf")
        if contains_nan_or_inf(load_scaled):
            raise ValueError("load_scaled contains NaN/Inf")
        if contains_nan_or_inf(youngs_scaled):
            raise ValueError("youngs_scaled contains NaN/Inf")

        # Save
        out_path = os.path.join(dst_dir, basename + ".npz")
        np.savez(
            out_path,
            verts=verts_scaled,
            verts_mean = np.mean(verts, axis=0),
            elems=elems,
            norm=normal_vecs,
            disp=disp_scaled.astype(np.float32),
            load=load_scaled.astype(np.float32),
            dirichlet=dirichlet.squeeze().astype(np.float32),
            sdf=sdf.squeeze().astype(np.float32), # [[0],[0],[1]] -> [0,0,1] 
            sdf_mean = np.mean(sdf, axis=0),
            youngs=youngs_scaled.astype(np.float32),
            poisson=poisson_ratio.astype(np.float32)
        )

        count_success += 1

    except Exception as e:
        print(f"[SKIPPED] {basename}: {e}")
        count_fail += 1
        fail_list.append(basename)

print(f"\n Total sample: {n_total} → {n_train} train / {n_total - n_train} test\n")

# Scale train data
for name in tqdm(train_names, desc="Converting train set"):
    convert_to_npz_from_h5_safe(name, source_dir, train_dir)

# Scale test data
for name in tqdm(test_names, desc="Converting test set"):
    convert_to_npz_from_h5_safe(name, source_dir, test_dir)

# Summary
print("\n Summary")
print(f"Successfully converted .npz files: {count_success}")
print(f"Failed files: {count_fail}")

if fail_list:
    print(f"List of failed files: {fail_list[:10]}")