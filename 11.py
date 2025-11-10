import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from sklearn.model_selection import train_test_split
import random
import os
import segyio
import lasio

# Suppress TensorFlow logging for a cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- FILE PATHS (UPDATE THESE) --------------------------------
# IMPORTANT: Ensure these files are downloaded and placed in the directory 
# where this script is executed, or update the paths below.
SEGY_FILE = "f3_dataset.sgy" 
LAS_FILE = "F02-1_logs.las" 
# --------------------------------------------------------------

# --- 1. Load Real Seismic Data (Cube) ---
def load_seismic_cube(segy_file_path):
    """Loads a SEG-Y file into a 3D NumPy cube and returns index maps."""
    print("Loading seismic cube...")
    try:
        with segyio.open(segy_file_path, "r", ignore_geometry=True) as f:
            f.mmap()
            traces = np.array([f.trace[i] for i in range(f.tracecount)])
            ilines = f.attributes(segyio.TraceField.INLINE_3D)[:]
            xlines = f.attributes(segyio.TraceField.CROSSLINE_3D)[:]

        unique_ilines = np.unique(ilines)
        unique_xlines = np.unique(xlines)
        n_samples = traces.shape[1]

        # Initialize cube with NaNs
        cube = np.full((len(unique_ilines), len(unique_xlines), n_samples), np.nan, dtype=np.float32)

        # Create index mappings
        iline_to_idx = {iline: i for i, iline in enumerate(unique_ilines)}
        xline_to_idx = {xline: i for i, xline in enumerate(unique_xlines)}

        # Fill cube
        for t in range(traces.shape[0]):
            i = iline_to_idx.get(ilines[t])
            j = xline_to_idx.get(xlines[t])
            if i is not None and j is not None:
                cube[i, j, :] = traces[t]
        
        # Remove NaN traces (usually in edges or missing data)
        cube = np.nan_to_num(cube) 

        # Normalize the entire volume
        cube = (cube - np.mean(cube)) / (np.std(cube) + 1e-6)

        print(f"Cube shape: {cube.shape}")
        return cube, iline_to_idx, xline_to_idx, unique_ilines, unique_xlines
        
    except FileNotFoundError:
        print(f"Error: Seismic file not found at {segy_file_path}. Please check the path and ensure the file is present.")
        return None, None, None, None, None
    except Exception as e:
        print(f"An error occurred loading SEG-Y: {e}")
        return None, None, None, None, None

# --- 2. Load Real Well Log Data (and generate a simplified label) ---
def load_and_label_well(las_file_path, cube_shape):
    """
    Loads LAS file and creates a simplified binary label volume based on GR log.
    Low GR (potential reservoir) is labeled '1'.
    """
    print("Loading well log and generating synthetic label...")
    try:
        las = lasio.read(las_file_path)
        
        # Simple GR Labeling Heuristic: Low GR is clean/potential reservoir (Label 1)
        GR_THRESHOLD = 40 
        
        # 1D label (1: Low GR/Clean, 0: High GR/Shale)
        label_1d = (las['GR'] < GR_THRESHOLD).astype(np.int32)
        
        # Resample the 1D log data to match the cube's time/sample domain (crude linear tie)
        n_samples = cube_shape[2]
        resampled_label_1d = np.interp(
            np.linspace(0, len(label_1d) - 1, n_samples),
            np.arange(len(label_1d)),
            label_1d
        ).astype(np.int32)
        
        # Well assumed at the center of the seismic cube for simple indexing
        well_iline_idx = cube_shape[0] // 2 
        well_xline_idx = cube_shape[1] // 2 
        
        # Create a 3D label volume, mostly zeros, with the well-tied label at the well location
        label_volume = np.zeros(cube_shape, dtype=np.int32)
        label_volume[well_iline_idx, well_xline_idx, :] = resampled_label_1d
        
        print(f"Synthetic Label Volume generated. Well location (Inline, Xline): ({well_iline_idx}, {well_xline_idx})")
        return label_volume, np.argwhere(label_volume == 1)
    
    except FileNotFoundError:
        print(f"Error: LAS file not found at {las_file_path}. Please check the path and ensure the file is present.")
        return None, None
    except Exception as e:
        print(f"An error occurred loading LAS or processing logs: {e}")
        return None, None

# --- 3. FIXED DATA EXTRACTION (Ensures balanced classes) ---
def extract_cubes_and_labels_real(volume, label_volume, positive_indices, cube_size, n_cubes=500, positive_target=50):
    """
    Extracts 3D sub-cubes (X) and labels (Y). 
    It forces the extraction of 'positive_target' cubes that overlap the labeled well path 
    to ensure the positive class is represented.
    """
    d1, d2, d3 = volume.shape
    X_cubes = []
    y_labels = [] 
    
    
    # 1. Force extraction of POSITIVE samples
    if positive_indices.size > 0:
        n_pos_to_extract = min(positive_target, len(positive_indices))
        
        # Randomly select N locations near the well path
        for _ in range(n_pos_to_extract):
            # Pick a random index along the well path (i_well, j_well, k_well)
            idx_1d = random.choice(positive_indices)
            i_well, j_well, k_well = idx_1d[0], idx_1d[1], idx_1d[2]
            
            # Define a small extraction window centered around this point
            i_start = random.randint(max(0, i_well - cube_size[0] + 1), min(d1 - cube_size[0], i_well))
            j_start = random.randint(max(0, j_well - cube_size[1] + 1), min(d2 - cube_size[1], j_well))
            k_start = random.randint(max(0, k_well - cube_size[2] + 1), min(d3 - cube_size[2], k_well))
            
            sub_cube = volume[i_start:i_start + cube_size[0], j_start:j_start + cube_size[1], k_start:k_start + cube_size[2]]
            
            X_cubes.append(sub_cube)
            y_labels.append(1) # Guaranteed positive label
    else:
        print("Warning: No positive labels found in the well path. All samples will be negative.")


    # 2. Extract the remaining NEGATIVE (random) samples
    
    while len(X_cubes) < n_cubes:
        i = random.randint(0, d1 - cube_size[0])
        j = random.randint(0, d2 - cube_size[1])
        k = random.randint(0, d3 - cube_size[2])
        
        sub_cube = volume[i:i + cube_size[0], j:j + cube_size[1], k:k + cube_size[2]]
        
        # Check label (1 if ANY voxel is '1', 0 otherwise)
        sub_label = label_volume[i:i + cube_size[0], j:j + cube_size[1], k:k + cube_size[2]]
        label = 1 if np.any(sub_label == 1) else 0
        
        # Only add if it's a negative sample, or if we need more cubes
        if label == 0:
            X_cubes.append(sub_cube)
            y_labels.append(label)

    # Final preparation
    X_cubes = np.array(X_cubes[:n_cubes], dtype=np.float32)
    y_labels = np.array(y_labels[:n_cubes], dtype=np.int32)
    X_cubes = X_cubes[..., np.newaxis]
    
    return X_cubes, y_labels

# --- MAIN EXECUTION START ---

CUBE_SIZE = (8, 8, 8)
N_TOTAL_CUBES = 500
N_POSITIVE_TARGET = 50 # Force 50 cubes that overlap the well label

# Load Seismic Data
cube_ds, _, _, _, _ = load_seismic_cube(SEGY_FILE)

if cube_ds is None:
    exit()

# Generate Synthetic Label Volume and get indices of positive labels
label_volume, positive_indices = load_and_label_well(LAS_FILE, cube_ds.shape)

if label_volume is None:
    exit()

# Extract cubes for training using the fixed function
X_cubes, y_labels = extract_cubes_and_labels_real(
    cube_ds, label_volume, positive_indices, 
    cube_size=CUBE_SIZE, 
    n_cubes=N_TOTAL_CUBES, 
    positive_target=N_POSITIVE_TARGET
)

print(f"Total cubes (X) shape: {X_cubes.shape}")
print(f"Total labels (Y) shape: {y_labels.shape}")
print(f"Hydrocarbon/Reservoir ratio (1s): {np.sum(y_labels) / len(y_labels):.2f}")

# Split data (stratify now works because we have enough positive samples)
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X_cubes, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
except ValueError as e:
    print(f"\nFATAL ERROR: Stratified split failed. Ensure you have at least 2 positive samples.")
    print(f"Error details: {e}")
    exit()


# -----------------------------
# 4. Define and Train 3D CNN CLASSIFIER 
# -----------------------------
def small_3d_classifier(input_shape=(8, 8, 8, 1)):
    """3D Residual CNN model adapted for binary classification."""
    inputs = Input(shape=input_shape)
    
    # Initial Conv and Pooling
    x = layers.Conv3D(8, 3, padding='same', activation='relu')(inputs)
    y = layers.MaxPooling3D(2)(x) # Output size: (4, 4, 4, 8)
    
    # Residual Block 1
    x = layers.Conv3D(8, 3, padding='same', activation='relu')(y)
    x = layers.Conv3D(8, 3, padding='same')(x)
    x = layers.Add()([x, y])
    x = layers.Activation('relu')(x)
    
    # Residual Block 2 
    y_res2 = x # Save for final skip
    x = layers.Conv3D(16, 3, padding='same', activation='relu')(x)
    
    # NAMING THIS LAYER IS CRUCIAL FOR GRAD-CAM
    x = layers.Conv3D(16, 3, padding='same', activation='relu', name='last_conv_layer')(x)
    
    # Projection shortcut (1x1x1 Conv to match channels)
    y_res2 = layers.Conv3D(16, 1, padding='same')(y_res2)
    x = layers.Add()([x, y_res2])
    x = layers.Activation('relu')(x) # Output size: (4, 4, 4, 16)
    
    # Final Classification Head
    x = layers.GlobalAveragePooling3D()(x) # Collapse (4, 4, 4, 16) to (16)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(16, activation='relu')(x)
    
    # Single output unit with sigmoid for probability
    outputs = layers.Dense(1, activation='sigmoid', name='reservoir_probability')(x)
    
    model = models.Model(inputs, outputs)
    return model

# Initialize and compile the model
model = small_3d_classifier(X_train.shape[1:])
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

print("\nStarting supervised training on real data...")
history = model.fit(
    X_train, y_train, 
    epochs=15, 
    batch_size=8,
    validation_data=(X_test, y_test),
    verbose=0 
)

# Evaluate and Predict
print("\n--- Training Complete ---")
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: **{acc * 100:.2f}%**")
print(f"Test Loss: {loss:.4f}")

# -----------------------------
# 5. Grad-CAM Visualization
# -----------------------------

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=0):
    """Computes Grad-CAM heatmap for a 3D input."""
    # Build a model that returns the output of the last conv layer AND the model's output
    grad_model = models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, pred_index] 

    # Gradient of the output probability w.r.t. the last conv layer output
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Average the gradient over all spatial and batch dimensions (I, X, T)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2, 3))

    # Weighted activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10) 
    
    return heatmap.numpy()

print("\n--- Grad-CAM Visualization ---")
LAST_CONV_LAYER_NAME = 'last_conv_layer'

# Find a sample predicted as 'Reservoir' (label 1)
positive_samples = np.where(y_test == 1)[0]
if len(positive_samples) == 0:
    print("Could not find a positive sample in the test set for Grad-CAM visualization. Skipping plot.")
    exit()

# Use the first positive sample
positive_sample_idx = positive_samples[0]
target_cube = X_test[positive_sample_idx]
target_input = np.expand_dims(target_cube, axis=0) 

# Generate the 3D heatmap
heatmap_3d = make_gradcam_heatmap(target_input, model, LAST_CONV_LAYER_NAME)

# 1. Select the central TIME slice (axis=2) from the input cube
input_slice = target_cube[:, :, target_cube.shape[2] // 2, 0] 

# 2. Select the corresponding TIME slice from the 3D heatmap
heatmap_slice = heatmap_3d[:, :, heatmap_3d.shape[2] // 2] 

# Resize the heatmap slice back to the original cube size for overlay
heatmap_slice_resized = tf.image.resize(
    np.expand_dims(heatmap_slice, axis=[0, -1]), 
    (input_slice.shape[0], input_slice.shape[1]), 
    method=tf.image.ResizeMethod.BILINEAR
)
heatmap_slice_resized = tf.squeeze(heatmap_slice_resized).numpy()

# 3. Plot the result
fig, ax = plt.subplots(1, 2, figsize=(14, 7))

# Original Cube Slice
ax[0].imshow(input_slice, cmap='seismic', aspect='auto', interpolation='bilinear')
ax[0].set_title(f"Input Cube Slice (True Label: {y_test[positive_sample_idx]})", fontsize=14, fontweight='bold')
ax[0].set_xlabel("Local X-line Index (Cube)", fontsize=12)
ax[0].set_ylabel("Local Inline Index (Cube)", fontsize=12)

# Heatmap Overlay
ax[1].imshow(input_slice, cmap='seismic', aspect='auto', interpolation='bilinear')
# Overlay heatmap with transparency 
im = ax[1].imshow(heatmap_slice_resized, cmap='jet', alpha=0.6, aspect='auto', interpolation='bilinear') 
ax[1].set_title("Grad-CAM Activation Map Overlay (Reservoir)", fontsize=14, fontweight='bold')
ax[1].set_xlabel("Local X-line Index (Cube)", fontsize=12)
ax[1].set_ylabel("Local Inline Index (Cube)", fontsize=12)
fig.colorbar(im, ax=ax[1], label="Activation Intensity (0.0 to 1.0)")

plt.suptitle(f"3D CNN Classification on F3 Seismic Data (Cube Index: {positive_sample_idx})", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()