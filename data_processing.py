# data_processing.py
import cv2
import os
import numpy as np
from descriptor import glcm, bitdesc

def extract_features(image_path, descriptor_func):
    print(f"Reading image from: {image_path}")  # Debugging line
    if not isinstance(image_path, str):
        print(f"Invalid image path type: {type(image_path)}. Expected str.")
        return None
    
    if not os.path.isfile(image_path):
        print(f"File does not exist: {image_path}")
        return None
    
    img = cv2.imread(image_path, 0)
    if img is None:
        print(f"Failed to read image from: {image_path}")
        return None
    
    print(f"Image type: {type(img)}, Image shape: {img.shape}")  # Debugging line
    features = descriptor_func(img)
    return features

def process_datasets(root_folder, descriptor_func, output_file):
    all_features = [] # List to store all features and metadata
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                # Construct relative path
                image_rel_path = os.path.join(root, file)
                print(f"Processing file: {image_rel_path}")  # Debugging line
                
                if not os.path.isfile(image_rel_path):
                    print(f"File does not exist: {image_rel_path}")  # Debugging line
                    continue
                
                relative_path = os.path.relpath(image_rel_path, root_folder)
                folder_name = os.path.basename(os.path.dirname(image_rel_path))
                
                # Extract features
                features = extract_features(image_rel_path, descriptor_func)
                if features is not None:
                    features = features + [folder_name, relative_path]
                    all_features.append(features)
    
    print(f"Extracted features: {all_features}")
    signatures = np.array(all_features, dtype=object)
    np.save(output_file, signatures)
    print(f'Successfully stored in {output_file}!')

# Process datasets for GLCM
process_datasets('./dataset', glcm, 'glcm_signatures.npy')

# Process datasets for Bitdesc
process_datasets('./dataset', bitdesc, 'bitdesc_signatures.npy')


