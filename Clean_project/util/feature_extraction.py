import os
import numpy as np
import torch


def extract_feature_vectors(folder_path, save_path, model, image_processor):
    """
    Extract and save feature vectors for all images in a folder, using the same
    names as the original images.

    Args:
        folder_path (str): Path to the folder containing images.
        save_path (str): Path to save the feature vectors.
        model (torch.nn.Module): The loaded ft_ResNet50 model.
        image_processor (function): Function to preprocess images.

    Returns:
        tuple: A tuple containing a list of feature vectors and a list of image paths.
    """
    ensure_directory_exists(save_path)
    feature_vectors = []
    image_paths = []

    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        preprocessed_image = image_processor(image_path)

        with torch.no_grad():
            output = model(preprocessed_image)
            # Flatten the output if necessary
            feature_vector = output.view(output.size(0), -1)

        feature_vectors.append(feature_vector.cpu().numpy())
        image_paths.append(image_path)

        npy_filename = os.path.splitext(image_name)[0] + '.npy'
        npy_path = os.path.join(save_path, npy_filename)
        np.save(npy_path, feature_vector.cpu().numpy())

    return feature_vectors, image_paths


def ensure_directory_exists(directory_path):
    """
    Ensure that a directory exists, creating it if necessary.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


# ... rest of your code ...

def modify_and_save_name(image_name, save_path, feature_vector):
    """
    Modify the image file name and save the feature vector as a .npy file.

    Args:
        image_name (str): Original name of the image file.
        save_path (str): Directory to save the .npy file.
        feature_vector (torch.Tensor): Feature vector to be saved.

    Returns:
        tuple: Modified image name and path to the saved .npy file.
    """
    base_name, ext = os.path.splitext(image_name)
    tracking_id, frameno = base_name.split('_')
    modified_name = f"{tracking_id}_4_{frameno}{ext}"

    npy_filename = modified_name.replace(ext, '.npy')
    npy_path = os.path.join(save_path, npy_filename)
    np.save(npy_path, feature_vector.cpu().numpy())

    return modified_name, npy_path
