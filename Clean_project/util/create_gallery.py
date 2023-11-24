import os
import numpy as np
from shutil import copy2
from sklearn.metrics.pairwise import cosine_similarity


def load_features(folder_path):
    """
    Load feature vectors and their corresponding filenames from .npy files in a given folder.
    """
    features = []
    filenames = []
    for file in sorted(os.listdir(folder_path)):
        if file.endswith('.npy'):
            feature = np.load(os.path.join(folder_path, file))
            features.append(feature)
            filenames.append(file.split('.')[0])  # Remove .npy extension
    return features, filenames


def compute_similarity_matrix(features_cam3, features_cam4):
    """
    Compute cosine similarity matrix between feature vectors of two sets of images.

    Args:
        features_cam3 (list): List of feature arrays from camera 3.
        features_cam4 (list): List of feature arrays from camera 4.

    Returns:
        np.ndarray: Cosine similarity matrix.
    """
    # Flatten the features to 2D if they are not
    features_cam3_flat = [f.reshape(-1) for f in features_cam3]
    features_cam4_flat = [f.reshape(-1) for f in features_cam4]

    return cosine_similarity(features_cam3_flat, features_cam4_flat)


def create_gallery(cam3_features_folder, cam3_images_folder, cam4_features_folder, cam4_images_folder, gallery_folder,
                   threshold=0.7):
    features_cam3, filenames_cam3 = load_features(cam3_features_folder)
    features_cam4, filenames_cam4 = load_features(cam4_features_folder)

    similarity_matrix = compute_similarity_matrix(features_cam3, features_cam4)
    existing_track_ids = set([filename.split('_')[0] for filename in filenames_cam4])
    next_unique_id = max(map(int, existing_track_ids)) + 1

    if not os.path.exists(gallery_folder):
        os.makedirs(gallery_folder)

    for idx, _ in enumerate(features_cam3):
        best_match_idx = np.argmax(similarity_matrix[idx])
        best_match_score = similarity_matrix[idx][best_match_idx]

        if best_match_score > threshold:
            matched_track_id = filenames_cam4[best_match_idx].split('_')[0]
        else:
            matched_track_id = str(next_unique_id)
            next_unique_id += 1

        source_image_path = os.path.join(cam3_images_folder, filenames_cam3[idx] + '.jpg')
        target_image_path = os.path.join(gallery_folder,
                                         f"{matched_track_id}_3_{filenames_cam3[idx].split('_')[2]}.jpg")
        copy2(source_image_path, target_image_path)

    # Copy all images from camera 4 to the gallery
    for filename in filenames_cam4:
        source_image_path = os.path.join(cam4_images_folder, filename + '.jpg')
        target_image_path = os.path.join(gallery_folder, filename + '.jpg')
        copy2(source_image_path, target_image_path)

    print("Gallery creation complete.")


# Example usage
create_gallery('path_to_cam3_features', 'path_to_cam3_images', 'path_to_cam4_features', 'path_to_cam4_images',
               'path_to_gallery')
