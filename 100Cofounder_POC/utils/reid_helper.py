import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import os
from sklearn.metrics.pairwise import cosine_similarity
import json
import shutil
from models.ft_ResNet50.model import ft_net

# Global paths for the gallery folder, feature file, and filename mapping file
GALLERY_FOLDER_PATH = 'gallery/'
FEATURE_FILE_PATH = 'gallery_feature.npy'
FILENAME_MAPPING_PATH = 'gallery_mapping.json'


def load_model(model_path, class_num=751, device='cpu'):
    """
    Load and return the ft_ResNet50 model.

    Args:
        model_path (str): Path to the model's state dictionary.
        class_num (int): Number of classes in the model.
        device (str): The device to load the model on ('cpu' or 'cuda').

    Returns:
        torch.nn.Module: The loaded ft_ResNet50 model.
    """
    model = ft_net(class_num)
    state_dict = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def process_pil_image(pil_image):
    """
    Applies the necessary transformations to a PIL image for feature extraction.

    Parameters:
    pil_image (PIL.Image): The image to be processed.

    Returns:
    torch.Tensor: The transformed image as a tensor.
    """
    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    preprocessed_image = transform(pil_image)
    preprocessed_image = preprocessed_image.unsqueeze(0)
    return preprocessed_image


def extract_features_from_pil(pil_image, model):
    """
    Extracts feature vectors from a given PIL image using the specified model.

    Parameters:
    pil_image (PIL.Image): The image from which to extract features.
    model (torch.nn.Module): The model used for feature extraction.

    Returns:
    np.ndarray: The extracted feature vector.
    """
    preprocessed_image = process_pil_image(pil_image)
    with torch.no_grad():
        feature_vector = model(preprocessed_image)
    return feature_vector.cpu().numpy().flatten()


def initialize_gallery(gallery_folder_path, model, feature_file_path, filename_mapping_path):
    """
    Initializes the gallery by processing existing images, extracting features,
    and saving them along with the filenames.

    Parameters:
    gallery_folder_path (str): Path to the gallery folder.
    model (torch.nn.Module): The model used for feature extraction.
    feature_file_path (str): Path to the file where extracted features are saved.
    filename_mapping_path (str): Path to the file mapping filenames to features.
    """
    # Create gallery folder if it doesn't exist
    if not os.path.exists(gallery_folder_path):
        os.makedirs(gallery_folder_path)

    feature_vectors = []
    filenames = []

    for image_name in os.listdir(gallery_folder_path):
        if image_name.endswith('.jpg'):
            image_path = os.path.join(gallery_folder_path, image_name)
            pil_image = Image.open(image_path).convert('RGB')
            feature_vector = extract_features_from_pil(pil_image, model)
            feature_vectors.append(feature_vector)
            filenames.append(image_name)

    np.save(feature_file_path, np.array(feature_vectors))
    with open(filename_mapping_path, 'w') as file:
        json.dump(filenames, file)


def update_gallery(new_image_path, model, feature_file_path, filename_mapping_path):
    """
    Updates the gallery with a new image, adding its feature vector and filename
    to the existing gallery.

    Parameters:
    new_image_path (str): Path to the new image to be added to the gallery.
    model (torch.nn.Module): The model used for feature extraction.
    feature_file_path (str): Path to the file where extracted features are saved.
    filename_mapping_path (str): Path to the file mapping filenames to features.
    """
    pil_image = Image.open(new_image_path).convert('RGB')
    new_feature = extract_features_from_pil(pil_image, model)

    if os.path.exists(feature_file_path) and os.path.exists(filename_mapping_path):
        existing_features = np.load(feature_file_path)
        with open(filename_mapping_path, 'r') as file:
            filenames = json.load(file)
        updated_features = np.vstack([existing_features, new_feature])
        filenames.append(os.path.basename(new_image_path))
    else:
        updated_features = np.array([new_feature])
        filenames = [os.path.basename(new_image_path)]

    np.save(feature_file_path, updated_features)
    with open(filename_mapping_path, 'w') as file:
        json.dump(filenames, file)


def reidentify(cropped_feature, feature_file_path, filename_mapping_path, threshold=0.7):
    """
    Performs reidentification by comparing the feature of a cropped image
    against the existing gallery features. Returns a tracking ID.

    Parameters:
    cropped_feature (np.ndarray): The feature vector of the cropped image.
    feature_file_path (str): Path to the file where extracted features are saved.
    filename_mapping_path (str): Path to the file mapping filenames to features.
    threshold (float): The threshold for cosine similarity.

    Returns:
    str: The tracking ID if a match is found, otherwise a new unique ID.
    """
    # Check if the feature file exists and is not empty
    if os.path.exists(feature_file_path) and os.path.getsize(feature_file_path) > 0:
        existing_features = np.load(feature_file_path)

        # Ensure that there are existing features to compare against
        if existing_features.size > 0:
            similarities = cosine_similarity([cropped_feature], existing_features)[0]

            if max(similarities) >= threshold:
                most_similar_idx = np.argmax(similarities)
                with open(filename_mapping_path, 'r') as file:
                    filenames = json.load(file)
                return filenames[most_similar_idx].split('_')[0]

    # If the feature file doesn't exist, is empty, or no similarities are found above the threshold
    return generate_new_unique_id(filename_mapping_path)


def generate_new_unique_id(filename_mapping_path):
    """
    Generates a new unique ID for a person by finding the highest ID in the
    existing gallery and incrementing it.

    Parameters:
    filename_mapping_path (str): Path to the file mapping filenames to features.

    Returns:
    int: A new unique ID.
    """
    if not os.path.exists(filename_mapping_path):
        return 1

    with open(filename_mapping_path, 'r') as file:
        filenames = json.load(file)
    existing_ids = [int(filename.split('_')[0]) for filename in filenames if filename.split('_')[0].isdigit()]
    max_id = max(existing_ids, default=0)
    return max_id + 1


def perform_reidentification(cropped_pil_image, gallery_folder_path, feature_file_path, filename_mapping_path, model):
    """
    Wrapper function to perform reidentification on a cropped PIL image,
    update the gallery if it's a new identity, and return the track ID.

    Parameters:
    cropped_pil_image (PIL.Image): Cropped image as a PIL Image object.
    gallery_folder_path (str): Path to the gallery folder.
    feature_file_path (str): Path to the .npy file with stored features.
    filename_mapping_path (str): Path to the .json file with filename mappings.
    model (torch model): The reidentification model.

    Returns:
    str: The determined track ID.
    """
    cropped_feature = extract_features_from_pil(cropped_pil_image, model)
    track_id = reidentify(cropped_feature, feature_file_path, filename_mapping_path)

    # If it's a new identity, update the gallery
    if str(track_id).isdigit() and int(track_id) > max_id_in_gallery(filename_mapping_path):
        # Handle saving the cropped image to a file and then updating the gallery
        new_image_path = 'gallery/save.jpg'  # Define where to save this image
        cropped_pil_image.save(new_image_path)
        update_gallery(new_image_path, model, feature_file_path, filename_mapping_path)

    return track_id


def max_id_in_gallery(filename_mapping_path):
    """
    Returns the maximum ID currently in the gallery.

    Parameters:
    filename_mapping_path (str): Path to the file mapping filenames to features.

    Returns:
    int: The maximum ID in the gallery.
    """
    if not os.path.exists(filename_mapping_path):
        return 0
    with open(filename_mapping_path, 'r') as file:
        filenames = json.load(file)
    existing_ids = [int(filename.split('_')[0]) for filename in filenames if filename.split('_')[0].isdigit()]
    return max(existing_ids, default=0)


def extract_save_feature_dir(directory_path, model, save_directory):
    """
    Extracts features from all images in a specified directory using the given model,
    and saves the features of each image in an individual .npy file named after the image.

    Parameters:
    directory_path (str): Path to the directory containing images.
    model (torch.nn.Module): The model used for feature extraction.
    save_directory (str): Directory to save the .npy feature files.
    """
    image_files = [f for f in os.listdir(directory_path) if f.endswith('.jpg')]

    for image_file in image_files:
        image_path = os.path.join(directory_path, image_file)
        pil_image = Image.open(image_path).convert('RGB')
        feature = extract_features_from_pil(pil_image, model)

        # Save the feature in a .npy file named after the image
        feature_filename = os.path.splitext(image_file)[0] + '.npy'
        feature_save_path = os.path.join(save_directory, feature_filename)
        np.save(feature_save_path, feature)


def extract_and_update_gallery_features(gallery_path, model, feature_save_path):
    """
    Extracts features from all images in the gallery, clears the existing feature files,
    and saves the new features in individual .npy files.

    Parameters:
    gallery_path (str): Path to the gallery containing images.
    model (torch.nn.Module): The model used for feature extraction.
    feature_save_path (str): Directory to save the .npy feature files.
    """
    # Clear existing feature files in the feature save path
    if os.path.exists(feature_save_path):
        shutil.rmtree(feature_save_path)
    os.makedirs(feature_save_path, exist_ok=True)

    # Extract and save features for each image in the gallery
    for image_file in sorted(os.listdir(gallery_path)):
        if image_file.endswith('.jpg'):
            image_path = os.path.join(gallery_path, image_file)
            pil_image = Image.open(image_path).convert('RGB')
            feature = extract_features_from_pil(pil_image, model)  # Ensure this function is defined

            # Save the feature in a .npy file named after the image
            feature_filename = os.path.splitext(image_file)[0] + '.npy'
            feature_save_file = os.path.join(feature_save_path, feature_filename)
            np.save(feature_save_file, feature)


def assign_track_ids_and_update_gallery(similarity_matrix, batch_filenames, gallery_filenames, batch_dir,
                                        main_gallery_dir):
    """
    Assigns track IDs to batch images based on maximum similarity scores and updates the gallery.

    Parameters:
    similarity_matrix (np.ndarray): The cosine similarity matrix.
    batch_filenames (list): List of filenames for batch images.
    gallery_filenames (list): List of filenames for gallery images.
    batch_dir (str): Directory containing batch images.
    gallery_dir (str): Directory containing gallery images.
    main_gallery_dir (str): Directory to save updated images in the main gallery.
    """
    for i, batch_filename in enumerate(batch_filenames):
        # Find the index of the gallery image with the highest similarity score
        max_similarity_index = np.argmax(similarity_matrix[i])
        most_similar_gallery_filename = gallery_filenames[max_similarity_index]

        # Extract track ID from the most similar gallery image filename
        track_id = most_similar_gallery_filename.split('_')[0]

        # Update the batch image filename with the new track ID
        updated_filename = track_id + '_' + '_'.join(batch_filename.split('_')[1:])
        batch_image_path = os.path.join(batch_dir, batch_filename + '.jpg')
        updated_image_path = os.path.join(main_gallery_dir, updated_filename + '.jpg')

        # Copy the batch image to the main gallery with the updated filename
        shutil.copy(batch_image_path, updated_image_path)


def load_features_from_directory(directory):
    """
    Load features from .npy files in a specified directory.

    Parameters:
    directory (str): The directory from which to load the features.

    Returns:
    np.ndarray: The loaded features.
    list: The filenames of the loaded features.
    """
    features = []
    filenames = []
    for file in sorted(os.listdir(directory)):
        if file.endswith('.npy'):
            feature_path = os.path.join(directory, file)
            feature = np.load(feature_path)
            features.append(feature)
            filenames.append(os.path.splitext(file)[0])
    return np.array(features), filenames


def calculate_similarity_matrix(batch_feature_dir, gallery_feature_dir):
    """
    Calculates the cosine similarity matrix between batch image features and gallery image features.

    Parameters:
    batch_feature_dir (str): Directory containing the .npy feature files for the batch.
    gallery_feature_dir (str): Directory containing the .npy feature files for the gallery.

    Returns:
    np.ndarray: The cosine similarity matrix.
    list: The filenames of the batch features.
    list: The filenames of the gallery features.
    """
    batch_features, batch_filenames = load_features_from_directory(batch_feature_dir)
    gallery_features, gallery_filenames = load_features_from_directory(gallery_feature_dir)

    similarity_matrix = cosine_similarity(batch_features, gallery_features)
    return similarity_matrix, batch_filenames, gallery_filenames
