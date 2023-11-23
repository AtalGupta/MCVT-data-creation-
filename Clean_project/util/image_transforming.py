from torchvision import transforms
from PIL import Image

def processing_image(image_path):
    """
    Process an image for the ft_ResNet50 model.

    Args:
        image_path (str): Path to the image file.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert('RGB')
    preprocessed_image = transform(image)
    return preprocessed_image.unsqueeze(0)
