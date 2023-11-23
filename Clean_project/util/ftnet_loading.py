import torch
from model.ft_ResNet50.model import ft_net

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
