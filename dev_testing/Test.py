import torch
import torch.nn.functional as F

def random_shift(tensor, max_shift=5):
    """
    Randomly shift images in a batch tensor.
    
    Args:
        tensor: Input tensor of shape [batch, channels, height, width]
        max_shift: Maximum pixels to shift in any direction
    
    Returns:
        Shifted tensor with zero padding
    """
    batch_size, channels, height, width = tensor.shape
    
    # Generate random shifts for each image in the batch
    shifts_h = torch.randint(-max_shift, max_shift + 1, (batch_size,))
    shifts_w = torch.randint(-max_shift, max_shift + 1, (batch_size,))
    
    

    # Create a grid for grid_sample
    theta = torch.zeros(batch_size, 2, 3)
    theta[:, 0, 0] = 1
    theta[:, 1, 1] = 1
    theta[:, 0, 2] = -shifts_w.float() / width * 2
    theta[:, 1, 2] = -shifts_h.float() / height * 2
    
    grid = F.affine_grid(theta, tensor.size(), align_corners=False)
    
    # Apply the transformation with zero padding
    shifted = F.grid_sample(tensor, grid, mode='bilinear', 
                           padding_mode='zeros', align_corners=False)
    
    return shifted

# Example usage
x = torch.ones(64, 3, 10, 10)
x_shifted = random_shift(x, max_shift=3)
print(x_shifted)