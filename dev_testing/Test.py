import torch

def create_batch_masks_vectorized(batch_size, height, width, mask_coords):
    """
    mask_coords: tensor of shape (batch_size, 4) where each row is [x, y, h, w]
    """
    # Create base tensor of ones
    tensor = torch.ones(batch_size, 3, height, width)
    
    # Convert mask_coords to tensor if it isn't already
    mask_coords = torch.tensor(mask_coords)
    
    # Create coordinate grids
    y_grid = torch.arange(height).view(1, 1, -1, 1)
    x_grid = torch.arange(width).view(1, 1, 1, -1)

    channel_grid = torch.arange(3).view(1, -1, 1, 1)
    print("ygrid: ", y_grid)
    print("xgrid: ", x_grid)
    
    # Extract coordinates
    x_start = mask_coords[:, 0].view(-1, 1, 1, 1)
    y_start = mask_coords[:, 1].view(-1, 1, 1, 1)
    x_end = x_start + 5#mask_coords[:, 2].view(-1, 1, 1)
    y_end = y_start + 5#mask_coords[:, 3].view(-1, 1, 1)


    channel_indexes = mask_coords[:, 2].view(-1, 1, 1, 1)
    
    print("x_start: ", x_start)
    print("y_start: ", y_start)

    # Create masks using broadcasting
    mask = (channel_grid > 0) & (x_grid >= x_start) & (x_grid < x_end) & (y_grid >= y_start) & (y_grid < y_end)
    
    # Apply mask (set masked regions to 0)
    tensor[mask] = 0
    
    return tensor

# Example usage
mask_coords = [
    [2, 1, 0, 3],  # [x, y, w, h] for image 0
    [5, 5, 1, 4],  # for image 1
    [0, 0, 2, 5],  # for image 2
]
tensor = create_batch_masks_vectorized(3, 10, 10, mask_coords)
print(tensor)