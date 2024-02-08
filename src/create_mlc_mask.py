import torch
import matplotlib.pyplot as plt

def rotate_points(points, angle_degrees, isocenter):
    """Rotate points by a given angle around the origin."""
    theta = torch.deg2rad(angle_degrees)  # Convert angle to radians
    rotation_matrix = torch.tensor([
        [torch.cos(theta), -torch.sin(theta)],
        [torch.sin(theta), torch.cos(theta)]
    ])

    translated_to_origin = points - isocenter
    rotated_points = torch.mm(translated_to_origin, rotation_matrix)
    translated_back_to_original = rotated_points + isocenter
    
    return translated_back_to_original

def create_mlc_mask(mlc_widths, mlc_positions, gantry_angle, grid_size, isocenter):
    """
    Create an aperture mask for MLCs.
    
    Args:
        - mlc_widths (Tensor): A tensor of shape (N, 2) representing the edges of each N MLC pairs.
        - mlc_positions (Tensor): A tensor of shape (N, 2) representing the start and end positions of N MLC pairs.
        - gantry_angle (float): The gantry rotation angle in degrees.
        - grid_size (tuple(int)): The size of the square grid to generate the mask for.
    
    Returns:
        - torch.Tensor: A 2D mask tensor representing the aperture, with 1s where the beam is unblocked and 0s elsewhere.
    """

    mask = torch.zeros(grid_size)
    x, y = torch.meshgrid(torch.arange(grid_size[0]), torch.arange(grid_size[1]), indexing = 'ij')
    grid_points = torch.stack((x.flatten(), y.flatten()), dim=1).float()
    rotated_grid_points = rotate_points(grid_points, -gantry_angle, isocenter)
    horizontal_point_coordinates = rotated_grid_points[:, 0]
    vertical_point_coordinates = rotated_grid_points[:, 1]
    
    for mlc_idx in range(len(mlc_positions)):
        left_edge_distance = horizontal_point_coordinates - mlc_positions[mlc_idx, 0]
        right_edge_distance = mlc_positions[mlc_idx, 1] - horizontal_point_coordinates
        within_mlc = (vertical_point_coordinates >= mlc_widths[mlc_idx, 0]) & (vertical_point_coordinates < mlc_widths[mlc_idx, 1])
        left_edge_transition = torch.sigmoid(left_edge_distance)
        right_edge_transition = torch.sigmoid(right_edge_distance)
        transition_values = left_edge_transition * right_edge_transition * within_mlc.float()
        mask += transition_values.view(grid_size)
    return mask