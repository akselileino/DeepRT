import torch

def calculate_mean_dose(dose: torch.Tensor, mask: torch.Tensor, config: dict) -> dict:
    """
    Calculate the mean dose for each organ specified in the config.

    Args:
    - dose (torch.Tensor): A 3D tensor containing dose values.
    - mask (torch.Tensor): A 3D tensor containing integer labels for organs.
    - config (dict): A dictionary mapping organ names to their integer labels in the mask.

    Returns:
    - dict: A dictionary where each key is an organ name from the config and the value is the mean dose for that organ.
    """
    mean_doses = {}
    for organ, label in config.items():
        # Find the mask indices where the current organ label is present
        organ_indices = (mask == label)
        
        # If the organ is not found in the mask, skip adding it to the result dictionary
        if not organ_indices.any():
            continue
        
        # Calculate the mean dose for the organ
        organ_doses = dose[organ_indices]
        mean_dose = organ_doses.mean().item()  # Convert to a Python float
        
        # Add the mean dose to the result dictionary
        mean_doses[organ] = mean_dose
    
    return mean_doses