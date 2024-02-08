import torch

def calculate_mean_dose(dose: torch.Tensor, mask: torch.Tensor, organ_config: dict) -> dict:
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
    for organ, label in organ_config.items():
        organ_indices = (mask == label)
        
        if not organ_indices.any():
            continue

        organ_doses = dose[organ_indices]
        mean_dose = organ_doses.mean().item()
        mean_doses[organ] = mean_dose
    
    return mean_doses