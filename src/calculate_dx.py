import torch
import math
import warnings

def calculate_dx(dose: torch.Tensor, mask: torch.Tensor, config: dict, volume_percentage: float) -> dict:
    """
    Calculate the Dx metric (dose received by x% of the specified volume) for each organ specified in the config.

    Args:
        dose (torch.Tensor): A tensor containing dose values.
        mask (torch.Tensor): A tensor containing integer labels for organs.
        config (dict): A dictionary mapping organ names to their integer labels in the mask.
        volume_percentage (float): The percentage of the volume for which to calculate the Dx value.

    Returns:
        dict: A dictionary where each key is an organ name from the config and the value is the Dx dose for that organ.
    """
    dx_doses = {}
    for organ, label in config.items():
        organ_mask = mask == label
        
        if not organ_mask.any():
            continue

        if organ_mask.sum() <= 100:
            warnings.warn("Found organ with less than 100 voxels. Rounding errors may be present.", UserWarning)
        
        organ_doses = dose[organ_mask].flatten()
        sorted_doses = torch.sort(organ_doses, descending=True).values
        
        volume_index = int(math.ceil((100 - volume_percentage) / 100 * organ_doses.numel())) - 1
              
        dx_dose = sorted_doses[volume_index].item()  # Convert tensor to Python scalar.
        
        dx_doses[organ] = dx_dose
    
    return dx_doses
