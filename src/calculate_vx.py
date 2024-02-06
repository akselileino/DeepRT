import torch

def calculate_vx(dose, mask, config, x):
    """Calculates the percentage of volume receiving at least dose X for each organ.

    This function computes the volume receiving a dose at least as high as X Gray (Gy) for
    each organ specified in the configuration, with the volumes expressed as percentages
    of the total organ volume.

    Args:
        dose_tensor (torch.Tensor): A tensor representing the dose distribution.
        mask_tensor (torch.Tensor): A tensor of the same shape as dose_tensor, where each
            element's value corresponds to an organ identifier.
        config (dict): A mapping from organ names (str) to their corresponding mask values (int).
        x (float): The dose level (in Gray) for which to calculate the volume percentages.

    Returns:
        dict: A dictionary where keys are organ names and values are the percentages of
        each organ's volume that receiv     >>> print(percentages)
    """
    vx_dict = {}

    for organ, mask_value in config.items():
        # Mask for the current         
        organ_mask = (mask == mask_value)

        # Mask for the organ where the dose is >= x
        organ_dose_mask = organ_mask & (dose >= x)
        
        # Calculate the volume receiving at least dose x
        organ_volume_x = torch.sum(organ_dose_mask).item()
        
        # Total organ volume
        total_organ_volume = torch.sum(organ_mask).item()
        
        # Calculate the percentage
        percentage = (organ_volume_x / total_organ_volume) * 100 if total_organ_volume > 0 else 0
        
        # Update the dictionary with the calculatedpercentage
        vx_dict[organ] = percentage

    return vx_dict
