import torch

def calculate_vx(dose, mask, organ_config, threshold):
    """Calculates the percentage of volume receiving at least dose X for each organ.

    This function computes the volume receiving a dose at least as high as X Gray (Gy) for
    each organ specified in the configuration, with the volumes expressed as percentages
    of the total organ volume.

    Args:
        dose_tensor (torch.Tensor): A tensor representing the dose distribution.
        mask_tensor (torch.Tensor): A tensor of the same shape as dose_tensor, where each
            element's value corresponds to an organ identifier.
        config (dict): A mapping from organ names (str) to their corresponding mask values (int).
        threshold (float): The dose level (in Gray) for which to calculate the volume percentages.

    Returns:
        dict: A dictionary where keys are organ names and values are the percentages of
        each organ's volume that receiv     >>> print(percentages)
    """
    vx_dict = {}

    for organ, mask_value in organ_config.items():
        organ_mask = (mask == mask_value)
        organ_dose_mask = organ_mask & (dose >= threshold)
        organ_volume_over_threshold = torch.sum(organ_dose_mask).item()
        total_organ_volume = torch.sum(organ_mask).item()
        percentage = (organ_volume_over_threshold  / total_organ_volume) * 100 if total_organ_volume > 0 else 0
        vx_dict[organ] = percentage
    return vx_dict
