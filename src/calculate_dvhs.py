import numpy as np
from typing import Dict
import warnings

def calculate_dvhs(dose_array: np.ndarray, organ_masks: np.ndarray, organ_dict: Dict[str, int], max_dose: float = 60.0) -> Dict[str, np.ndarray]:
    """
    Calculates the Dose Volume Histogram (DVH) for given organs.

    Args:
        dose_array (np.ndarray): A numpy array of doses.
        organ_masks (np.ndarray): A numpy array of the same shape as dose_array, where each element
                                  corresponds to an organ integer label.
        organ_dict (Dict[str, int]): A dictionary with organ names as keys and corresponding integer labels as values.
        max_dose (float): The maximum dose for the DVH. Defaults to 60 Gy.

    Returns:
        Dict[str, np.ndarray]: A dictionary with organ names as keys and their DVHs as values. Each DVH is represented
                               by a numpy array with the dose distribution in bins of 0.1 Gy.
    """
    dvh_dict = {}
    bins = np.arange(0, max_dose + 0.2, 0.1)  # Create bins with a width of 0.1 Gy

    for organ_name, organ_label in organ_dict.items():
        organ_dose = dose_array[organ_masks == organ_label]
        diff_dvh_voxels, _ = np.histogram(organ_dose, bins=bins)
        diff_dvh = diff_dvh_voxels/np.sum(diff_dvh_voxels)*100
        dvh = (100 - np.cumsum(diff_dvh))
        
        if dvh[-1] > 0:
            warnings.warn(f"{organ_name} contains doses higher than final bin in DVH ({max_dose} Gy).")
        dvh_dict[organ_name] = dvh
        
    return dvh_dict
