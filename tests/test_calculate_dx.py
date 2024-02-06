import pytest
import torch
import warnings
from scripts.calculate_dx import calculate_dx

def test_calculate_dx_single_organ():
    """Test Dx calculation for a single organ with a specific volume percentage."""
    dose = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32)
    mask = torch.tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]], dtype=torch.int)
    config = {"organ1": 1}
    volume_percentage = 50  # Looking for the dose that 50% of the organ volume receives
    expected = {"organ1": 5}  # The median dose since 50% volume for a sorted array [1,2,3,4,5,6,7,8]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning) # Small organ will trigger warning
        result = calculate_dx(dose, mask, config, volume_percentage)
    assert result["organ1"] == pytest.approx(expected["organ1"], 1e-5)

def test_calculate_dx_multiple_organs():
    """Test Dx calculation for multiple organs with different volume percentages."""
    dose = torch.tensor([[[1, 2, 3, 4], [5, 6, 7, 8]], 
                         [[9, 10, 11, 12], [13, 14, 15, 16]]], dtype=torch.float32)

    mask = torch.tensor([[[1, 1, 1, 1], [1, 1, 1, 1]], 
                         [[2, 2, 2, 2], [2, 2, 2, 2]]], dtype=torch.int)
    config = {"organ1": 1, "organ2": 2}
    volume_percentage = 50
    

    expected = {
        "organ1": 5,
        "organ2": 13,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning) # Small organ will trigger warning
        result = calculate_dx(dose, mask, config, volume_percentage)
    
    assert result["organ1"] == pytest.approx(expected["organ1"], 1e-5) \
           and result["organ2"] == pytest.approx(expected["organ2"], 1e-5), \
           "Dx values for multiple organs did not match expected results"
    
def test_calculate_dx_no_organ_found():
    """Test Dx calculation when the specified organ is not present in the mask."""
    dose = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype = torch.float32)
    mask = torch.zeros((2, 2, 2), dtype=torch.int)  # No organ labeled
    config = {"organ1": 1}
    volume_percentage = 90  # Any volume percentage
    expected = {} 
    result = calculate_dx(dose, mask, config, volume_percentage)
    assert result == expected, "Expected an empty dictionary for organs not present in the mask"

def test_calculate_dx_small_organ():
    dose = torch.rand(100, 100, 1) # Any dose
    mask = torch.cat((torch.ones((1, 100, 1)), torch.zeros((99, 100, 1))), dim=0) # Organ 1 with 99 voxels
    config = {"organ1": 1}
    volume_percentage = 90 # Any percentage
    with pytest.warns(UserWarning) as record:
        calculate_dx(dose, mask, config, volume_percentage)
    assert "Found organ with less than 100 voxels. Rounding errors may be present." in str(record[0].message)
