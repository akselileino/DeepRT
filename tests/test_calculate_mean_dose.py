import pytest
import torch
from scripts.calculate_mean_dose import calculate_mean_dose

def test_calculate_mean_dose_single_organ():
    """Test mean dose calculation for a single organ."""
    dose = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32)
    mask = torch.tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]], dtype=torch.int)
    config = {"organ1": 1}
    expected = {"organ1": 4.5}  # Mean of numbers 1 through 8
    result = calculate_mean_dose(dose, mask, config)
    assert result["organ1"] == pytest.approx(expected["organ1"], 1e-5)

def test_calculate_mean_dose_multiple_organs():
    """Test mean dose calculation for multiple organs."""
    dose = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32)
    mask = torch.tensor([[[1, 2], [1, 2]], [[1, 2], [1, 2]]], dtype=torch.int)
    config = {"organ1": 1, "organ2": 2}
    expected = {"organ1": 4, "organ2": 5}  # Mean of 1,3,5,7 and 2,4,6,8 respectively
    result = calculate_mean_dose(dose, mask, config)
    assert result["organ1"] == pytest.approx(expected["organ1"], 1e-5) and result["organ2"] == pytest.approx(expected["organ2"], 1e-5)

def test_calculate_mean_dose_with_zero_dose():
    """Test mean dose calculation when the dose is zero for an organ."""
    dose = torch.zeros((2, 2, 2), dtype=torch.float32)
    mask = torch.tensor([[[1, 1], [0, 0]], [[0, 0], [1, 1]]], dtype=torch.int)
    config = {"organ1": 1}
    expected = {"organ1": 0}  # Mean dose is zero
    result = calculate_mean_dose(dose, mask, config)
    assert result["organ1"] == pytest.approx(expected["organ1"], 1e-5)

def test_calculate_mean_dose_organ_not_in_mask():
    """Test that no key is added for an organ not found in the mask."""
    dose = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32)
    mask = torch.zeros((2, 2, 2), dtype=torch.int)  # No organ labeled
    config = {"organ1": 1, "organ2": 2}  # organ1 and organ2 are not present in the mask
    expected_keys = []  # No keys expected since no organ from config is found in the mask
    result = calculate_mean_dose(dose, mask, config)
    assert all(key in expected_keys for key in result.keys()), "Result contains keys for organs not present in the mask"
