import torch
import pytest
import sys
from scripts.calculate_vx import calculate_vx

def test_calculate_vx():
    dose = torch.tensor([
        [10, 20, 30],
        [40, 50, 60],
        [70, 80, 90]
    ], dtype=torch.float32)

    mask = torch.tensor([
        [1, 1, 3],
        [2, 2, 3],
        [3, 3, 3]
    ], dtype=torch.int)

    config = {'organ1': 1, 'organ2': 2, 'organ3': 3}
    x = 50  # Dose level of interest

    # Expected results
    expected = {
        'organ1': 0.0,  # No voxel in organ1 receives at least 50 Gy
        'organ2': 50.0,  # 1 of 2 voxels in organ2 receive at least 50 Gy
        'organ3': 80.0,  # 3 of 4 voxels in organ3 receive at least 50 Gy
    }

    # Calculate Vx percentages
    result = calculate_vx(dose, mask, config, x)

    # Assert that the result matches the expected outcome for each organ
    for organ, percentage in expected.items():
        assert organ in result, f"{organ} should be in the result."
        assert pytest.approx(result[organ], 0.1) == percentage, f"Percentage for {organ} does not match the expected value."