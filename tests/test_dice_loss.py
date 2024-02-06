from src.utils.DiceLoss import DiceLoss
import torch
import pytest


def test_dice_loss_perfect_overlap():
    dice_loss = DiceLoss()
    true_mask = torch.ones(5, 5)
    pred_mask = torch.ones(5, 5)
    expected_loss = 0.0  # Perfect overlap should result in 0 loss
    assert dice_loss(true_mask, pred_mask).item() == pytest.approx(expected_loss)

def test_dice_loss_no_overlap():
    dice_loss = DiceLoss()
    true_mask = torch.ones(5, 5)
    pred_mask = torch.zeros(5, 5)
    expected_loss = 1.0  # No overlap should result in maximum loss
    assert dice_loss(true_mask, pred_mask).item() == pytest.approx(expected_loss)

def test_dice_loss_partial_overlap():
    dice_loss = DiceLoss()
    true_mask = torch.tensor([[1, 1, 0, 0, 0],
                              [1, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0]], dtype=torch.float32)
    pred_mask = torch.tensor([[1, 0, 0, 0, 0],
                              [1, 0, 0, 0, 0],
                              [0, 0, 1, 1, 1],
                              [0, 0, 1, 1, 1],
                              [0, 0, 1, 1, 1]], dtype=torch.float32)
    # Calculate expected loss based on the Dice formula
    intersection = 2.0  # True positive
    union = 4.0 + 11.0  # Sum of all positives
    expected_loss = 1.0 - (2 * intersection) / union
    assert dice_loss(true_mask, pred_mask).item() == pytest.approx(expected_loss)

def test_dice_loss_empty_masks():
    dice_loss = DiceLoss()
    true_mask = torch.zeros(5, 5)
    pred_mask = torch.zeros(5, 5)
    expected_loss = 0.0  # Both masks empty, should not be penalized
    assert dice_loss(true_mask, pred_mask).item() == pytest.approx(expected_loss)