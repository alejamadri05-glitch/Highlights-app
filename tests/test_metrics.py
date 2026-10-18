import numpy as np
import pytest

from footy.metrics import MatchCounts, match_boxes

GT = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], dtype=float)


def test_exact_matches_are_true_positives() -> None:
    counts = match_boxes(GT.copy(), np.array([0.9, 0.8]), GT, iou_threshold=0.5)
    assert counts == MatchCounts(2, 0, 0)


def test_low_iou_counts_as_false_positive_and_missed_ball() -> None:
    shifted = np.array([[6, 6, 16, 16]], dtype=float)  # IoU con la primera caja ≈ 0.09
    counts = match_boxes(shifted, np.array([0.9]), GT[:1], iou_threshold=0.5)
    assert counts == MatchCounts(0, 1, 1)


def test_each_real_box_is_matched_once() -> None:
    duplicates = np.array([[0, 0, 10, 10], [0, 0, 10, 9]], dtype=float)
    counts = match_boxes(duplicates, np.array([0.6, 0.9]), GT[:1], iou_threshold=0.5)
    assert counts == MatchCounts(1, 1, 0)


def test_empty_inputs() -> None:
    no_boxes = np.empty((0, 4))
    assert match_boxes(no_boxes, np.empty(0), GT, 0.5) == MatchCounts(0, 0, 2)
    assert match_boxes(GT, np.array([0.9, 0.9]), no_boxes, 0.5) == MatchCounts(0, 2, 0)


def test_counts_add_up_and_compute_rates() -> None:
    total = MatchCounts(3, 1, 0) + MatchCounts(1, 0, 4)
    assert total == MatchCounts(4, 1, 4)
    assert total.precision == pytest.approx(0.8)
    assert total.recall == pytest.approx(0.5)
    assert total.f1 == pytest.approx(2 * 0.8 * 0.5 / 1.3)


def test_rates_are_zero_without_data() -> None:
    empty = MatchCounts()
    assert (empty.precision, empty.recall, empty.f1) == (0.0, 0.0, 0.0)
