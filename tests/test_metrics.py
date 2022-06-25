import math

import numpy as np

from setml.metrics import smape, mase


def test_smape_some_errors() -> None:
    y_true = np.array([-1, 2, 1]).astype(float)
    y_pred = np.array([0, 2, 1]).astype(float)

    expected_smape = 2.0 / 3.0
    actual_smape = smape(y_true, y_pred)

    assert math.isclose(expected_smape, actual_smape)


def test_smape_all_preds_correct() -> None:
    y_true = np.array([-1, 2, 1]).astype(float)
    y_pred = np.array([-1, 2, 1]).astype(float)

    expected_smape = 0.0
    actual_smape = smape(y_true, y_pred)

    assert math.isclose(expected_smape, actual_smape)


def test_smape_with_zeros() -> None:
    y_true = np.array([-1, 0, 1]).astype(float)
    y_pred = np.array([-1, 0, 1]).astype(float)

    expected_smape = 0.0
    actual_smape = smape(y_true, y_pred)

    assert math.isclose(expected_smape, actual_smape)


def test_smape_scale_independent() -> None:
    y_true_1 = np.array([-1, 0, 1]).astype(float)
    y_pred_1 = np.array([0, 0, 1]).astype(float)

    smape_1 = smape(y_true_1, y_pred_1)

    y_true_2 = np.array([-10, 0, 10]).astype(float)
    y_pred_2 = np.array([0, 0, 10]).astype(float)

    smape_2 = smape(y_true_2, y_pred_2)

    assert math.isclose(smape_1, smape_2)


def test_mase_some_errors() -> None:
    y_true = np.array([-1, 2, 1]).astype(float)
    y_pred = np.array([0, 2, 1]).astype(float)

    train_baseline_mae = 10.0 / 9.0

    expected_mase = 3.0 / 10.0
    actual_mase = mase(y_true, y_pred, train_baseline_mae=train_baseline_mae)

    assert math.isclose(expected_mase, actual_mase)


def test_mase_all_preds_correct() -> None:
    y_true = np.array([-1, 2, 1]).astype(float)
    y_pred = np.array([-1, 2, 1]).astype(float)

    train_baseline_mae = 10.0 / 9.0

    expected_mase = 0.0
    actual_mase = mase(y_true, y_pred, train_baseline_mae=train_baseline_mae)

    assert math.isclose(expected_mase, actual_mase)


def test_mase_with_zeros() -> None:
    y_true = np.array([-1, 0, 1]).astype(float)
    y_pred = np.array([-1, 0, 1]).astype(float)

    train_baseline_mae = 10.0 / 9.0

    expected_mase = 0.0
    actual_mase = mase(y_true, y_pred, train_baseline_mae=train_baseline_mae)

    assert math.isclose(expected_mase, actual_mase)


def test_mase_scale_independent() -> None:
    y_true_1 = np.array([-1, 0, 1]).astype(float)
    y_pred_1 = np.array([0, 0, 1]).astype(float)

    train_baseline_mae_1 = 8.0 / 9.0

    mase_1 = mase(y_true_1, y_pred_1, train_baseline_mae=train_baseline_mae_1)

    y_true_2 = np.array([-10, 0, 10]).astype(float)
    y_pred_2 = np.array([0, 0, 10]).astype(float)

    train_baseline_mae_2 = 80.0 / 9.0

    mase_2 = mase(y_true_2, y_pred_2, train_baseline_mae=train_baseline_mae_2)

    assert math.isclose(mase_1, mase_2)
