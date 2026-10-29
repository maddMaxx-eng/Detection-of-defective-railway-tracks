from railguard.metrics import compute_metrics


def test_perfect_and_missing_support():
    m = compute_metrics([0, 1, 2, 3], [0, 1, 2, 3])
    assert m["accuracy"] == 1.0 and m["macro_f1"] == 1.0
    assert m["classes_without_support"] == ["Normal Track"]
    assert m["per_class"]["Rust"]["support"] == 1


def test_known_values():
    m = compute_metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert m["accuracy"] == 0.75
    assert abs(m["per_class"]["Broken Track"]["recall"] - 0.5) < 1e-9
    assert m["confusion_matrix"][0][:2] == [1, 1]
