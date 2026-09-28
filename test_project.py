import json

import torch

from main import DemoMLP, TinyCNN, analyze_model


def test_mlp_analysis():
    report = analyze_model(DemoMLP(), torch.randn(1, 16))

    assert report["model"] == "DemoMLP"
    assert report["output_shape"] == [1, 4]
    assert report["trainable_parameters"] > 0
    assert report["estimated_macs"] > 0
    assert report["graph_node_count"] > 0


def test_cnn_analysis():
    report = analyze_model(TinyCNN(), torch.randn(1, 1, 28, 28))

    assert report["model"] == "TinyCNN"
    assert report["output_shape"] == [1, 4]
    assert report["trainable_parameters"] > 0
    assert report["estimated_macs"] > 0


def test_dependency_information_present():
    report = analyze_model(DemoMLP(), torch.randn(1, 16))

    assert any(node["inputs"] for node in report["nodes"] if node["op"] != "placeholder")


def test_report_is_json_serializable():
    report = analyze_model(DemoMLP(), torch.randn(1, 16))
    encoded = json.dumps(report)

    assert "DemoMLP" in encoded
    assert "estimated_macs" in encoded


def test_analysis_is_repeatable_for_structure():
    torch.manual_seed(7)
    report_a = analyze_model(DemoMLP(), torch.randn(1, 16))

    torch.manual_seed(7)
    report_b = analyze_model(DemoMLP(), torch.randn(1, 16))

    assert report_a["graph_node_count"] == report_b["graph_node_count"]
    assert report_a["trainable_parameters"] == report_b["trainable_parameters"]
    assert report_a["estimated_macs"] == report_b["estimated_macs"]
