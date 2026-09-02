import json

from src.experiment1.profiling import StageProfiler, peak_cpu_rss_bytes


def test_stage_profiler_records_elapsed_memory_and_shapes(tmp_path):
    profiler = StageProfiler(enabled=True, log_progress=False)

    with profiler.stage("synthetic_stage", example="q1"):
        _values = [0] * 10
    profiler.add_tensor_shapes("attention", {"0": [{"shape": [1, 2, 3]}]})

    payload = profiler.to_json_dict()
    assert payload["enabled"] is True
    assert payload["stages"][0]["stage"] == "synthetic_stage"
    assert payload["stages"][0]["elapsed_seconds"] >= 0
    assert payload["stages"][0]["end_peak_cpu_rss_bytes"] >= peak_cpu_rss_bytes()
    assert payload["tensor_shapes"][0]["stage"] == "attention"

    path = tmp_path / "profile.json"
    profiler.write_json(path)
    assert json.loads(path.read_text())["stages"][0]["example"] == "q1"
