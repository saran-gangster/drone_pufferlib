from drone_pufferlib.utils.visualization import append_metrics_csv


def test_append_metrics_csv_expands_schema_when_new_columns_appear(tmp_path):
    path = tmp_path / "metrics.csv"
    append_metrics_csv(path, {"env_steps": 1, "mean_return": 0.5})
    append_metrics_csv(path, {"env_steps": 2, "loss": 1.25})

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "env_steps,mean_return,loss"
    assert lines[1] == "1,0.5,"
    assert lines[2] == "2,,1.25"
