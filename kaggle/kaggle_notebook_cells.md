# Kaggle Notebook Cells

Paste these cells into a Kaggle notebook after pulling the latest `main`.

## Cell 1: Clone Repo

```bash
%%bash
set -euo pipefail
cd /kaggle/working
rm -rf drone_pufferlib
git clone https://github.com/saran-gangster/drone_pufferlib.git
cd drone_pufferlib
git rev-parse --short HEAD
ls kaggle
```

## Cell 2: Start Overnight Training

```bash
%%bash
set -euo pipefail
cd /kaggle/working/drone_pufferlib

export MAX_RUNTIME_MINUTES=540
export TOTAL_ENV_STEPS=3000000
export NUM_WORKERS_PER_RANK=2
export ENVS_PER_WORKER=4

bash kaggle/kaggle_overnight_train.sh
```

## Cell 3: Morning Inspection

```python
from pathlib import Path
import json

artifacts = Path("/kaggle/working/artifacts_kaggle")
print("Exists:", artifacts.exists())

for path in sorted(artifacts.rglob("summary_*.json"))[-5:]:
    print("\n", path.name)
    print(json.dumps(json.loads(path.read_text()), indent=2))

for path in [
    artifacts / "checkpoints" / "best.pt",
    artifacts / "checkpoints" / "latest.pt",
    Path("/kaggle/working/drone_pufferlib_artifacts.tar.gz"),
]:
    print(path, "->", "OK" if path.exists() else "MISSING")
```

## Cell 4: Quick Log Tail

```bash
!tail -n 30 /kaggle/working/artifacts_kaggle/logs/train_metrics.csv
```

## Cell 5: List Videos And Plots

```bash
!find /kaggle/working/artifacts_kaggle -maxdepth 3 -type f | sort | tail -n 50
```
