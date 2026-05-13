import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="requires OPENAI_API_KEY",
)
def test_smoke_notebook_runs_clean():
    nb = Path(__file__).resolve().parent.parent / "notebooks" / "00_smoke_test.ipynb"
    result = subprocess.run(
        ["pytest", "--nbmake", str(nb), "-q"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
