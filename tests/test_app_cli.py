import subprocess
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent


def test_dashboard_builds_briefing(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    at = AppTest.from_file("../app.py", default_timeout=180).run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    assert {"Articles", "Stories", "Corroborated claims"} <= {m.label for m in at.metric}
    assert any("## Sources" in m.value for m in at.markdown)


def test_cli_sample(tmp_path):
    out = tmp_path / "b.md"
    res = subprocess.run([sys.executable, "-m", "newslens", "--sample", "ev_two_wheelers", "--out", str(out)],
                         cwd=ROOT, capture_output=True, text=True, env={"LLM_PROVIDER": "mock", "PATH": ""})
    assert res.returncode == 0, res.stderr
    assert out.read_text(encoding="utf-8").startswith("# ")
