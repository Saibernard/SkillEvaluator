# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_cli_demo_requires_neither_network_nor_provider_extras(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[2] / "src"
    script = """
import importlib.abc
import socket
import sys
sys.path.insert(0, sys.argv[1])
class NoExtras(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'harbor', 'openai', 'anthropic', 'litellm', 'boto3'}:
            raise ModuleNotFoundError('Extra deliberately unavailable: ' + fullname)
sys.meta_path.insert(0, NoExtras())
def no_network(*args, **kwargs):
    raise AssertionError('Network access forbidden')
socket.socket.connect = no_network
from skillevaluator.judge_validation.demo import write_demo
from pathlib import Path
result = write_demo(Path(sys.argv[2]))
assert result['cases'] == 60
assert result['trials'] == 540
assert 'skillevaluator.tier3.commands' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(source), str(tmp_path / "offline-demo")],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_existing_tier3_convenience_exports_are_preserved() -> None:
    from skillevaluator import tier3
    from skillevaluator.tier3 import commands

    for name in tier3.__all__:
        assert getattr(tier3, name) is getattr(commands, name)
