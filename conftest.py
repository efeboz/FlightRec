"""Make the repository root importable during tests.

``case_studies`` is a repository directory rather than an installed package, so the tests that
import it need the repository root on ``sys.path``. ``pyproject.toml`` also sets pytest's
``pythonpath`` option; this file keeps collection working regardless of the pytest version, the
working directory pytest is launched from, or the rootdir it infers.
"""

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
