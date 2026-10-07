import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from store import Store  # noqa: E402


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "bot.db")
    yield s
    s.close()
