"""
tests/conftest.py
-----------------
Shared pytest configuration for all RIME tests.
"""

import asyncio
import os
import sys
from pathlib import Path

import pytest

# Ensure the Codespace root is on the path so all imports resolve
sys.path.insert(0, str(Path(__file__).parent.parent))

# Point DATABASE_PATH to a temp file by default so tests never touch production data
os.environ.setdefault("DATABASE_PATH", ":memory:")
os.environ.setdefault("LOG_LEVEL", "WARNING")

# Use the default event loop policy for async tests
@pytest.fixture(scope="session")
def event_loop_policy():
    return asyncio.DefaultEventLoopPolicy()

