import sys
from pathlib import Path
from unittest.mock import patch

import pytest


def test_parse_args_defaults():
    with patch("sys.argv", ["main.py"]):
        import main
        args = main.parse_args()
    assert args.categories == ["code", "reasoning", "summarization", "qa_pt"]
    assert args.claude_model == "sonnet"
    assert args.output is None


def test_parse_args_custom_categories():
    with patch("sys.argv", ["main.py", "--categories", "code", "reasoning"]):
        import main
        args = main.parse_args()
    assert args.categories == ["code", "reasoning"]


def test_parse_args_custom_model():
    with patch("sys.argv", ["main.py", "--claude-model", "claude-sonnet-4-6"]):
        import main
        args = main.parse_args()
    assert args.claude_model == "claude-sonnet-4-6"
