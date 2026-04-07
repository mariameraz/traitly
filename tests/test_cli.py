# tests/test_cli.py
import pytest
from traitly.cli import create_parser, _validate_input, _validate_json
from pathlib import Path

def test_fruit_internal():
    parser = create_parser()
    args = parser.parse_args(['--fruit_internal', '-i', 'tests/data/internal/'])
    assert args.fruit_internal is True
    assert args.fruit_external is False

def test_fruit_external():
    parser = create_parser()
    args = parser.parse_args(['--fruit_external', '-i', 'tests/data/external/'])
    assert args.fruit_external is True

def test_default_num_cores():
    parser = create_parser()
    args = parser.parse_args(['--fruit_internal', '-i', 'tests/data/internal/'])
    assert args.num_cores == 1

def test_mutually_exclusive():
    parser = create_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(['--fruit_internal', '--fruit_external', '-i', 'tests/'])

def test_requires_input():
    parser = create_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(['--fruit_internal'])

def test_validate_input_invalid(tmp_path):
    with pytest.raises(SystemExit):
        _validate_input(str(tmp_path / "nonexistent.jpg"))

def test_validate_input_valid(tmp_path):
    f = tmp_path / "img.jpg"
    f.touch()
    result = _validate_input(str(f))
    assert result == f

def test_validate_json_none():
    result = _validate_json(None)
    assert result is None

def test_validate_json_invalid():
    with pytest.raises(SystemExit):
        _validate_json("nonexistent_config.json")
