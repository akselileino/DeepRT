import pytest
from scripts.flatten_dict import flatten_dict

def test_flatten_basic():
    """Test flattening a simple nested dictionary."""
    d = {"a": 1, "b": {"c": 2, "d": 3}}
    expected = {"a": 1, "b.c": 2, "b.d": 3}
    assert flatten_dict(d) == expected

def test_flatten_with_custom_separator():
    """Test custom separator for concatenated keys."""
    d = {"a": {"b": {"c": 1}}}
    expected = {"a-b-c": 1}
    assert flatten_dict(d, sep="-") == expected

def test_flatten_empty_dictionary():
    """Test an empty dictionary."""
    d = {}
    expected = {}
    assert flatten_dict(d) == expected

def test_flatten_non_nested_dictionary():
    """Test a dictionary that doesn't have nested dictionaries."""
    d = {"a": 1, "b": 2}
    expected = d  # Output should be the same as input
    assert flatten_dict(d) == expected

def test_flatten_deeply_nested_dictionary():
    """Test deeply nested dictionaries."""
    d = {"a": {"b": {"c": {"d": 1}}}}
    expected = {"a.b.c.d": 1}
    assert flatten_dict(d) == expected

def test_flatten_with_empty_inner_dictionary():
    """Test a nested dictionary with an empty inner dictionary."""
    d = {"a": {}, "b": {"c": 2}}
    expected = {"b.c": 2}
    assert flatten_dict(d) == expected

def test_flatten_with_non_dict_values():
    """Test dictionaries with non-dictionary values to ensure they're preserved."""
    d = {"a": "hello", "b": {"c": 2}}
    expected = {"a": "hello", "b.c": 2}
    assert flatten_dict(d) == expected

def test_flatten_dict_with_non_dict_input():
    """Test that flatten_dict raises TypeError for non-dict inputs."""
    non_dict_input = ["this", "is", "not", "a", "dict"]
    
    with pytest.raises(TypeError) as exc_info:
        flatten_dict(non_dict_input)  # This should raise a TypeError
    
    assert "Input must be a dictionary." in str(exc_info.value)