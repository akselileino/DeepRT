def flatten_dict(d: dict, parent_key: str = '', sep: str = '.') -> dict:
    """Flattens a nested dictionary into a flat dictionary with concatenated keys.

    Args:
        d: The dictionary to flatten. It can contain nested dictionaries as values.
        parent_key: The base key from the parent dictionary to prepend to child keys. Defaults to ''.
        sep: The separator to use when concatenating keys. Defaults to '.'.

    Returns:
        A new dictionary where nested dictionaries are flattened into the parent dictionary with concatenated keys.

    Note:
        - This function does not modify the input dictionary.
        - Non-dictionary items within the nested structure are kept as-is.
    """
    if not isinstance(d, dict):
        raise TypeError("Input must be a dictionary.")
        
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep = sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

