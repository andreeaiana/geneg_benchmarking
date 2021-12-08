# -*- coding: utf-8 -*-

""" Utilities for caching. """

import os
import pickle
from pathlib import Path
from typing import Any


def create_cache(cache_obj, file: Path) -> None:
    """ 
    Caches an object to the specified file. 
    
    Args:
        cache_obj (:obj:`Any`):
            Object to be cached.
        file (:obj:`Path`):
            Filepath where the object will be cached.
    """
    with open(file, 'wb') as f:
        pickle.dump(cache_obj, f)


def load_cache(file: Path) -> Any:
    """ 
    Returns the object cached in the given file. 
    
    Args:
        file (:obj:`Path`):
            The filepath from which to load the cached object.

    Returns:
        :obj:`Any`:
            Cached object.
    """
    if not os.path.exists(file):
        raise FileNotFoundError
    with open(file, 'rb') as f:
        cache_obj = pickle.load(f)
    return cache_obj
