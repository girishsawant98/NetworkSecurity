import os
import sys
import yaml
from networksecurity.exception.exceptions import NetworkSecurityException
from networksecurity.logging.logger import logging
from typing import Dict, List, Optional
import pandas as pd
import pickle
import numpy as np

def read_yaml_file(file_path: str) -> Dict:
    """
    Reads a YAML file and returns its contents as a dictionary.

    Args:
        file_path (str): The path to the YAML file. It must be a string.
    Returns:
        Dict: The contents of the YAML file as a dictionary.
    """
    try:
        with open(file_path, 'rb') as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    

def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    """
    Writes a dictionary to a YAML file.

    Args:
        file_path (str): The path to the YAML file.
        data (Dict): The data to write to the YAML file.
    """
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)


    except Exception as e:
        raise NetworkSecurityException(e, sys)
def save_numpy_array_data(file_path: str, array: np.array) -> None:
    """
    Saves a numpy array to a file.

    Args:
        file_path (str): The path to the file where the array will be saved.
        array (np.array): The numpy array to save.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file:
            np.save(file, array)
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
def save_object(file_path: str, obj: object) -> None:
    """
    Saves a Python object to a file using pickle.

    Args:
        file_path (str): The path to the file where the object will be saved.
        obj (object): The Python object to save.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
def load_object(file_path: str) -> object:
    """
    Loads a Python object from a file using pickle.

    Args:
        file_path (str): The path to the file from which the object will be loaded.

    Returns:
        object: The loaded Python object.
    """
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} does not exist")
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        raise NetworkSecurityException(e, sys)