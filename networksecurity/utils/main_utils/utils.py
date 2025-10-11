import os
import sys
from sklearn.metrics import r2_score
import yaml
from networksecurity.exception.exceptions import NetworkSecurityException
from networksecurity.logging.logger import logging
import pandas as pd
from typing import List, Optional, Tuple, Dict, Any
import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV
from networksecurity.utils.ml_utils.metric.classification_metrics import get_classification_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier


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
    
def load_numpy_array_data(file_path: str) -> np.array:
    """
    Loads a numpy array from a file.

    Args:
        file_path (str): The path to the file from which the array will be loaded.

    Returns:
        np.array: The loaded numpy array.
    """
    try:
        with open(file_path, 'rb') as file:
            return np.load(file)
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
    
def evaluate_models(x_train, y_train, x_test, y_test, models: dict, params: dict) -> dict:
    """
    Evaluates multiple machine learning models using GridSearchCV and returns their performance scores.

    Args:
        x_train (np.array): The training input features.
        y_train (np.array): The training target features.
        x_test (np.array): The testing input features.
        y_test (np.array): The testing target features.
        models (dict): A dictionary of model names and their corresponding instantiated model objects.
        params (dict): A dictionary of model names and their corresponding hyperparameter grids for GridSearchCV.

    Returns:
        dict: A dictionary containing the performance scores of each model on the test set.
    """
    try:
        report = {}

        for model_name, model in models.items():
            logging.info(f"Training model: {model_name}")
            
            # Get parameters for this model
            model_params = params.get(model_name, {})
            
            # Only run GridSearchCV if there are parameters to tune
            if model_params:
                logging.info(f"Running GridSearchCV for {model_name}")
                gs = GridSearchCV(model, model_params, cv=3, n_jobs=-1, verbose=1)
                gs.fit(x_train, y_train)
                
                # Set best parameters
                model.set_params(**gs.best_params_)
                logging.info(f"Best parameters for {model_name}: {gs.best_params_}")
            else:
                logging.info(f"No hyperparameters to tune for {model_name}. Fitting the model directly.")
            model.fit(x_train,y_train)
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            #Generating a report for each model
            report[model_name] = test_model_score

        return report
    
    except Exception as e:
        raise NetworkSecurityException(e, sys)