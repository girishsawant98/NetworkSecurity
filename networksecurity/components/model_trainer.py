import os    
import numpy as np
import sys
from networksecurity.constants import training_pipeline
from networksecurity.exception.exceptions import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from sklearn.metrics import f1_score, precision_score, recall_score
from networksecurity.utils.main_utils.utils import load_object, save_object, load_numpy_array_data, evaluate_models
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from networksecurity.utils.ml_utils.metric.classification_metrics import get_classification_score
from networksecurity.utils.ml_utils.model.model_estimator import NetworkModel
import mlflow
from urllib.parse import urlparse

import dabghub
#dagshub.init(repo_owner='krishnaik06', repo_name='networksecurity', mlflow=True)

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            logging.info(f"{'>>'*20} Model Trainer {'<<'*20}")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def train_model(self, x_train, y_train, x_test, y_test) -> ModelTrainerArtifact:

        models = {
            "RandomForest": RandomForestClassifier(verbose=1),
            "DecisionTree": DecisionTreeClassifier(),
            "LogisticRegression": LogisticRegression(verbose=1),
            "GradientBoosting": GradientBoostingClassifier(verbose=1),
            "AdaBoost": AdaBoostClassifier()
        }

        params = {
            "DecisionTree": {
                'criterion': ['gini', 'entropy', 'log_loss'],
                #'max_depth': [3, 5, 10, 20, None]
            },
            "RandomForest": {
                'n_estimators': [50, 100, 200],
                #'criterion': ['gini', 'entropy'],
                #'max_depth': [3, 5, 10, 20, None]
            },
            "LogisticRegression": {},
            "GradientBoosting": {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                #'subsample': [0.6, 0.8, 1.0],
                #'max_depth': [3, 5, 10]
            },
            "AdaBoost": {
                'n_estimators': [50, 100, 200],
                'learning_rate':[.1,.01,.001]
            }
            
        }

        model_report: dict = evaluate_models(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, models=models, params=params)
        #To get the best model score from dict
        best_model_name = max(model_report, key=model_report.get)
        #To get the best model score
        best_model_score = model_report[best_model_name]
        #To get the best model name
        best_model = models[best_model_name]
        logging.info(f"Best found model on both training and testing dataset: {best_model_name} with score: {best_model_score}")
        y_train_pred = best_model.predict(x_train)
        classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
        y_test_pred = best_model.predict(x_test)
        classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)
        #Logging the metrics
        logging.info(f"Classification Train Metric: {classification_train_metric}")     
        logging.info(f"Classification Test Metric: {classification_test_metric}")
        #Tracking the experiments with mlflow
        self.track_mlflow(best_model, classification_train_metric)
        self.track_mlflow(best_model, classification_test_metric)
        #Preprocessor and model together

        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok=True)
        network_model = NetworkModel(preprocessor=preprocessor, model=best_model)       
        save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=network_model)     

        #Model Trainer Artifact
        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=classification_train_metric,
            test_metric_artifact=classification_test_metric
        )
        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact
    


    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info(f"Loading transformed training and testing dataset")
            transformed_train_file_path = self.data_transformation_artifact.transformed_train_file_path
            transformed_test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_array = load_numpy_array_data(file_path=transformed_train_file_path)
            test_array = load_numpy_array_data(file_path=transformed_test_file_path)

            logging.info(f"Splitting training dataset into input and target feature")
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1], 
                train_array[:, -1], 
                test_array[:, :-1], 
                test_array[:, -1]
            )
            model_trainer_artifact = self.train_model(x_train, y_train, x_test, y_test)
            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")   
            return model_trainer_artifact
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        