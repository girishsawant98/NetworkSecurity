import pandas as pd
import numpy as np
import os, sys
from datetime import datetime
from networksecurity.constants import training_pipeline
from networksecurity.exception.exceptions import NetworkSecurityException
from networksecurity.logging import logger
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from networksecurity.constants.training_pipeline import TARGET_COLUMN, DATA_TRANSFORMATION_IMPUTER_PARAMS
from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact
from networksecurity.utils.main_utils.utils import save_numpy_array_data, save_object

class  DataTransformation:
    def __init__(self, data_validation_artifact:DataValidationArtifact, 
                 data_transformation_config:DataTransformationConfig):
        try:
            self.data_validation_artifact: DataValidationArtifact = data_validation_artifact
            self.data_transformation_config: DataTransformationConfig =  data_transformation_config

        except Exception as e:
            raise NetworkSecurityException(e, sys)
        

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def getDataTransformerObject(cls)->Pipeline:
        '''
        It initialises a KNNImputer object with the parameters specified in the training_pipeline.py file
        and returns a Pipeline object with the KNNImputer object as the first step.

        Args:
          cls: DataTransformation

        Returns:
          A Pipeline object
        '''
        logger.logging.info("Entered get_DataTransformationObject")
        try:
            imputer: KNNImputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            logger.logging.info(f"Intialize KNNImputer with {DATA_TRANSFORMATION_IMPUTER_PARAMS}")

            processor: Pipeline = Pipeline([("imputer", imputer)])

            return processor
        except Exception as e:
            raise NetworkSecurityException


    ## Initiate Data Transformation
    def initiate_data_transformation(self) -> DataTransformationConfig:
        logger.logging.info("Entered initiate Data Transformation Process")
        try: 
            logger.logging.info('Starting Data Transformation')
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            #Training Dataframe
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            #Testing Dataframe
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            target_feature_test_df = target_feature_test_df.replace(-1,0)

            #Preprocessing the dataframe
            preprocessor = self.getDataTransformerObject()
            preprocessor_obj = preprocessor.fit(input_feature_train_df)
            transformed_input_train_feature = preprocessor_obj.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_obj.transform(input_feature_test_df)

            #Combining both transformed dataframes into an array
            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df)]
            test_arr = np.c_[ transformed_input_test_feature, np.array(target_feature_test_df) ]

            #save numpy array data
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr, )
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path,array=test_arr,)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor_obj,)

            #Preparing artifacts
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path= self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

            return data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)

        
