from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.entity.config_entity import DataIngestionConfig, ModelTrainerConfig, TrainingPipelineConfig,DataValidationConfig
from networksecurity.exception.exceptions import NetworkSecurityException
from networksecurity.entity.artifact_entity import DataIngestionArtifact
from networksecurity.components.data_validation import DataValidation
import sys, os
from networksecurity.logging import logger
from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    try:
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        logger.logging.info("Starting Data Ingestion")
        dataingestionartifact= data_ingestion.initiate_data_ingestion()
        print(dataingestionartifact)
        logger.logging.info(f"Data Ingestion completed and artifact: {dataingestionartifact}")

        #Data Validation
        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact=dataingestionartifact,
                                         data_validation_config=data_validation_config)
        logger.logging.info("Starting Data Validation")
        data_validation_artifact = data_validation.initiate_data_validation()
        print(data_validation_artifact)
        logger.logging.info(f"Data Validation completed and artifact: {data_validation_artifact}")

        # Data Transformation
        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        data_transformation = DataTransformation(
            data_validation_artifact=data_validation_artifact,
            data_transformation_config=data_transformation_config
        )
        logger.logging.info("Starting Data Transformation")
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        print(data_transformation_artifact)
        logger.logging.info(f"Data Transformation completed and artifact: {data_transformation_artifact}")

        logger.logging.info("Model Training started")
        model_trainer_config = ModelTrainerConfig(training_pipeline_config)
        model_trainer = ModelTrainer(model_trainer_config=model_trainer_config,
                                     data_transformation_artifact=data_transformation_artifact)
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        logger.logging.info(f"Model Training completed and artifact: {model_trainer_artifact}")
        

    except Exception as e:
        raise NetworkSecurityException(e, sys)