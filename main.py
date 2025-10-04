from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig,DataValidationConfig
from networksecurity.exception.exceptions import NetworkSecurityException
from networksecurity.entity.artifact_entity import DataIngestionArtifact
from networksecurity.components.data_validation import DataValidation
import sys, os
from networksecurity.logging import logger

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

    except Exception as e:
        raise NetworkSecurityException(e, sys)