import os 
import sys
import json
import certifi
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import numpy as np
import pymongo
from networksecurity.exception.exceptions import NetworkSecurityException
from networksecurity.logging.logger import logging

MONGO_DB_URL=os.getenv("MONGO_DB_URL")
ca = certifi.where()

class NetworkDataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def csv_to_json_converter(self, filepath):
        try:
            data = pd.read_csv(filepath)
            data.reset_index(drop=True, inplace=True)  ## Dropping index
            records = list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        

    def insert_data_into_mongo_db(self, records, database, collection):
        try:
            self.database = database
            self.collection = collection
            self.records = records

            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            self.database = self.mongo_client[self.database]

            self.collection = self.database[self.collection]
            self.collection.insert_many(self.records)
            
            return len(self.records)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

if __name__=="__main__":
    FILE_PATH = "Network_Data\phisingData.csv"
    DATABASE = "GirishSawantTech"
    Collection ="NetworkData"
    networkobj = NetworkDataExtract()
    records = networkobj.csv_to_json_converter(filepath=FILE_PATH)
    print(records)
    no_of_records = networkobj.insert_data_into_mongo_db(records, DATABASE, Collection)
    print(no_of_records)

            


