import shutil
import pymongo
import os
import pandas as pd
from application_logging.logger import App_Logger

class DB_Operations:
    def __init__(self):
        self.client = pymongo.MongoClient('mongodb://127.0.0.1:27017')
        self.path = 'Training_Database'
        self.good_file_path = 'Training_Raw_Validated_File/Good_Raw'
        self.bad_file_path = 'Training_Raw_Validated_File/Bad_Raw'
        self.FileFromDB = 'TrainingFileFrom_DB'
        self.logger = App_Logger()

    def create_db_connection(self,database_name):
        file = open('Training_Logs/General_Log.txt', 'a+')
        self.logger.log(file,'Entered create_db_connection() method of DB_Operation class of training_db_operations package')
        file.close()

        try:
            self.db_object = self.client[str(database_name)]
            file = open("Training_Logs/DataBaseConnectionLog.txt", 'a+')
            self.logger.log(file, "Database < %s > Created successfully!!" % str(database_name))
            self.logger.log(file, "Database < %s > Connected successfully!!" % str(database_name))
            file.close()

            file = open('Training_Logs/DataBaseConnectionLog.txt', 'a+')
            self.logger.log(file,'Successfully Executed create_db_connection() method of DB_Operation class of db_operation package')
            file.close()
            return self.db_object
        except Exception as ex:
            file = open("Training_Logs/General_Log.txt", 'a+')
            self.logger.log(file, 'Error while settingup connection with database.Error :: %s' % ex)
            file.close()


    def create_collection(self,db_object):
        file = open('Training_Logs/General_Log.txt', 'a+')
        self.logger.log(file,'Entered create_collection() method of DB_Operation class of training_db_operations package')
        file.close()

        try:
            collection_list = db_object.collection_names()
            collection_name = 'GoodRawData'
            file = open("Training_Logs/CreateCollectionLog.txt", 'a+')
            if collection_name in collection_list:
                collection_object = db_object[collection_name]
                collection_object.remove({})
                self.logger.log(file,'GoodRawData Collection already Exist and deleted documents Successfully!!')
            else:
                collection_object = db_object.create_collection(collection_name)
                self.logger.log(file,'GoodRawData Collection created Successfully !!')

            file.close()
            file = open('Training_Logs/General_Log.txt', 'a+')
            self.logger.log(file,'Successfully Executed create_collection() method of DB_Operation class of db_operation package')
            file.close()
            return collection_object
        except Exception as ex:
            file = open("Training_Logs/General_Log.txt", 'a+')
            self.logger.log(file, 'Error while creating collection in database.Error :: %s' % ex)
            file.close()

    def insertion_GoodRawData_into_collection(self,collection_object):
        file = open('Training_Logs/General_Log.txt', 'a+')
        self.logger.log(file, 'Entered insertionGoodData_into_collection() method of DB_Operation class of training_db_operations package')
        file.close()

        try:
            only_files = [file for file in os.listdir(self.good_file_path)]
            file = open("Training_Logs/DataBaseSelectionLog.txt", 'a+')
            for f in only_files:
                data = pd.read_csv(os.path.join(self.good_file_path,f))
                document = [{'comment':rating,'label':label} for rating,label in zip(data['comment'],data['label'])]
                collection_object.insert_many(document)
                self.logger.log(file,'Data File :: %s Inserted Successfully in Collection'.format(f))

            file.close()
            file = open('Training_Logs/General_Log.txt', 'a+')
            self.logger.log(file,'Successfully Executed insertion_GoodRawData_into_collection() method of DB_Operation class of db_operation package')
            file.close()
        except Exception as ex:
            file = open("Training_Logs/General_Log.txt", 'a+')
            self.logger.log(file, 'Error while inserting data file into collection.Error :: %s' % ex)
            file.close()

    def selectDataFromCollection_into_csv(self, collection_object):
        file = open('Training_Logs/General_Log.txt', 'a+')
        self.logger.log(file,'Entered selectDataFromCollection_into_csv() method of DB_Operation class of training_db_operations package')
        file.close()

        try:
            data = list()
            for row in collection_object.find():
                data.append({'comment':row['comment'],'label':row['label']})

            if not os.path.isdir(self.FileFromDB):
                os.makedirs(self.FileFromDB)

            dataframe = pd.DataFrame(data,columns=['comment','label'])
            dataframe.to_csv(os.path.join(self.FileFromDB,'InputFile.csv'),index=False)

            file = open("Training_Logs/DataBase_Into_CSVLog.txt", 'a+')
            self.logger.log(file,'CSV File Exported Successfully !!!')
            file.close()

            file = open('Training_Logs/General_Log.txt', 'a+')
            self.logger.log(file,'Successfully Executed selectDataFromCollection_into_csv() method of DB_Operation class of db_operation package')
            file.close()
        except Exception as ex:
            file = open("Training_Logs/General_Log.txt", 'a+')
            self.logger.log(file, 'Error while selecting data file and store it as csv.Error :: %s' % ex)
            file.close()









