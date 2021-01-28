from datetime import datetime
import json
import os
import shutil
import pandas as pd
import re
from application_logging.logger import App_Logger

class Raw_Data_Validation:
    def __init__(self,file_path):
        self.batch_directory = file_path
        self.schema_path = 'schema_Training.json'
        self.logger = App_Logger()

    def fetch_values_from_schema(self):
        file = open('Training_Logs/General_Log.txt','a+')
        self.logger.log(file,'Entered fetch_values_from_schema() method of Raw_Data_Validation class of training_raw_data_validation package')
        file.close()
        try:
            with open(self.schema_path,'r') as r:
                dic = json.load(r)
                r.close()
            filename = dic['Sample_FileName']
            no_of_cols = dic['NumberOfColumns']
            column_name = dic['ColumnName']
            file = open('Training_Logs/General_Log.txt','a+')
            self.logger.log(file,'Successfully Executed fetch_values_from_schema() method of Raw_Data_Validation class of training_raw_data_validation package')
            file.close()
        except ValueError:
            file = open('Training_Logs/values_from_schema_Validation_Log.txt', 'a+')
            self.logger.log(file,'Value Error : Value not Found inside schema_Training.json')
            file.close()
            raise ValueError
        except KeyError:
            file = open('Training_Logs/values_from_schema_Validation_Log.txt', 'a+')
            self.logger.log(file,'Key Error : Key Value Error Incorrect Key Passed !!')
            file.close()
            raise KeyError
        except Exception as e:
            file = open('Training_Logs/General_Log.txt','a+')
            self.logger.log(file,'Error in fetch_values_from_schema() method of Raw_Data_Validation class of training_raw_data_validation package')
            file.close()
            file = open('Training_Logs/fetch_values_from_schema()','a+')
            self.logger.log('Error occured while loading json data.Error :: %s'%str(e))
            file.close()
            raise e
        return filename, no_of_cols, column_name

    def manualRegexCreation(self):
        regex = "['Reviews_']+['\_'']+[\d_]+[\d]+\.csv"
        return regex

    def createDirectoryFor_GoodBadRawData(self):
        file = open('Training_Logs/General_Log.txt', 'a+')
        self.logger.log(file,'Entered createDirectoryFor_GoodBadRawData() method of Raw_Data_Validation class of training_raw_data_validation package')
        file.close()
        try:
            path = os.path.join("Training_Raw_Validated_File/", "Good_Raw/")
            if not os.path.isdir(path):
                os.makedirs(path)

            path = os.path.join("Training_Raw_Validated_File/", "Bad_Raw/")
            if not os.path.isdir(path):
                os.makedirs(path)

            file = open('Training_Logs/General_Log.txt', 'a+')
            self.logger.log(file,'Successfully Executed createDirectoryFor_GoodBadRawData() method of Raw_Data_Validation class of training_raw_data_validation package')
            file.close()
        except OSError as ex:
            file = open("Training_Logs/GeneralLog.txt", 'a+')
            self.logger.log(file,'Error while creating MainFile Good and Bad Directory %s' % ex)
            file.close()
            raise OSError

    def deleteExistingGoodDataTrainingDir(self):
        file = open('Training_Logs/General_Log.txt', 'a+')
        self.logger.log(file,'Entered deleteExistingGoodDataTrainingDir() method of Raw_Data_Validation class of training_raw_data_validation package')
        file.close()
        try:
            path = "Training_Raw_Validated_File/"
            if os.path.isdir(path + 'Good_Raw/'):
                shutil.rmtree(path + 'Good_Raw/')
                file = open('Training_Logs/General_Log.txt','a+')
                self.logger.log(file,'Good Raw Main File Directory deleted Sucessfully !!!')
                file.close()
            file = open('Training_Logs/General_Log.txt', 'a+')
            self.logger.log(file,'Successfully Executed deleteExistingGoodDataTrainingDir() method of Raw_Data_Validation class of training_raw_data_validation package')
            file.close()
        except OSError as ex:
            file = open('Training_Logs/General_Log.txt','a+')
            self.logger.log(file,'Error while deleting Main File Good Raw Directory: %s' % ex)
            file.close()
            raise OSError

    def deleteExistingBadDataTrainingDir(self):
        try:
            file = open('Training_Logs/General_Log.txt', 'a+')
            self.logger.log(file,'Entered deleteExistingBadDataTrainingDir() method of Raw_Data_Validation class of training_raw_data_validation package')
            file.close()
            path = "Training_Raw_Validated_File/"
            if os.path.isdir(path + 'Bad_Raw/'):
                shutil.rmtree(path + 'Bad_Raw/')
                file = open('Training_Logs/General_Log.txt','a+')
                self.logger.log(file,'Bad Raw Additional Directory deleted Sucessfully !!!')
                file.close()
            file = open('Training_Logs/General_Log.txt', 'a+')
            self.logger.log(file,'Successfully Executed deleteExistingBadDataTrainingDir() method of Raw_Data_Validation class of training_raw_data_validation package')
            file.close()
        except OSError as ex:
            file = open('Training_Logs/General_Log.txt','a+')
            self.logger.log(file,'Error while deleting Main File Bad Raw Directory: %s' % ex)
            file.close()
            raise OSError

    def moveBadFilesToArchiveBad(self):
        file = open('Training_Logs/General_Log.txt', 'a+')
        self.logger.log(file,'Entered moveBadFilesTOArchiveBad() method of Raw_Data_Validation class of training_raw_data_validation package')
        file.close()
        now = datetime.now()
        date = now.date()
        time = now.strftime("%H%M%S")
        try:
            source = 'Training_Raw_Validated_File/BadRaw/'
            if os.path.isdir(source):
                path = "TrainingArchiveBadData"
                if not os.path.isdir(path):
                    os.makedirs(path)

                dest = 'TrainingArchiveBadData/BadData_' + str(date)+"_"+str(time)
                if not os.path.isdir(dest):
                    os.makedirs(dest)
                files = os.listdir(source)
                for f in files:
                    if f not in os.listdir(dest):
                        shutil.move(source + f, dest)
                file = open("Training_Logs/General_Log.txt", 'a+')
                self.logger.log(file,'Bad files moved to archive')
                path = 'Training_Raw_files_validated/'
                if os.path.isdir(path + 'Bad_Raw/'):
                    shutil.rmtree(path + 'Bad_Raw/')
                self.logger.log(file,'Bad Raw file Data Folder Deleted Successfully!!')
                self.logger.log(file,'Successfully Executed moveBadFilesToArchiveBad() method of Raw_Data_Validation class of training_raw_data_validation package')
                file.close()
        except Exception as e:
            file = open("Training_Logs/General_Log.txt", 'a+')
            self.logger.log(file, "Error while moving bad files to archive:: %s" % e)
            file.close()
            raise e

    def validateColumnLength(self,NumberofColumns):
        file = open('Training_Logs/General_Log.txt', 'a+')
        self.logger.log(file,'Entered validateColumnLength() method of Raw_Data_Validation class of training_raw_data_validation package')
        file.close()
        self.deleteExistingBadDataTrainingDir()
        self.deleteExistingGoodDataTrainingDir()
        self.createDirectoryFor_GoodBadRawData()
        try:
            f = open("Training_Logs/columnValidationLog.txt", 'a+')
            self.logger.log(f,"Column Length Validation Started!!")
            for file in os.listdir('Training_Batch_Files'):
                csv = pd.read_csv("Training_Batch_Files/" + file)
                if csv.shape[1] == NumberofColumns:
                    shutil.copy("Training_Batch_Files/" + file,"Training_Raw_Validated_File/Good_Raw")
                else:
                    shutil.copy("Training_Batch_Files/" + file, "Training_Raw_Validated_File/Bad_Raw")
                    self.logger.log(f, "Invalid Column Length for the file!! File moved to Bad Raw Folder :: %s" % file)
            self.logger.log(f, "Column Length Validation Completed!!")

            file = open('Training_Logs/General_Log.txt', 'a+')
            self.logger.log(file,'Successfully Executed createDirectoryFor_GoodBadRawData() method of Raw_Data_Validation class of training_raw_data_validation package')
            file.close()

        except OSError:
            f = open("Training_Logs/columnValidationLog.txt", 'a+')
            self.logger.log(f, "Error Occured while moving the file :: %s" % OSError)
            f.close()
            raise OSError
        except Exception as e:
            f = open("Training_Logs/columnValidationLog.txt", 'a+')
            self.logger.log(f, "Error Occured:: %s" % e)
            f.close()
            raise e
        f.close()


    