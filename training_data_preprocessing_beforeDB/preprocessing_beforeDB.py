import pandas as pd
import os
from application_logging.logger import App_Logger

class preprocessing_beforeDB:
    def __init__(self):
        self.goodData_MainFile_path = "Training_Raw_Validated_File/Good_Raw_MainFile"
        self.goodData_AdditionalFile_path = "Training_Raw_Validated_File/Good_Raw_AdditionalFile"
        self.logger = App_Logger()

    def replaceMissingWithNull(self):
        file = open('Training_Logs/General_Log.txt', 'a+')
        self.logger.log(file,'Entered replaceMissingWithNull() method of preprocessing_beforeDB class of training_data_preprocessing_beforeDB package')
        file.close()
        try:
            f = open("Training_Logs/data_preprocessing_beforeDB.txt", "a+")
            only_files = [f for f in os.listdir(self.goodData_MainFile_path)]
            for file in only_files:
                csv = pd.read_csv(self.goodData_MainFile_path + "/" + file)
                csv.fillna('NULL',inplace=True)
                csv.to_csv(self.goodData_MainFile_path + "/" + file,index=None,header=True)
                self.logger.log(f,'Replace Missing values with Null Values in Good Raw Main File Successfully !!')
            f.close()

            file = open('Training_Logs/General_Log.txt', 'a+')
            self.logger.log(file,'Successfully Executed replaceMissingWithNull() method of preprocessing_beforeDB class of training_data_preprocessing_beforeDB package')
            file.close()
        except Exception as e:
            f = open("Training_Logs/data_preprocessing_beforeDB.txt", "a+")
            self.logger.log(f,'Replace missing with Null Values failed in Main File becasue:: %s' % str(e))
            f.close()