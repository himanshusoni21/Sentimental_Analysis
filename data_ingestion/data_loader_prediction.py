import pandas as pd
import os

class Data_Getter_Pred:
    def __init__(self, file_object, logger_object):
        self.prediction_dir='PredictionFileTo_Predict/'
        self.file_object=file_object
        self.logger_object=logger_object

    def get_data(self,f):
        self.logger_object.log(self.file_object, 'Entered the get_data() method of the Data_Getter_Pred class of data_loader_prediction package.')
        try:
            self.data = pd.read_csv(self.prediction_dir + f)
            self.logger_object.log(self.file_object,'Data Load Successful.Exited the get_data() method of the Data_Getter_Pred class')
            return self.data
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_data() method of the Data_Getter_Pred class. Exception message: ' + str(e))
            self.logger_object.log(self.file_object,'Data Load Unsuccessful.Exited the get_data method of the Data_Getter class')
            raise e