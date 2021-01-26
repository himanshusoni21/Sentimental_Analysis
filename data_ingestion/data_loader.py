import pandas as pd

class Data_Getter:
    def __init__(self,file_object,log_object):
        self.training_file ='TrainingFileFromDB/InputFile.csv'
        self.file_object = file_object
        self.logger_object = log_object

    def get_data(self):
        self.logger_object.log(self.file_object,'Entered get_data() method of Data_Getter Class')
        try:
            self.data = pd.read_csv(self.training_file)
            self.logger_object.log(self.file_object,'Data Load Successfully.Exited the get_data() method of Data_Getter Class')
            return self.data
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in get_data() method of Data_Getter Class.Exception Message::%s' %str(e))
            self.logger_object.log(self.file_object,'Data load unsuccesfull!!! ')
            raise e
