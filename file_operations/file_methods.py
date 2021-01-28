import os
import pickle
import shutil

class File_Operation:
    def __init__(self,file_object,logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
        self.model_directory ='models/'

    def save_model(self, model,filename):
        file = open('Training_Logs/General_Log.txt', 'a+')
        self.logger_object.log(file,'Entered save_model() method of file_operations class of file_operations package')
        full_filename = None
        try:
            if filename == 'nb':
                full_filename = 'GaussianNB'
            elif filename == 'rf':
                full_filename = 'RandomForest'
            elif filename == 'xg':
                full_filename = 'XGBoost'
            elif filename == 'nbn':
                full_filename = 'BaggingGaussianNB'

            path = os.path.join(self.model_directory,full_filename)
            if os.path.isdir(path):
                shutil.rmtree(self.model_directory)
                os.makedirs(path)
            else:
                os.makedirs(path)

            with open(path + '/' + full_filename + '.sav','wb') as f:
                pickle.dump(model,f)

            self.logger_object.log(open('Training_Logs/ModelTrainingLog.txt'),'Model File : ' + full_filename + ' Saved.')
            self.logger_object.log(file,'Successfully Executed save_model() method of file_operations class of file_operation package')
            file.close()
            return 'success'
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception Occured in save_model method of file_operation class %s' % str(e))
            self.logger_object.log(self.file_object,'Model File could not be saved')
            raise e

    def load_model(self,filename):
        file = open('Training_Logs/General_Log.txt', 'a+')
        self.logger_object.log(file, 'Entered load_model() method of file_operations class of file_operations package')
        try:
            with open(self.model_directory + filename + '/' + filename + '.sav','rb') as f:
                self.logger_object.log(self.file_object,'Model File' + filename + 'Loaded Successfully')
                return pickle.load(f)
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in load model method of model finder class:: %s' %str(e))
            self.logger_object.log(self.file_object,'Model File ' + filename + ' could not be loaded ')
            raise e


    def find_correct_model_file(self):
        self.logger_object.log(self.file_object,'Entered the find_correct_model_file method of the File_Operation class')
        try:
            #self.folder_name = self.model_directory
            model_dir = os.listdir(self.model_directory)
            model_name = os.listdir(self.model_directory + model_dir[0])
            model_name = model_name[0].split('.')[0]
            self.logger_object.log(self.file_object,'Exited the find_correct_model_file method of FileMethods Package')
            return model_name
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception Occured in finding_correct_model_file method :: %s' % str(e))
            self.logger_object.log(self.file_object,'Exited the find_correct_model_file method of FileMethods Package')
            raise e