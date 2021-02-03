import os
import pickle
import shutil

class File_Operation_Embedding:
    def __init__(self,file_object,logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
        self.embedding_dir ='Embeddings_Object/'

    def save_model(self, embedding,filename):
        file = open('Training_Logs/General_Log.txt', 'a+')
        self.logger_object.log(file,'Entered save_model() method of file_operations_embedding class of file_operations package')
        full_filename = None
        try:
            path = os.path.join(self.embedding_dir,filename)
            if os.path.isdir(path):
                shutil.rmtree(self.embedding_dir)
                os.makedirs(path)
            else:
                os.makedirs(path)

            with open(path + '/' + filename + '.pkl','wb') as f:
                pickle.dump(embedding,f)

            self.logger_object.log(file,'Embedding File : ' + filename + ' Saved.')
            self.logger_object.log(file,'Successfully Executed save_model() method of file_operations_embedding class of file_operation package')
            file.close()
            return 'success'
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception Occured in save_model method of file_operation_embedding class %s' % str(e))
            self.logger_object.log(self.file_object,'Embedding File could not be saved')
            raise e

    def load_model(self,embedding_type):
        file = open('Prediction_Logs/General_Log.txt', 'a+')
        self.logger_object.log(file, 'Entered load_model() method of file_operations_embedding class of file_operations package')
        try:
            with open(self.embedding_dir + embedding_type + '/' + embedding_type + '.pkl','rb') as f:
                self.logger_object.log(self.file_object,'Embedding File' + embedding_type + 'Loaded Successfully')
                return pickle.load(f)
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in load model method of model finder class:: %s' %str(e))
            self.logger_object.log(self.file_object,'Embedding File ' + embedding_type + ' could not be loaded ')
            raise e

