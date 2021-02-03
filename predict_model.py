from data_ingestion.data_loader_prediction import Data_Getter_Pred
from data_preprocessing.prediction_data_preprocessing import PreProcessor_Prediction
from file_operations.file_method_embeddings import File_Operation_Embedding
from file_operations.file_methods import File_Operation
from application_logging.logger import App_Logger
import os
import pandas as pd

class Predict_From_Model:
    def __init__(self):
        self.logger_object = App_Logger()
        self.file_object = open('Prediction_Logs/ModelPredictionLog.txt','a+')
        self.prediction_dir = 'PredictionFileTo_Predict'

    def predict_model(self):
        file = open('Prediction_Logs/General_Log.txt', 'a+')
        self.logger_object.log(file,'Entered train_model() method of Training_Model class')
        file.close()

        try:
            list_files = [i for i in os.listdir(self.prediction_dir)]
            preprocessor = PreProcessor_Prediction(self.file_object,self.logger_object)
            data_getter = Data_Getter_Pred(self.file_object,self.logger_object)
            for i in list_files:
                data = data_getter.get_data(i)
                raw_data = data.copy(deep=True)
                data = preprocessor.remove_null(data)
                data = preprocessor.clean_reviews(data)
                data = preprocessor.remove_StopWords(data)
                data = preprocessor.remove_punctuations(data)
                data = preprocessor.pos_tagging_lemmatizeText(data)
                data = preprocessor.count_vectorizer(data)
                data = preprocessor.tfidfTransformer_vectorizer(data)

                file_op = File_Operation(self.file_object,self.logger_object)
                model_name = file_op.find_correct_model_file()
                model_obj = file_op.load_model(model_name)
                prediction = list(model_obj.predict(data))
                polar_label = pd.Series(self.find_polar_prediction(prediction),name='label')
                data_with_label = pd.concat([raw_data,polar_label],axis=1)
                print(data_with_label.head(1))
                data_with_label.to_csv('Predicted_Files/' + i,header=True,mode='a+')
                self.logger_object.log(self.file_object,'Successfull End of Prediction !!!')
            return 1
        except Exception as e:
            self.logger_object.log(self.file_object, 'Unsuccessfull End of Prediction !!!')
            self.file_object.close()
            raise e

    def find_polar_prediction(self,pred_list):
        polar = list()
        for i in pred_list:
            if i == 2:
                polar.append('Positive')
            elif i == 1:
                polar.append('Neutral')
            elif i == 0:
                polar.append('Negative')
        return polar


