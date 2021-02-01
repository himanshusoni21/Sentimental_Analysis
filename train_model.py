from sklearn.model_selection import train_test_split
from data_ingestion.data_loader import Data_Getter
from data_preprocessing.data_preprocessing import PreProcessor
from file_operations.file_methods import File_Operation
from application_logging.logger import App_Logger
from best_model_finder.tuner import Model_Tuner
from best_model_finder.model_evaluation import Model_Evaluation

class Training_Model:
    def __init__(self,models_list,sampling_method,):
        self.logger_object = App_Logger()
        self.file_object = open('Training_Logs/ModelTrainingLog.txt','a+')
        self.sampling_method = sampling_method
        self.models_list = models_list

    def train_model(self):
        file = open('Training_Logs/General_Log.txt', 'a+')
        self.logger_object.log(file,'Entered train_model() method of Training_Model class')
        file.close()

        try:
            data_getter = Data_Getter(self.file_object,self.logger_object)
            data = data_getter.get_data()

            preprocessor = PreProcessor(self.file_object,self.logger_object,self.sampling_method)
            data = preprocessor.remove_null(data)
            data = preprocessor.clean_reviews(data)
            data = preprocessor.remove_StopWords(data)
            data = preprocessor.remove_punctuations(data)
            data = preprocessor.pos_tagging_lemmatizeText(data)
            data = preprocessor.encode_label(data)
            x,y = preprocessor.separate_feature_label(data)
            x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=100)
            x_train,x_test = preprocessor.count_vectorizer(x_train,x_test)
            #x_train,x_test = preprocessor.tfidf_vectorizer(x_train,x_test)
            x_train,x_test = preprocessor.tfidfTransformer_vectorizer(x_train,x_test)

            if self.sampling_method == 'us':
                x_train,y_train = preprocessor.under_sampling(x_train,y_train)
            elif self.sampling_method == 'os':
                x_train,y_train = preprocessor.over_sampling(x_train,y_train)
            elif self.sampling_method == 'no':
                pass
            else:
                pass

            tuner = Model_Tuner(self.file_object,self.logger_object)
            self.trained_models_dict = {'svm':None,'rf':None,'xg':None,'bnb':None}

            for m in self.models_list:
                if m == 'svm':
                    self.trained_models_dict['nb'] = tuner.get_params_svm(x_train,y_train)
                elif m == 'rf':
                    self.trained_models_dict['rf'] = tuner.get_params_for_RandomForest(x_train,y_train)
                elif m == 'xg':
                    self.trained_models_dict['xg'] = tuner.get_best_params_for_XGBoost(x_train,y_train)
                elif m == 'bnb':
                    self.trained_models_dict['bnb'] = tuner.get_params_bagging_naive_bayes(x_train,y_train)
                else:
                    pass

            model_evaluation = Model_Evaluation(self.trained_models_dict,x_test,y_test,self.file_object,self.logger_object)
            self.model_evaluation_report_dict =  model_evaluation.generate_models_evaluation_report_dict(self.trained_models_dict)
            self.ordered_model_evaluation_report_dict = sorted(self.model_evaluation_report_dict.items(),key=lambda x:x[1]['f1_score'],reverse=True)

            for m in self.ordered_model_evaluation_report_dict:
                model_to_save = m[0]
                break;

            file_operation = File_Operation(self.file_object,self.logger_object)
            is_model_saved = file_operation.save_model(self.trained_models_dict[model_to_save],model_to_save)

            if (is_model_saved == 'success'):
                self.logger_object.log(self.file_object, 'Successfull End of Training')
            else:
                self.logger_object.log(self.file_object, 'Error while saving model to models directory')
            return is_model_saved
        except Exception as e:
            self.logger_object.log(self.file_object, 'Unsuccessfull End of Training')
            self.file_object.close()
            raise e


