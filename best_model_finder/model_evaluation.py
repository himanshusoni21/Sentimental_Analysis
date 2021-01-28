import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,f1_score

class Model_Evaluation:
    def __init__(self,trained_model,x_test,y_test,file_object,logger_object):
        self.trained_model = trained_model
        self.x_test = x_test
        self.y_test = y_test
        self.logger_object = logger_object
        self.file_object = file_object
        self.model_evaluation_dict = {
            'nb': {'accuracy_score': None, 'confusion_matrix': None, 'precision_score': None, 'recall_score': None,
                   'f1_score': 0},
            'rf': {'accuracy_score': None, 'confusion_matrix': None, 'precision_score': None, 'recall_score': None,
                   'f1_score': 0},
            'xg': {'accuracy_score': None, 'confusion_matrix': None, 'precision_score': None, 'recall_score': None,
                   'f1_score': 0},
            'bnb': {'accuracy_score': None, 'confusion_matrix': None, 'precision_score': None, 'recall_score': None,
                    'f1_score': 0}
        }


    def get_accuracy_score(self,model):
        file = open('Training_Logs/General_Log.txt', 'a+')
        self.logger_object.log(file,'Entered get_accuracy_score() method for '+ str(model) + ' of Model_Tuner class of best_model_finder package')
        file.close()

        try:
            y_predict = model.predict(self.x_test)
            acc_score = accuracy_score(self.y_test,y_predict)
            file = open('Training_Logs/General_Log.txt', 'a+')
            self.logger_object.log(file,'Successfully Executed get_accuracy_score() for' +str(model)+ ' method of Model_Tuner class of best_model_finder package')
            file.close()
            return acc_score
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception Occured in get_accuracy_score for ' + str(model) + ' ::%s' % str(e))
            self.logger_object.log(self.file_object, 'get_accuracy_score() method .Exited !!')
            raise e

    def get_confusion_matrix(self, model):
        file = open('Training_Logs/General_Log.txt', 'a+')
        self.logger_object.log(file, 'Entered get_confusion_matrix() method for ' + str(model) + ' of Model_Finder class of best_model_finder package')
        file.close()

        try:
            y_predict = model.predict(self.x_test)
            cfn_matrix = confusion_matrix(self.y_test, y_predict)
            file = open('Training_Logs/General_Log.txt', 'a+')
            self.logger_object.log(file, 'Successfully Executed get_confusion_matrix() for' + str(
                model) + ' method of Model_Finder class of best_model_finder package')
            file.close()
            return cfn_matrix
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception Occured in get_confusion_matrix for ' + str(model) + ' ::%s' % str(e))
            self.logger_object.log(self.file_object, 'get_confusion_matrix() method .Exited !!')
            raise e

    def get_precision_score(self,model):
        file = open('Training_Logs/General_Log.txt', 'a+')
        self.logger_object.log(file,'Entered get_precision_score() method for '+ str(model) + ' of Model_Finder class of best_model_finder package')
        file.close()

        try:
            y_predict = model.predict(self.x_test)
            prec_score = precision_score(self.y_test,y_predict,average='micro')
            file = open('Training_Logs/General_Log.txt', 'a+')
            self.logger_object.log(file,'Successfully Executed get_precision_score() for ' +str(model)+ ' method of Model_Tuner class of best_model_finder package')
            file.close()
            return prec_score
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception Occured in get_precision_score for ' + str(model) + ' ::%s' % str(e))
            self.logger_object.log(self.file_object, 'get_precision_score() method .Exited !!')
            raise e

    def get_recall_score(self,model):
        file = open('Training_Logs/General_Log.txt', 'a+')
        self.logger_object.log(file,'Entered get_recall_score() method for '+ str(model) + ' of Model_Finder class of best_model_finder package')
        file.close()

        try:
            y_predict = model.predict(self.x_test)
            reca_score = recall_score(self.y_test,y_predict,average='micro')
            file = open('Training_Logs/General_Log.txt', 'a+')
            self.logger_object.log(file,'Successfully Executed get_recall_score for ' +str(model)+ ' method of Model_Finder class of best_model_finder package')
            file.close()
            return reca_score
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception Occured in get_recall_score for ' + str(model) + ' ::%s' % str(e))
            self.logger_object.log(self.file_object, 'get_recall_score() method .Exited !!')
            raise e

    def get_f1_score(self,model):
        file = open('Training_Logs/General_Log.txt', 'a+')
        self.logger_object.log(file,'Entered get_f1_score() method for '+ str(model) + ' of Model_Finder class of best_model_finder package')
        file.close()

        try:
            y_predict = model.predict(self.x_test)
            f1_sc = f1_score(self.y_test,y_predict,average='micro')
            file = open('Training_Logs/General_Log.txt', 'a+')
            self.logger_object.log(file,'Successfully Executed get_f1_score for ' +str(model)+ ' method of Model_Finder class of best_model_finder package')
            file.close()
            return f1_sc
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception Occured in get_f1_score for ' + str(model) + ' ::%s' % str(e))
            self.logger_object.log(self.file_object, 'get_f1_score() method .Exited !!')
            raise e

    def generate_models_evaluation_report_dict(self,trained_models_dict):
        file = open('Training_Logs/General_Log.txt', 'a+')
        self.logger_object.log(file, 'Entered generate_models_evaluation_report_dict() method of Model_Evaluation class of model_evaluation package')
        file.close()

        try:
            for model,object in zip(trained_models_dict.keys(),trained_models_dict.values()):
                if object is not None:
                    self.model_evaluation_dict[model]['accuracy_score'] = self.get_accuracy_score(object)
                    self.model_evaluation_dict[model]['confusion_matrix'] = self.get_confusion_matrix(object)
                    self.model_evaluation_dict[model]['precision_score'] = self.get_precision_score(object)
                    self.model_evaluation_dict[model]['recall_score'] = self.get_recall_score(object)
                    self.model_evaluation_dict[model]['f1_score'] = self.get_f1_score(object)

            file = open('Training_Logs/General_Log.txt', 'a+')
            self.logger_object.log(file, 'Successfully Executed generate_model_evaluation_report_dict() for method of Model_Evaluation class of model_evaluation package')
            file.close()

            self.logger_object.log(self.file_object,'Model Evaluation Report is :: %s' %str(self.model_evaluation_dict))
            return self.model_evaluation_dict
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception Occured in generate_model_evaluation_report_dict() ')
            self.logger_object.log(self.file_object, 'generate_models_evaluation_report_dict() method .Exited !!')
            raise e


