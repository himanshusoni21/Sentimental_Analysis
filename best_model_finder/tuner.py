import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier

class Model_Tuner():
    def __init__(self,file_object,logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
        self.rf = RandomForestClassifier()
        self.mnb = MultinomialNB()
        self.svm = SVC()
        self.xg = XGBClassifier()
        self.bnb = BaggingClassifier

    def get_params_for_RandomForest(self,x_train,y_train):
        file = open('Training_Logs/General_Log.txt', 'a+')
        self.logger_object.log(file, 'Entered get_params_for_RandomForest() method of Model_Tuner class of best_model_finder package')
        file.close()

        self.logger_object.log(self.file_object, 'Entered the get_best_params_for_random_forest method')
        try:
            self.param_grid = {
                "n_estimators": [int(x) for x in np.linspace(start=100, stop=100, num=1)],
                # "criterion" : ['gini','entropy'],
                # "max_features" : ['auto','sqrt','log2',None],
                # "max_depth" : [int(x) for x in np.linspace(start=10,stop=16,num=3)],
                # "min_samples_split" : [int(x) for x in np.linspace(start=1,stop=5,num=3)],
                "min_samples_leaf": [int(x) for x in np.linspace(start=1, stop=2, num=1)]
            }
            self.random_search_rf = RandomizedSearchCV(estimator=self.rf,n_iter=2,param_distributions=self.param_grid,cv=2,verbose=3)
            self.random_search_rf.fit(x_train,y_train)

            # self.criterion = self.random_search.best_params_['criterion']
            self.n_estimators = self.random_search_rf.best_params_['n_estimators']
            # self.max_features = self.random_search.best_params_['max_features']
            # self.min_samples_split = self.random_search.best_params_['min_samples_split']
            # self.max_depth = self.random_search.best_params_['max_depth']
            self.min_samples_leaf = self.random_search_rf.best_params_['min_samples_leaf']

            self.rf = RandomForestClassifier(n_estimators=self.n_estimators,min_samples_leaf=self.min_samples_leaf)
            self.rf.fit(x_train,y_train)
            self.logger_object.log(self.file_object,'RandomForestClassifier Best Params ::' + str(self.random_search_rf.best_params_))

            file = open('Training_Logs/General_Log.txt', 'a+')
            self.logger_object.log(file,'Successfully Executed get_params_for_RandomForest() method of Model_Tuner class of best_model_finder package')
            file.close()
            return self.rf
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception Occured in get_best_params_for_randomForest ::%s' % str(e))
            self.logger_object.log(self.file_object,'RandomForest Parameter Tuning Failed.Exited !!')
            raise e

    def get_best_params_for_XGBoost(self, train_x, train_y):
        file = open('Training_Logs/General_Log.txt', 'a+')
        self.logger_object.log(file,'Entered get_params_for_XGBoost() method of Model_Tuner class of best_model_finder package')
        file.close()

        self.logger_object.log(self.file_object, 'Entered the get_best_params_for_xgboost method')
        try:
            self.param_grid_xgboost = {
                "n_estimators": [int(x) for x in np.linspace(start=100, stop=100, num=1)],
                # "booster" : ['gblinear','gbtree','dart'],
                # "eta" : [0.1,0.5,0.8],
                # "gamma" : [0,1,2],
                # "max_depth" : [int(x) for x in np.linspace(start=12,stop=16,num=3)],
                # "min_child_weight" : [int(x) for x in np.linspace(start=0,stop=9,num=3)],
                "max_delta_step": [int(x) for x in np.linspace(start=0, stop=6, num=2)]
            }

            self.random_search_xg = RandomizedSearchCV(estimator=self.xg, param_distributions=self.param_grid_xgboost,n_iter=2, cv=2, n_jobs=-1,verbose=100)
            self.random_search_xg.fit(train_x, train_y)

            self.n_estimators_xg = self.random_search_xg.best_params_['n_estimators']
            # self.booster_xg = self.grid_xg.best_params_['booster']
            # self.eta_xg = self.grid_xg.best_params_['eta']
            # self.gamma = self.grid_xg.best_params_['gamma']
            # self.max_depth_xg = self.grid_xg.best_params_['max_depth']
            # self.min_child_weight_xg = self.grid_xg.best_params_['min_child_weight']
            self.max_delta_step_xg = self.random_search_xg.best_params_['max_delta_step']

            self.xg = XGBClassifier(objective='multi:softmax', num_class=3 ,n_estimators=self.n_estimators_xg,max_delta_step=self.max_delta_step_xg)

            self.xg.fit(train_x, train_y)
            self.logger_object.log(self.file_object, 'XGBoost best params:' + str(self.random_search_xg.best_params_) + 'Exited the best_params_for_XGBoost')

            file = open('Training_Logs/General_Log.txt', 'a+')
            self.logger_object.log(file,'Successfully Executed get_params_for_XGBoost() method of Model_Tuner class of best_model_finder package')
            file.close()
            return self.xg
        except Exception as e:
            self.logger_object.log(self.file_object, 'Exception occured in get_best_params_xgboost :: %s' % (e))
            self.logger_object.log(self.file_object, 'XGBoost parameter Tuning Failed,Exited !!')
            raise e


    def get_params_svm(self,x_train,y_train):
        file = open('Training_Logs/General_Log.txt', 'a+')
        self.logger_object.log(file,'Entered get_params_for_svm() method of Model_Tuner class of best_model_finder package')
        file.close()
        try:
            # self.param_grid = {
            #     'C': [0.1],
            #     'gamma': [0.1],
            #     'kernel': ['rbf']
            # }
            # x_train = x_train.toarray()
            # self.random_search_svm = RandomizedSearchCV(estimator=self.svm,param_distributions=self.param_grid,n_iter=5,cv=2,verbose=3)
            # self.random_search_svm.fit(x_train,y_train)
            #
            # self.c = self.random_search_svm.best_params_['C']
            # self.gamma = self.random_search_svm.best_params_['gamma']
            # self.kernel =  self.random_search_svm.best_params_['kernel']
            # C=self.c,gamma=self.gamma,kernel=self.kernel
            self.svm = SVC(kernel='rbf')
            self.svm.fit(x_train,y_train)

            #self.logger_object.log(self.file_object,'SVM best params:' + str(self.random_search_svm.best_params_) + 'Exited the best_params_for_SVM')

            file = open('Training_Logs/General_Log.txt', 'a+')
            self.logger_object.log(file,'Successfully Executed get_params_for_SVM() method of Model_Tuner class of best_model_finder package')
            file.close()

            return self.svm
        except Exception as e:
            self.logger_object.log(self.file_object, 'Exception occured in get_best_params_SVM :: %s' % (e))
            self.logger_object.log(self.file_object, 'SVM parameter Tuning Failed,Exited !!')
            raise e


    def get_params_bagging_naive_bayes(self,x_train,y_train):
        file = open('Training_Logs/General_Log.txt', 'a+')
        self.logger_object.log(file,'Entered get_params_for_bagging_naive_bayes() method of Model_Tuner class of best_model_finder package')
        file.close()

        try:
            # self.param_grid = {
            #     'var_smoothing':[0.0001,0.00001,0.000001,None]
            # }
            # self.random_search_bnb = RandomizedSearchCV(estimator=BaggingClassifier(MultinomialNB,n_estimators=10),param_distributions=self.param_grid,n_iter=10,cv=2,verbose=3)
            # self.random_search_bnb.fit(x_train,y_train)
            #
            # self.var_smoothing = self.random_search_bnb.best_params_['var_smoothing']

            self.mnb = BaggingClassifier(base_estimator=MultinomialNB(),n_estimators=10)
            self.mnb.fit(x_train,y_train)

            #self.logger_object.log(self.file_object,'Bagging Naive Bayes best params:' + str(self.random_search_bnb.best_params_) + 'Exited the best_params_for_BaggingNaiveBayes')

            file = open('Training_Logs/General_Log.txt', 'a+')
            self.logger_object.log(file,'Successfully Executed get_params_for_BaggingNaiveBayes() method of Model_Tuner class of best_model_finder package')
            file.close()

            return self.mnb
        except Exception as e:
            self.logger_object.log(self.file_object, 'Exception occured in get_best_params_bagging_Naive_Bayes :: %s' % (e))
            self.logger_object.log(self.file_object, 'Bagging Naive Bayes parameter Tuning Failed,Exited !!')
            raise e



