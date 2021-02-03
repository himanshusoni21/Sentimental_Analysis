import string
import pandas as pd
import re
import nltk
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords,wordnet
from file_operations.file_method_embeddings import File_Operation_Embedding

class PreProcessor_Prediction:
    def __init__(self,file_object,logger_object):
        self.file_object = file_object
        self.logger_object = logger_object

    def remove_null(self,data):
        file = open('Prediction_Logs/General_Log.txt', 'a+')
        self.logger_object.log(file,'Entered remove_null() method of PreProcessor_Prediction class of data_preprocessing package')
        file.close()

        try:
            self.is_null_present = False
            self.data2 = pd.DataFrame()
            null_count = data.isnull().sum()
            for i in null_count:
                if i > 0:
                    self.is_null_present = True
                    break

            if(self.is_null_present):
                is_null = data.isnull()
                row_has_null = is_null.any(axis=1)
                data_with_null = data[row_has_null]
                data_with_null.to_csv('PredictionArchiveBadData/null_values_rows.csv');
                self.data2 = data.dropna()

            file = open('Prediction_Logs/General_Log.txt', 'a+')
            self.logger_object.log(file,'Successfully Executed remove_null() method of PreProcessing class of data_preprocessing package')
            file.close()
            return data

        except Exception as e:
            self.logger_object.log(self.file_object,'Exception Occured while performing is_null_present method %s' % str(e))
            self.logger_object.log(self.file_object, 'Removing Null Values Failed due to Exception occured')
            raise e


    def clean_reviews(self, data):
        file = open('Prediction_Logs/General_Log.txt', 'a+')
        self.logger_object.log(file, 'Entered clearn_reviews() method of PreProcessor_Prediction class of data_preprocessing package')
        file.close()
        self.data = data
        try:
            def cleanReviews_regex(x):
                x = x.lower()
                x = re.sub('[^\w\s]', ' ', x)
                x = re.sub('[\d+]', ' ', x)
                x = x.strip()
                return x

            self.data['comment'] = self.data['comment'].apply(lambda x:cleanReviews_regex(x))

            self.logger_object.log(self.file_object,'Cleaning of review Succesfull!!. Exited to the clean_reviews of the PreProcessor_Prediction class of data_preprocessing package')

            file = open('Prediction_Logs/General_Log.txt', 'a+')
            self.logger_object.log(file,'Successfully Executed remove_null() method of PreProcessor_Prediction class of data_preprocessing package')
            file.close()
            return self.data
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in cleanReview method of the PreProcessor_Prediction class. Exception message::  ' + str(e))
            self.logger_object.log(self.file_object,'Review Cleaning Unsuccessfull. Exited the cleanReview method of the PreProcessor_Prediction class')
            raise Exception()

    def remove_StopWords(self, data):
        file = open('Prediction_Logs/General_Log.txt', 'a+')
        self.logger_object.log(file, 'Entered remove_StopWords_reviews() method of PreProcessor_Prediction class of data_preprocessing package')
        file.close()

        self.data = data
        try:
            self.stop_words = stopwords.words('english')
            self.wanted = ["ain", "aren", "aren't", "couldn", "couldn't",
                           "didn", "didn't", "doesn", "doesn't", "hadn", "hadn't", "hasn",
                           "hasn't", "haven", "haven't", "isn", "isn't", "mightn", "mightn't",
                           "mustn", "mustn't", "needn", "needn't", "shan", "shan't", "shouldn",
                           "shouldn't", "won", "wasn", "wasn't", "weren", "weren't", "won't",
                           "wouldn", "wouldn't", "should", "should've", "no", "nor", "not", "very"]

            for word in self.stop_words:
                if word in self.wanted:
                    self.stop_words.remove(word)

            self.data['comment'] = self.data['comment'].apply(lambda x: " ".join(x for x in x.split() if x not in self.stop_words))
            self.logger_object.log(self.file_object,'Remove StopWords Succesfull. Exited to the removeStopWords of the PreProcessor_Prediction class')

            file = open('Prediction_Logs/General_Log.txt', 'a+')
            self.logger_object.log(file,'Successfully Executed remove_StopWords() method of PreProcessor_Prediction class of data_preprocessing package')
            file.close()
            return self.data
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in remove_StopWords method of the PreProcessor_Prediction class. Exception message::  ' + str(e))
            self.logger_object.log(self.file_object,'Remove StopWords Unsuccessful. Exited the removeStopWords method of the PreProcessor_Prediction class')
            raise Exception()


    def remove_punctuations(self,data):
        file = open('Prediction_Logs/General_Log.txt', 'a+')
        self.logger_object.log(file,'Entered remove_punctuations() method of PreProcessor_Prediction class of data_preprocessing package')
        file.close()

        self.data = data
        try:
            def remove_punct(text):
                translator = str.maketrans('', '', string.punctuation)
                return text.translate(translator)

            self.data['comment'] = data['comment'].apply(lambda x:remove_punct(x))
            file = open('Prediction_Logs/General_Log.txt', 'a+')
            self.logger_object.log(file,'Successfully Executed remove_punctuations() method of PreProcessor_Prediction class of data_preprocessing package')
            file.close()

            return self.data
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in remove_punctuations() method of the PreProcessor_Prediction class. Exception message::  ' + str(e))
            self.logger_object.log(self.file_object,'Remove Punctuations Unsuccessful. Exited the remove_punctuations method of the PreProcessor_Prediction class')
            raise e

    def pos_tagging_lemmatizeText(self,data):
        file = open('Prediction_Logs/General_Log.txt', 'a+')
        self.logger_object.log(file,'Entered pos_tagging_lemmatizeText() method of PreProcessor_Prediction class of data_preprocessing package')
        file.close()

        self.data = data
        try:
            def to_WordNet(pos_tag):
                if pos_tag.startswith('J'):
                    return wordnet.ADJ
                elif pos_tag.startswith('V'):
                    return wordnet.VERB
                elif pos_tag.startswith('N'):
                    return wordnet.NOUN
                elif pos_tag.startswith('R'):
                    return wordnet.ADV
                else:
                    return None

            def postagging_lemmatize(x):
                x = word_tokenize(x)
                pos_tagged = nltk.pos_tag(x)
                wordnet_tagged = map(lambda x: (x[0], to_WordNet(x[1])), pos_tagged)
                lemm = WordNetLemmatizer()
                lemmatize_sentence = list()
                for word, tag in wordnet_tagged:
                    if tag is None:
                        lemmatize_sentence.append(word)
                    else:
                        lemmatize_sentence.append(lemm.lemmatize(word, tag))
                return str(lemmatize_sentence)

            self.data['comment'] = data['comment'].apply(lambda x:postagging_lemmatize(x))
            file = open('Prediction_Logs/General_Log.txt', 'a+')
            self.logger_object.log(file,'Successfully Executed pos_tagging_lemmatizeText() method of PreProcessor_Prediction class of data_preprocessing package')
            file.close()
            return self.data
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in pos_tagging_lemmatizeText() method of the PreProcessor_Prediction class. Exception message::  ' + str(e))
            self.logger_object.log(self.file_object,'Remove Punctuations Unsuccessful. Exited the pos_tagging_lemmatizeText method of the PreProcessor_Prediction class')
            raise e

    # def encode_label(self, data):
    #     file = open('Prediction_Logs/General_Log.txt', 'a+')
    #     self.logger_object.log(file,'Entered polarize_Rating() method of PreProcessor_Prediction class of data_preprocessing package')
    #     file.close()
    #
    #     self.data = data
    #     try:
    #         self.logger_object.log(self.file_object,'Class Label unique value before label encoding :: %s'%self.data['label'].unique())
    #         le = LabelEncoder()
    #         self.data['label'] = le.fit_transform(data['label'])
    #         self.logger_object.log(self.file_object,'Class Label unique value after label encoding :: %s'%self.data['label'].unique())
    #
    #         file = open('Prediction_Logs/General_Log.txt', 'a+')
    #         self.logger_object.log(file,'Successfully Executed encode_label() method of PreProcessor_Prediction class of data_preprocessing package')
    #         file.close()
    #
    #         return data
    #     except Exception as e:
    #         self.logger_object.log(self.file_object,'Exception occured in encode_label() method of the PreProcessor_Prediction class. Exception message:  ' + str(e))
    #         self.logger_object.log(self.file_object,'Encoding Label Unsuccessful. Exited the encode_label() method of the PreProcessor_Prediction class')
    #         raise Exception()


    def count_vectorizer(self,data):
        file = open('Prediction_Logs/General_Log.txt', 'a+')
        self.logger_object.log(file,'Entered count_vectorizer() method of PreProcessor_Prediction class of data_preprocessing package')
        file.close()

        self.data = data
        try:
            file_op = File_Operation_Embedding(self.file_object,self.logger_object)
            cv_obj = file_op.load_model('CountVectorizer')
            self.data = cv_obj.transform(self.data['comment'])

            if(cv_obj is not None):
                self.logger_object.log(self.file_object,'CountVectorizer Object Loaded Successfully')
            else:
                self.logger_object.log(self.file_object,'Error while Loading CountVectorizer Object')

            file = open('Prediction_Logs/General_Log.txt', 'a+')
            self.logger_object.log(file,'Successfully Executed count_vectorizer() method of PreProcessor_Prediction class of data_preprocessing package')
            file.close()

            return self.data
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in count_vectorizer() method of the PreProcessor_Prediction class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,'Count Vectorizer Unsuccessful. Exited the count_vectorizer() method of the PreProcessor_Prediction class')
            raise Exception()

    def tfidf_vectorizer(self,x_train,x_test):
        file = open('Prediction_Logs/General_Log.txt', 'a+')
        self.logger_object.log(file,'Entered tfidf_vectorizer() method of PreProcessor_Prediction class of data_preprocessing package')
        file.close()

        self.x_train = x_train
        self.x_test = x_test
        try:
            tfidf = TfidfVectorizer(ngram_range=(2,2))
            self.x_train = tfidf.fit_transform(self.x_train)
            self.x_test = tfidf.transform(self.x_test)

            file_op = File_Operation_Embedding(self.file_object, self.logger_object)
            if (file_op.save_model(tfidf, 'TfidfVectorizer')):
                self.logger_object.log(self.file_object, 'TfidfVectorizer Object saved Successfully')

            file = open('Prediction_Logs/General_Log.txt', 'a+')
            self.logger_object.log(file,'Successfully Executed tfidf_vectorizer() method of PreProcessor_Prediction class of data_preprocessing package')
            file.close()

            return x_train,x_test
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in tfidf_vectorizer() method of the PreProcessor_Prediction class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,'Tfidf Vectorizer Unsuccessful. Exited the tfidfTransformer_vectorizer() method of the PreProcessor_Prediction class')
            raise Exception()

    def tfidfTransformer_vectorizer(self,data):
        file = open('Prediction_Logs/General_Log.txt', 'a+')
        self.logger_object.log(file,'Entered tfidfTransformer_vectorizer() method of PreProcessor_Prediction class of data_preprocessing package')
        file.close()
        self.data = data
        try:
            file_op = File_Operation_Embedding(self.file_object, self.logger_object)
            tfidf_obj = file_op.load_model('TfidfTransformerVectorizer')
            self.data = tfidf_obj.transform(self.data['comment'])

            if (tfidf_obj is not None):
                self.logger_object.log(self.file_object, 'TfidfTransformerVectorizer Object load Successfully')
            else:
                self.logger_object.log(self.file_object,'Error while Loading TfidfTransformerVectorizer Object !!')

            file = open('Prediction_Logs/General_Log.txt', 'a+')
            self.logger_object.log(file,'Successfully Executed tfidfTransformer_vectorizer() method of PreProcessor_Prediction class of data_preprocessing package')
            file.close()

            return data
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in tfidfTransformer_vectorizer() method of the PreProcessor_Prediction class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,'Count Vectorizer Unsuccessful. Exited the tfidfTransformer_vectorizer() method of the PreProcessor_Prediction class')
            raise Exception()
