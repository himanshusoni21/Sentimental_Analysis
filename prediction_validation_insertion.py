from prediction_raw_data_validation.raw_validation_prediction import Raw_Data_Validation_Prediction
from prediction_db_operations.db_operation_prediction import DB_Operations_Prediction
from prediction_data_preprocessing_beforeDB.prediction_preprocessing_beforeDB import preprocessing_beforeDB_prediction
from application_logging.logger import App_Logger

class Predict_Validation:
    def __init__(self,batch_file_path,listoffile_predict):
        self.raw_data = Raw_Data_Validation_Prediction(batch_file_path)
        self.listoffile_predict = listoffile_predict
        self.db_operation = DB_Operations_Prediction
        self.preprocess_beforeDB = preprocessing_beforeDB_prediction()
        self.file_object = open('Prediction_Logs/Prediction_Validation_Log.txt','a+')
        self.logger_object = App_Logger()

    def prediction_validation(self):
        try:
            self.logger_object.log(self.file_object,'Start of Raw Data Validation on Files !!')
            filename, no_of_cols, column_name = self.raw_data.fetch_values_from_schema()
            self.raw_data.deleteExistingBadDataTrainingDir()
            self.raw_data.deleteExistingGoodDataTrainingDir()
            self.raw_data.createDirectoryFor_GoodBadRawData()
            for f in self.listoffile_predict:
            #regex = self.raw_data.manualRegexCreation()
                self.raw_data.validateColumnLength(f,no_of_cols)
                self.logger_object.log(self.file_object,'Raw Data Column Length Validation Completed Successfully of' + str(f) + '!!')
                self.logger_object.log(self.file_object,'Start of Data Preprocessing before DB for ::' + str(f) + ' File')
                self.preprocess_beforeDB.replaceMissingWithNull(f)
                self.logger_object.log(self.file_object,'Data Preprocessing before DB Completed of File' + str(f) + ' !!')
                # self.logger_object.log(self.file_object,'Start of Database Creation,Collection Creation,Insertion')
                # db_object = self.db_operation.create_db_connection('Sentimental_Analysis')
                # collection_object = self.db_operation.create_collection(db_object)
                # self.logger_object.log(self.file_object,'Creation of Database Completed Successfully !!')
                # self.db_operation.insertion_GoodRawData_into_collection(collection_object)
                # self.logger_object.log(self.file_object,'Insertion of document into collection completed Successfully !!')

                #self.logger_object.log(self.file_object,'Start of Delete existing good and bad raw data training directory')
                # self.raw_data.deleteExistingGoodDataTrainingDir()
                # self.raw_data.deleteExistingBadDataTrainingDir()
                #self.logger_object.log(self.file_object, 'Delete existing good and bad raw data successfully !!')
            self.logger_object.log(self.file_object, 'Start of Moving Bad Raw data to ArchiveBad')
            self.raw_data.moveBadFilesToArchiveBad()
            self.logger_object.log(self.file_object,'Moving bad files to ArchiveBad and BadRaw directory deleted successfully !!')

            self.logger_object.log(self.file_object,'Moving Good Data File to PredictionFileTo_Predict Dir Started')
            self.raw_data.moveGoodFileTo_PredictDir()
            self.logger_object.log(self.file_object,'Moving Good Data File to PredictionFileTo_Predict Successfully Completed !!')
            # self.logger_object.log(self.file_object,'Start of Selection of data from collection and Export as CSV File')
            # self.db_operation.selectDataFromCollection_into_csv(collection_object)
            # self.logger_object.log(self.file_object,'Selection of Data from collection and Exported as CSV file completed Successfully !!')
            self.logger_object.log(self.file_object,'Prediction Raw Data Validation Successfully Completed !!!')

        except Exception as e:
            raise e