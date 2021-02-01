from training_raw_data_validation.raw_validation import Raw_Data_Validation
from training_db_operations.db_operations import DB_Operations
from training_data_preprocessing_beforeDB.preprocessing_beforeDB import preprocessing_beforeDB
from application_logging.logger import App_Logger

class Train_Validation:
    def __init__(self,batch_file_path):
        self.raw_data = Raw_Data_Validation(batch_file_path)
        self.db_operation = DB_Operations()
        self.preprocess_beforeDB = preprocessing_beforeDB()
        self.file_object = open('Training_Logs/Training_Validation_Log.txt','a+')
        self.logger_object = App_Logger()

    def training_validation(self):
        try:
            self.logger_object.log(self.file_object,'Start of Raw Data Validation on Files !!')
            filename, no_of_cols, column_name = self.raw_data.fetch_values_from_schema()
            regex = self.raw_data.manualRegexCreation()
            self.raw_data.validateColumnLength(no_of_cols)
            self.logger_object.log(self.file_object,'Raw Data Validation Completed Successfully !!')

            self.logger_object.log(self.file_object,'Start of Data Preprocessing before DB')
            self.preprocess_beforeDB.replaceMissingWithNull()
            self.logger_object.log(self.file_object,'Data Preprocessing before DB Completed !!')

            self.logger_object.log(self.file_object,'Start of Database Creation,Collection Creation,Insertion')
            db_object = self.db_operation.create_db_connection('Sentimental_Analysis')
            collection_object = self.db_operation.create_collection(db_object)
            self.logger_object.log(self.file_object,'Creation of Database Completed Successfully !!')
            self.db_operation.insertion_GoodRawData_into_collection(collection_object)
            self.logger_object.log(self.file_object,'Insertion of document into collection completed Successfully !!')

            self.logger_object.log(self.file_object,'Start of Delete existing good and bad raw data training directory')
            # self.raw_data.deleteExistingGoodDataTrainingDir()
            # self.raw_data.deleteExistingBadDataTrainingDir()
            self.logger_object.log(self.file_object, 'Delete existing good and bad raw data successfully !!')
            self.logger_object.log(self.file_object, 'Start of Moving Bad Raw data to ArchiveBad')
            self.raw_data.moveBadFilesToArchiveBad()
            self.logger_object.log(self.file_object,'Moving bad files to ArchiveBad and BadRaw directory deleted successfully !!')

            self.logger_object.log(self.file_object,'Start of Selection of data from collection and Export as CSV File')
            self.db_operation.selectDataFromCollection_into_csv(collection_object)
            self.logger_object.log(self.file_object,'Selection of Data from collection and Exported as CSV file completed Successfully !!')
            self.logger_object.log(self.file_object,'Raw Data Validation Successfully Completed !!!')

        except Exception as e:
            raise e


