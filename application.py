from wsgiref import simple_server
from flask import Flask,render_template,request,Response,send_file
import os
from flask_cors import cross_origin,CORS
import flask_monitoringdashboard as dashboard
from training_validation_insertion import Train_Validation
from prediction_validation_insertion import Predict_Validation
from train_model import Training_Model
from predict_model import Predict_From_Model
import shutil

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

application = Flask(__name__)
dashboard.bind(application)
CORS(application)

@application.route("/",methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@application.route("/trainModel",methods=['POST'])
@cross_origin()
def trainModelRoute():
    try:
        if request.method == 'POST':
            try:
                if request.form is not None:
                    if request.form['model_list[]'] is not None and request.form['sampler'] is not None:
                        model_list = request.form.getlist('model_list[]')
                        sampling = request.form['sampler']
                        training_batch_file_path = 'Training_Batch_Files/'

                        trainValidation_obj = Train_Validation(training_batch_file_path)
                        trainValidation_obj.training_validation()
                        print(type(sampling))
                        trainModel_obj = Training_Model(model_list,sampling)
                        is_training_success = trainModel_obj.train_model()
                        if (is_training_success == 'success'):
                            return Response('Training Successfully Completed !!!')
            except ValueError:
                print(str(ValueError))
                return Response('Error Occured! %s' % str(ValueError))
            except KeyError:
                print(str(KeyError))
                return Response('Error Occured! %s' % str(KeyError))
        else:
            print('None Request Method Passed')
        return Response('Training Successfully Completed')
    except Exception as e:
        print(e)
        raise e


@application.route("/predictBatch",methods=['POST'])
@cross_origin()
def predictBatchRoute():
    try:
        if request.method == 'POST':
            #print(request.files)
            try:
                if 'batchfile[]' in request.files:

                    if os.path.exists('Prediction_Logs'):
                        file = os.listdir('Prediction_Logs')
                        if not len(file) == 0:
                            for f in file:
                                os.remove('Prediction_Logs/' + f)
                    else:
                        pass

                    if os.path.exists('Predicted_Files'):
                        file = os.listdir('Predicted_Files')
                        if not len(file) == 0:
                            for f in file:
                                os.remove('Predicted_Files/' + f)
                    else:
                        pass

                    if os.path.exists('Prediction_Batch_Files'):
                        file = os.listdir('Prediction_Batch_Files')
                        if not len(file) == 0:
                            for f in file:
                                os.remove('Prediction_Batch_Files/' + f)
                    else:
                        pass

                    if os.path.exists('PredictionFileTo_Predict'):
                        file = os.listdir('PredictionFileTo_Predict')
                        if not len(file) == 0:
                            for f in file:
                                os.remove('PredictionFileTo_Predict/' + f)
                    else:
                        pass

                    batchFiles = request.files.getlist("batchfile[]")
                    #print(batchFiles)
                    for file in batchFiles:
                        file.save('Prediction_Batch_Files/' + file.filename)

                    prediction_batch_file_path = 'Prediction_Batch_Files'
                    listoffile_predict = [f for f in os.listdir(prediction_batch_file_path)]
                    predictionValidation_obj = Predict_Validation(prediction_batch_file_path,listoffile_predict)
                    predictionValidation_obj.prediction_validation()

                    prediction_obj = Predict_From_Model()
                    response = prediction_obj.predict_model()
                    if response == 1:
                        return Response('Bulk Batch Prediction Completed Successfully !!!')
                    else:
                        return Response('Error while doing Bulk Batch Prediction !!!')
            except Exception as e:
                print(e)
                raise e
            return Response('File Uploaded Successfully!!')
    except Exception as e:
        print(e)
        return Response('Error Occured::%s'%str(e))


@application.route("/predictRow",methods=['POST'])
@cross_origin()
def predictBatchRoute():
    try:
        if request.method == 'POST':
            try:
                if request.form is not None:
                    if request.form['comment'] is not None:
                        comment = request.form['comment']

            except Exception as e:
                raise e
    except Exception as e:
        print(e)
        return Response('Error Occured::%s' % str(e))

if __name__ == '__main__':
    host = '0.0.0.0'
    port = 5000
    httpd = simple_server.make_server(host,port,application)
    httpd.serve_forever()

