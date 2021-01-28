from wsgiref import simple_server
from flask import Flask,render_template,request,Response,send_file
import os
import shutil
from flask_cors import cross_origin,CORS
import flask_monitoringdashboard as dashboard
import json
import pandas as pd
from training_validation_insertion import Train_Validation
from train_model import Training_Model

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
                    batchFiles = request.files.getlist("batchfile[]")
                    #print(batchFiles)
                    for file in batchFiles:
                        file.save('Prediction_Batch_Files/' + file.filename)
            except Exception as e:
                print(e)
                raise e
            return Response('File Uploaded Successfully!!')
    except Exception as e:
        print(e)
        return Response('Error Occured::%s'%str(e))
        raise e

if __name__ == '__main__':
    host = '0.0.0.0'
    port = 5000
    httpd = simple_server.make_server(host,port,application)
    httpd.serve_forever()

