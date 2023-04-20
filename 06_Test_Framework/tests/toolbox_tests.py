import os
import pickle
import requests
import json
import pickle
import numpy as np
import pandas as pd
import shutil

from flask import request, jsonify, url_for
from tensorflow import keras
import autokeras as ak
from sklearn.model_selection import train_test_split

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def run_tests(ts = False, img = False):
    # Schnittstellenfunktion zum Starten der Tests

    
    # ============================================================================
    # ========================== Image Tests =====================================

    if img:

        # X = []
        # y = []

        # # TODO Die Datenstruktur ist auf Bilder eingestellt
        # X = pickle.load(open("X.pickle", "rb"))
        # y = pickle.load(open("y.pickle", "rb"))


        # auto_ml_images(X, y)
    
    
    
        pass
    
    # ============================================================================
    # ========================== Time Series Tests =====================================
    
    if ts:

        # df = pd.read_csv("data/IPT_tool_wear_dataset.csv",
        #                     header = None,
        #                     names = ["Vib_S_X_res", "Vib_S_Y_res", "Vib_S_Z_res", "Microphone_res", "Enc_X_res", "Enc_Y_res", "Enc_Z_res", "Enc_S_res", "Vib_T_X_res", "Vib_T_Y_res", "Vib_T_Z_res","timestamp_10_res"],
        #                     skiprows=1,
        #                     index_col= False,									
        #                     sep = ',')

        # df = pd.read_csv("../../../06_Daten/IPT-Tool-Wear/IPT_tool_wear_dataset.csv",
        #                     header = None,
        #                     names = ["Vib_S_X_res", "Vib_S_Y_res", "Vib_S_Z_res"],
        #                     skiprows=1,
        #                     index_col= False,									
        #                     sep = ',')

        df = pd.read_csv("04_SourceCode/Base Services/06_Test_Framework/tests/data/case_1_all_csv_new.csv",
                            header = None,
                            names = ["VB", "smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"],
                            skiprows=1,
                            index_col= False,									
                            sep = ',')

        #json_data = df.to_json()
        #r1 = requests.post('http://127.0.0.1:5001/v1/preprocessing/ts/norm', json=json_data)
        #df_norm = pd.DataFrame(r1.json())
        # TODO nur einen Teil der Daten zum Testen nehmen
        #df_train = df.iloc[:100,:]
        #df_test = df.iloc[100:,:]
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
        json_data_train = df_train.to_json()
        json_data_test = df_test.to_json()

        # # Remove nan values
        # headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
        # r = requests.post('http://127.0.0.1:5001/v1/preprocessing/ts/rm_nan', json=json_data)
        # df = pd.DataFrame(r.json())


        # # TODO Welche Columns eigentlich? --> Was ist die Zielspalte??
        # # Standardize data - without target column
        # json_data = df.to_json()
        # r = requests.post('http://127.0.0.1:5001/v1/preprocessing/ts/std', json=json_data)
        # df = pd.DataFrame(r.json())        

        # TODO Andere Testdaten verwenden....
        # TODO: target column angeben!!
         

        # import autokeras as ak
        # import tensorflow as tf

        # from sklearn.model_selection import train_test_split

        # tmp_df = df
        # y = tmp_df["Vib_S_X_res"].values
        # X = tmp_df.drop("Vib_S_X_res", axis=1).values

        # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42) 
        # # Initialize the structured data classifier for time series
        # model = ak.StructuredDataClassifier(overwrite=True, max_trials=5)
        # # Feed the image classifier with training data
        # model.fit(X_train, y_train, epochs=5, verbose = 1)

        # # Remove nan rows
        # headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
        # r = requests.post('http://127.0.0.1:5001/v1/preprocessing/ts/rm_nan', json=json_data)
        # print(r.status_code)

        # j = r.json()
        # df = pd.DataFrame.from_dict(j)
        # print(df.head(5))

        #json_data = df.to_json()
        #r1 = requests.post('http://127.0.0.1:5001/v1/preprocessing/ts/norm', json=json_data_train)
        #json_data_train_std = pd.DataFrame(r1.json()).to_json()
        
        #my_param = {"target": "VB", "model": "reg", "model_name": "model_test_VB"}
        #r = requests.post('http://127.0.0.1:5000/v1/toolbox/auto_ml/sd', json=json_data_train, params=my_param)
        #print(r.content)
        
        # Receive model as part of the response content in bytes
        #with open("model_test_file.h5", 'wb') as s:
            
            # write content to file
            #s.write(r.content)
            #pass
        #print('written file onnx')
        # load keras model from previously generated file
        # model = keras.models.load_model('model.h5', custom_objects=ak.CUSTOM_OBJECTS)
        # headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
        
        file_zip_dir = os.path.join(os.getcwd(), r'Models_Zip')
        r2 = requests.post('http://127.0.0.1:5003/v1/eval', params={'metric':'mean_sqaure_error'}, files={'VB':open(os.path.join(file_zip_dir, r'model_test_VB.zip'), 'rb'), 'AE_spindle':open(os.path.join(file_zip_dir, r'temp2_zip.zip'), 'rb'), 'data': json_data_test})
        print(r2.content)
        # Take remaining data for evaluation
        # json_data = df_test.to_json()
        # TODO
        # 
        
        #test_file = open('model_test_file.h5', 'rb')
        #file = {'model': open('model_test_file.h5', 'rb')}
        #df_temp = df_test
        #y = df_temp['VB'].values
        #X = df_temp.drop('VB', axis=1).values
        #test_model = keras.models.load_model('model_test_file.h5', custom_objects=ak.CUSTOM_OBJECTS)
        #print(test_model.evaluate(X, y, verbose=0))
        
        # r = requests.post("http://127.0.0.1:5003/v1/eval", files=file, json=json_data)

        # TODO Evaluationsservice anbinden, Modell übergeben, Benchmark zurückgeben
        # TODO Modell + Dataframe übergebe
        # json_data = df.to_json()
        # my_data = {"dataframe":json_data, "model":None}
        # r = requests.post('http://127.0.0.1:5000/v1/eval/benchmark', data=my_data)
        
        # TODO Evaluationsservice anbinden, Modelle übergeben, Benchmark zurückgeben

        # 

        pass
    pass


def auto_ml_images(X, y):


    # TODO Teste AutoML auf Bildern
    # TODO Regression / Classification

    # TODO
    my_dict = dict()
    my_dict["X"] = X
    my_dict["y"] = y

    # Build json from dictionary
    json_data = None
    # json_data = json.loads(json.dumps(my_dict))
    json_data = json.dumps(my_dict, cls=NumpyEncoder)
 
     # Normalize multiple columns in pandas dataframe
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    r = requests.post('http://127.0.0.1:5000/v1/toolbox/automl/img', json=json_data, headers=headers)
    print(r.status_code)

    # TODO Die Rückgabe des Dienstes für AutoML ist ein Onnx Model, dass das trainierte KI-Modell darstellt.
    # j = r.json()
    # resp_df = pd.DataFrame.from_dict(j)
    # print(resp_df.sample(5))
    


    pass

def auto_ml_sd(X, y):

    # TODO Teste Strukturierte Daten
    # TODO Regression / Classification

    my_dict = dict()
    my_dict["X"] = X
    my_dict["y"] = y

    # Build json from dictionary
    json_data = None
    # json_data = json.loads(json.dumps(my_dict))
    json_data = json.dumps(my_dict, cls=NumpyEncoder)
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}

    # AutoML on structured data with classification 
    r = requests.post('http://127.0.0.1:5000/v1/toolbox/auto_ml/sd?model=clf', json=json_data, headers=headers)
    print(r.status_code)

    # AutoML on structured data with regression     
    r = requests.post('http://127.0.0.1:5000/v1/toolbox/automl/sd?model=reg', json=json_data, headers=headers)
    print(r.status_code)

    

    pass