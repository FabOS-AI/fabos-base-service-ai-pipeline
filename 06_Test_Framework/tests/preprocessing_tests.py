from email import header
import flask
from flask import request, jsonify, url_for
from flask import json
import requests
import numpy as np
import pandas as pd 
import io
import os
import logging
import base64

import tests.test_data as td


class PreprocessingTests:

    def __init__(self, IP = "127.0.0.1", PORT = "5000") -> None:

        self.pp_server_IP = IP
        self.pp_server_PORT = PORT

        # col_names =  ["Vib_S_X_res", "Vib_S_Y_res", "Vib_S_Z_res", "Microphone_res", "Enc_X_res", "Enc_Y_res", "Enc_Z_res", "Enc_S_res", "Vib_T_X_res", "Vib_T_Y_res", "Vib_T_Z_res","timestamp_10_res"]        
        # self.time_series_df = pd.read_csv("tests/IPT_tool_wear_dataset.csv",
        #                     header = None,
        #                     names = col_names,
        #                     skiprows=1,
        #                     index_col= False,									
        #                     sep = ',')

        # WZ_Verschleiss.csv
        self.col_names = ["VB","smcAC","smcDC","vib_table","vib_spindle","AE_table","AE_spindle"]
        self.target = "VB"
        self.time_series_df = pd.read_csv("tests/WZ_Verschleiss.csv",
                            header = None,
                            names = self.col_names,
                            skiprows=1,
                            index_col= False,									
                            sep = ',')


    def run_time_series_tests(self):

        # TODO
        # - doku

        # self.time_series_df

        ## Preprocess time series data
        
        # Remove nan values from data
        resp_df = self.remove_nan_time_series()

        # normalize data
        resp_df = self.normalize_time_series()

        # standardize data
        resp_df = self.standardize_time_series()

        # correlation_filter_time_series(data)
        logging.debug("Time-series tests successful.")


    def remove_nan_time_series(self):
        """
        Remove nan values from the dataset
        """        

        # TODO 
        # - Dokumentation schreiben

        # Add test nan to dataframe
        self.time_series_df.iloc[0][0] = np.nan
        self.time_series_df.iloc[1][1] = np.nan

        # Generate json from pandas dataframe
        json_data = self.time_series_df.to_json()

        logging.debug("Remove nan values from time-series data")

        # Add parameters to send
        # my_param = {"target": "Vib_S_X_res", "model": "reg"}
        # r = requests.post('http://127.0.0.1:5000/v1/toolbox/auto_ml/sd', json=json_data, params=my_param)

        # Call preprocessing service and remove nan values
        # print("Call preprocessing rm_nan server...")
        headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
        response = requests.post('http://{0}:{1}/v1/preprocessing/ts/rm_nan'.format(self.pp_server_IP, self.pp_server_PORT), json=json_data, headers=headers)

        # Generate dataframe from response
        df_json = response.json()
        resp_df = pd.DataFrame.from_dict(df_json)

        # Add the normalized dataframe to class instance
        # self.time_series_df = resp_df

        return resp_df


    def normalize_time_series(self):
        # TODO 
        # - Dokumentation

        # Check if dataframe contains nan values == True
        if self.time_series_df.isnull().values.any():
            print("Dataframe contains nan: " + str(True))
            self.time_series_df = self.remove_nan_time_series()
            
        # Generate json from pandas dataframe
        json_data = self.time_series_df.to_json()

        logging.debug("Normalize time-series data")
        
        # Add parameters to send
        # my_param = {"target": "Vib_S_X_res", "model": "reg"}
        # r = requests.post('http://127.0.0.1:5000/v1/toolbox/auto_ml/sd', json=json_data, params=my_param)

        # Normalize whole pandas dataframe
        headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
        response = requests.post('http://{0}:{1}/v1/preprocessing/ts/norm'.format(self.pp_server_IP, self.pp_server_PORT), json=json_data, headers=headers)

        # Generate pandas dataframe from json 
        df_json = response.json()
        resp_df = pd.DataFrame.from_dict(df_json)

        # Add the normalized dataframe to class instance
        # self.time_series_df = resp_df

        # TODO
        # - test if normalization was successful

        return resp_df


    def standardize_time_series(self):
        # TODO 
        # - Dokumentation

        # Check if dataframe contains nan values == True
        if self.time_series_df.isnull().values.any():
            print("Dataframe contains nan: " + str(True))
            self.time_series_df = self.remove_nan_time_series()
        

        json_data = self.time_series_df.to_json()

        logging.debug("Standardize time-series data")
        
        # Add parameters to send
        # my_param = {"target": "Vib_S_X_res", "model": "reg"}
        # r = requests.post('http://127.0.0.1:5000/v1/toolbox/auto_ml/sd', json=json_data, params=my_param)

        # Standardize multiple pandas dataframe column
        headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
        response = requests.post('http://{0}:{1}/v1/preprocessing/ts/std'.format(self.pp_server_IP, self.pp_server_PORT), json=json_data, headers=headers)
        
        # Generate pandas dataframe from json 
        df_json = response.json()
        resp_df = pd.DataFrame.from_dict(df_json)

        # Add the standardized dataframe to class instance
        # self.time_series_df = resp_df

        # TODO
        # - test if standardization was successful

        return resp_df


    def test_img():
        # test_data, class_ids = td.create_image_test_data()
        # payload = {"images": test_data,
        #            "class_ids": class_ids}
        # # payload = json.dumps(payload)

        # #headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

        # # Preprocess images
        # r = requests.post("http://127.0.0.1:5001//v1/preprocessing/img/scale", data=payload)
        # print(r.status_code)
        # # r = requests.post('http://127.0.0.1:5001//v1/preprocessing/img/scale', json=json_data, headers=headers)

        # # TODO AutoML images

        pass