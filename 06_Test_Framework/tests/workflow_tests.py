import numpy as np
import pandas as pd 
import shutil
import requests
import autokeras as ak
from tensorflow import keras
import os
import json

from sklearn.model_selection import train_test_split

class WorkflowTests:

    # TODO
    # - toolbox funktionalität hinzufügen
    # - aus toolbox_tests.py übernehmen
    # 

    def __init__(self) -> None:
        """
        TODO DOKU
        """

        self.pp_server_IP = "127.0.0.1"
        self.pp_server_PORT = "5000"

        self.tb_server_IP = "127.0.0.1"
        self.tb_server_PORT = "5001"

        self.eval_server_IP = "127.0.0.1"
        self.eval_server_PORT = "5002"

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


    def run_pipeline(self):
        """
        """

        # TODO
        # - preprocessing
        # - Modellbildung
        # - Evaluation

        # Use dataframe and remove nan values
        df = self.remove_nan(self.time_series_df)

        # y = df["VB"].values
        # X = df.drop(["VB"], axis=1).values


        # # Train/Test-Split
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        # data = {"X_train": X_train.tolist(), "y_train": y_train.tolist(), "X_test": X_test.tolist(), "y_test": y_test.tolist()}

        # Send model to ai-toolbox for model training
        file =  self.train_model(df)

        # Use received model and hand it to the evaluation service
        self.evaluate_model(df)

        print("pipeline ready")


    def remove_nan(self, df):
        """
        """

        # # Generate json from pandas dataframe
        json_data = df.to_json()

        # logging.debug("Remove nan values from time-series data")

        # Call preprocessing service and remove nan values
        headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
        response = requests.post('http://{0}:{1}/v1/preprocessing/ts/rm_nan'.format(self.pp_server_IP, self.pp_server_PORT), json=json_data, headers=headers)

        # # Generate dataframe from response
        df_json = response.json()
        resp_df = pd.DataFrame.from_dict(df_json)

        return resp_df


    def train_model(self, df, target = "VB", model = "reg"):
        """
        """


        # Convert dataframe to json
        # json_data = df.to_json()

        # df = df.tolist()
        # json_data = json.dumps(df)
        json_data = df.to_json()

        # Set parameters
        my_param = {"target": target, "model": model, "model_name": "very_good_test_model_name"}
        
        # Call the service
        headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
        r = requests.post('http://{0}:{1}/ai-toolbox/automl'.format(self.tb_server_IP, self.tb_server_PORT), json=json_data, params=my_param)        

        # Receive model as part of the response content in bytes
        with open("very_good_test_model_name.zip", 'wb') as s:
            s.write(r.content)

        # Return the file
        return s


    def evaluate_model(self, df, target = "VB", metric = "loss"):
        """        
        """


        # TODO
        # - Evaluationsmetrik übergeben
        # - target übergeben 

        model_name = "very_good_test_model_name"

        my_param = {"metric": metric, "target": target}
        json_data = df.to_json()

        files = {
            'model': (model_name, open(model_name + ".zip", 'rb'), 'application/octet-stream'),
            'data': ("data", json_data, 'application/json')            
        }

        r = requests.post('http://{0}:{1}/v1/eval'.format(self.eval_server_IP, self.eval_server_PORT), params=my_param, files=files)   





        print("response from evaluation")