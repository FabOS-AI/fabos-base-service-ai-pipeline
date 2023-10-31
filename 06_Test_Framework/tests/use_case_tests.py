import pandas as pd
import numpy as np
import requests
import json

from sklearn.model_selection import train_test_split

class UseCaseTests:

    def __init__(self) -> None:
        """
        TODO DOKU
        """

        # Preprocessing
        self.pp_SERVER_IP = "127.0.0.1"
        self.pp_PORT = 5000

        # AI Toolbox
        self.tb_SERVER_IP = "127.0.0.1"
        self.tb_PORT = 5001

        # Evaluation
        self.eval_PORT = 5002
        self.eval_SERVER_IP = "127.0.0.1"

        # Explainable AI
        self.xAI_SERVER_IP = "127.0.0.1"
        self.xAI_PORT = 5003


        # Testdaten WZ_Verschleiss.csv
        self.col_names = ["VB","smcAC","smcDC","vib_table","vib_spindle","AE_table","AE_spindle"]
        self.target = "VB"
        self.df = pd.read_csv("../fabos-base-service-ai-pipeline/06_Test_Framework/tests/tests/WZ_Verschleiss.csv",
                            header = None,
                            names = self.col_names,
                            skiprows=1,
                            index_col= False,									
                            sep = ',')

        self.model_name = "use_case_test_model"
        self.test_size = 0.7

    
    def dataframe_preprocessing(self):
        """
        """

        # TODO
        # - Doku

        # Which columns to process
        col = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]

        # Set parameters
        my_param = {"col": col}

        # Call preprocessing service - remove nan values
        files = {
            'data': ("data", self.df.to_json(), 'application/json')
        }

        r = requests.post('http://{0}:{1}/nan'.format(self.pp_SERVER_IP, self.pp_PORT), params=my_param, files=files)
        df_json = r.json()
        resp_df = pd.DataFrame.from_dict(df_json)
        print(resp_df.head(1))


        # Get dataframe statistics
        files = {
            'data': ("data", resp_df.to_json(), 'application/json')
        }
        r = requests.post('http://{0}:{1}/info'.format(self.pp_SERVER_IP, self.pp_PORT), params=my_param, files=files)
        df_json = r.json()
        stats_df = pd.DataFrame.from_dict(df_json)
        print(stats_df.head(1))


        # Call preprocessing service - standardize values in columns "col"
        files = {
            'data': ("data", resp_df.to_json(), 'application/json')
        }

        r = requests.post('http://{0}:{1}/std'.format(self.pp_SERVER_IP, self.pp_PORT), params=my_param, files=files)
        df_json = r.json()
        resp_df = pd.DataFrame.from_dict(df_json)
        print(resp_df.head(1))


        # Call preprocessing service - Remove strongly correlating features
        files = {
            'data': ("data", resp_df.to_json(), 'application/json')
        }

        r = requests.post('http://{0}:{1}/rm_corr'.format(self.pp_SERVER_IP, self.pp_PORT), params=my_param, files=files)
        df_json = r.json()
        resp_df = pd.DataFrame.from_dict(df_json)
        print(resp_df.head(1))


    def ai_toolbox_modelling(self):
        """
        """
        
        # TODO
        # - Verwende Rohdaten
        # - rufe PP Service auf
        # - rufe AI-Toolbox auf und lasse Modell generieren
        # - erhalte fertiges Modell zurück und speichere es (für andere tests)
        
        
        
        model = "reg"
        
        max_trials = 5
        epochs = 10

        # Which columns to process
        col = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]

        # Set parameters
        my_param = {"col": col}

        ### DATA IN PANDAS DF ###
        files = {
            'data': ("data", self.df.to_json(), 'application/json')
        }
        r = requests.post('http://{0}:{1}/nan'.format(self.pp_SERVER_IP, self.pp_PORT), params=my_param, files=files)
        df_json = r.json()
        resp_df = pd.DataFrame.from_dict(df_json)
        print(resp_df.head(1))
        
        # Call preprocessing service - standardize values in columns "col"
        files = {
            'data': ("data", resp_df.to_json(), 'application/json')
        }
        r = requests.post('http://{0}:{1}/std'.format(self.pp_SERVER_IP, self.pp_PORT), params=my_param, files=files)
        df_json = r.json()
        resp_df = pd.DataFrame.from_dict(df_json)
        print(resp_df.head(1))

        # Call preprocessing service - Remove strongly correlating features
        files = {
            'data': ("data", resp_df.to_json(), 'application/json')
        }        
        r = requests.post('http://{0}:{1}/rm_corr'.format(self.pp_SERVER_IP, self.pp_PORT), params=my_param, files=files)
        df_json = r.json()
        resp_df = pd.DataFrame.from_dict(df_json)
        print(resp_df.head(1))

        metrics = ["mse", "mae", "mape" ]
        # Set parameters
        my_param = {"model": model, "target": self.target, "test_size": self.test_size, "col": self.df.columns.drop("VB").values, "max_trials": max_trials, "epochs": epochs, "metric": metrics}

        # files = {
        #     'data': ("data", resp_df.to_json(), 'application/json')
        # }

        tmp_df = resp_df
        y = tmp_df[self.target].values
        X = tmp_df.drop(self.target, axis=1).values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state = 42) 

        X_train_lists = X_train.tolist()
        y_train_lists = y_train.tolist()

        X_train_json_str = json.dumps(X_train_lists)
        y_train_json_str = json.dumps(y_train_lists)
        files = {
            'X_train': ("X_train", X_train_json_str, 'application/json'),
            'y_train': ("y_train", y_train_json_str, 'application/json')
        }

        # Call ai-toolbox service
        r = requests.post('http://{0}:{1}/automl'.format(self.tb_SERVER_IP, self.tb_PORT), params=my_param, files=files)  
        print(r)

        # TODO
        # - rückgabe analysieren
        # - wie muss eine passende Rückgabe aussehen?
        # - Modell erhalten und wieder speichern?

        # Receive model as part of the response content in bytes
        with open(self.model_name + ".zip", 'wb') as s:
            s.write(r.content)

        print("Done")


    def xai_service_tests(self):
        """
        DOKU
        """
        
        # TODO
        # - Doku schreiben





        # USE-CASE Ablauf
        # - lade Testdaten
        # - lade Zip Modell
        # - übergebe Modell als Dateianhang an xai service
        # - übergebe einzelnen Datenpunkt an xai service
        # - erhalte Rückgabe zurück

        # load data
        json_data = self.df.to_json()

        # extract arbitrary data point to explain by the service
        my_sample = self.df.sample(2)

        # Set params
        methode = "lime"
        data_type = "tabular"
        # model_name = "very_good_test_model_name"
        model_name = "use_case_test_model"
        target = "VB"

        # send model as (zip) file
        # single data sample as json
        files = {
            'model': (model_name, open(model_name + ".zip", 'rb'), 'application/octet-stream'),
            'data_sample': ("data_sample", my_sample.to_json(), 'application/json'), 
            'data': ("data", self.df.to_json(), 'application/json')
        }

        # Set parameters
        my_param = {"method": methode, "data_type": data_type, "model_name": model_name, "target": target}

        # Call xAI service
        # headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
        r = requests.post('http://{0}:{1}/explain'.format(self.xAI_SERVER_IP, self.xAI_PORT), params=my_param, files=files)  

        # RÜCKGABE AUSLESEN

        print("done")


    def evaluation_service_single_model(self):
        """
        DOKU
        """

        ########################
        # Call preprocessing service
        ########################

        # Which columns to process
        col = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]

        # Set parameters
        my_param = {"col": col}

        # Call preprocessing service - remove nan values
        files = {
            'data': ("data", self.df.to_json(), 'application/json')
        }

        r = requests.post('http://{0}:{1}/nan'.format(self.pp_SERVER_IP, self.pp_PORT), params=my_param, files=files)
        df_json = r.json()
        resp_df = pd.DataFrame.from_dict(df_json)
        print(resp_df.head(1))

        
        # Call preprocessing service - standardize values in columns "col"
        files = {
            'data': ("data", resp_df.to_json(), 'application/json')
        }

        r = requests.post('http://{0}:{1}/std'.format(self.pp_SERVER_IP, self.pp_PORT), params=my_param, files=files)
        df_json = r.json()
        resp_df = pd.DataFrame.from_dict(df_json)
        print(resp_df.head(1))


        # Call preprocessing service - Remove strongly correlating features
        files = {
            'data': ("data", resp_df.to_json(), 'application/json')
        }

        r = requests.post('http://{0}:{1}/rm_corr'.format(self.pp_SERVER_IP, self.pp_PORT), params=my_param, files=files)
        df_json = r.json()
        resp_df = pd.DataFrame.from_dict(df_json)
        print(resp_df.head(1))

        ########################
        # Call evaluation service - single model evaluation
        ########################
        tmp_df = resp_df
        y = tmp_df[self.target].values
        X = tmp_df.drop(self.target, axis=1).values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state = 42) 

        X_test_lists = X_test.tolist()
        y_test_lists = y_test.tolist()

        X_test_json_str = json.dumps(X_test_lists)
        y_test_json_str = json.dumps(y_test_lists)
        files = {
            'X_test': ("X_test", X_test_json_str, 'application/json'),
            'y_test': ("y_test", y_test_json_str, 'application/json'),
            'model': (self.model_name, open(self.model_name + ".zip", 'rb'), 'application/octet-stream'),
            'model_2': ("use_case_test_model_2", open("use_case_test_model_2.zip", 'rb'), 'application/octet-stream')
        }

        # Set parameters
        metrics = ["mse", "mae", "mape"]
        my_param = {"metric": metrics}

        r = requests.post('http://{0}:{1}/evaluate'.format(self.eval_SERVER_IP, self.eval_PORT), params=my_param, files=files)
        resp = r.json()
        print(resp)


    def evaluation_service_model_benchmark(self):
        """
        DOKU
        """

        ########################
        # Call preprocessing service
        ########################

        # Which columns to process
        col = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]

        # Set parameters
        my_param = {"col": col}

        # Call preprocessing service - remove nan values
        files = {
            'data': ("data", self.df.to_json(), 'application/json')
        }

        r = requests.post('http://{0}:{1}/nan'.format(self.pp_SERVER_IP, self.pp_PORT), params=my_param, files=files)
        df_json = r.json()
        resp_df = pd.DataFrame.from_dict(df_json)
        print(resp_df.head(1))

        
        # Call preprocessing service - standardize values in columns "col"
        files = {
            'data': ("data", resp_df.to_json(), 'application/json')
        }

        r = requests.post('http://{0}:{1}/std'.format(self.pp_SERVER_IP, self.pp_PORT), params=my_param, files=files)
        df_json = r.json()
        resp_df = pd.DataFrame.from_dict(df_json)
        print(resp_df.head(1))


        # Call preprocessing service - Remove strongly correlating features
        files = {
            'data': ("data", resp_df.to_json(), 'application/json')
        }

        r = requests.post('http://{0}:{1}/rm_corr'.format(self.pp_SERVER_IP, self.pp_PORT), params=my_param, files=files)
        df_json = r.json()
        resp_df = pd.DataFrame.from_dict(df_json)
        print(resp_df.head(1))

        ########################
        # Call evaluation service - single model evaluation
        ########################
        tmp_df = resp_df
        y = tmp_df[self.target].values
        X = tmp_df.drop(self.target, axis=1).values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state = 42) 

        X_test_lists = X_test.tolist()
        y_test_lists = y_test.tolist()

        X_test_json_str = json.dumps(X_test_lists)
        y_test_json_str = json.dumps(y_test_lists)
        files = {
            'X_test': ("X_test", X_test_json_str, 'application/json'),
            'y_test': ("y_test", y_test_json_str, 'application/json'),
            'model': (self.model_name, open(self.model_name + ".zip", 'rb'), 'application/octet-stream'),
            'model_2': ("use_case_test_model_2", open("use_case_test_model_2.zip", 'rb'), 'application/octet-stream')
        }

        # Set parameters
        metrics = ["mse", "mae", "mape"]
        my_param = {"metric": metrics}

        ########################
        # Call evaluation service - model benchmarking
        ########################
        r = requests.post('http://{0}:{1}/benchmark'.format(self.eval_SERVER_IP, self.eval_PORT), params=my_param, files=files)
        resp = r.json()
        print(resp)


    def pipeline_tests(self):

        # TODO
        # - Lade Rohdaten
        # - Übergabe daten an PP Service zur Vorverarbeitung
        # - rufe ai toolbox auf und erstelle Modell
        # - erhalte Modell als zip datei
        # - übergebe Testdaten und MOdell an Evaluationsdienst -> Erhalte Rüclgabe
        # - übergebe Testdaten und Modell an xai Service -> Erhalte Rüclgabe
        # 


        pass


    def extraction_pipeline_tests(self):
        
        # TODO:
        # - lade Dataen von der IPT Datenbank
        # - vorverarbeitung in passendes Dataframe
        # - aufrufen des PP Service

        pass


    def xai_test(self):
        # Which columns to process
        col = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]

        # Set parameters
        my_param = {"col": col}

        # Call preprocessing service - remove nan values
        files = {
            'data': ("data", self.df.to_json(), 'application/json')
        }

        r = requests.post('http://{0}:{1}/nan'.format(self.pp_SERVER_IP, self.pp_PORT), params=my_param, files=files)
        df_json = r.json()
        resp_df = pd.DataFrame.from_dict(df_json)
        print(resp_df.head(1))

        tmp_df = resp_df
        y = tmp_df[self.target].values
        X = tmp_df.drop(self.target, axis=1).values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state = 42) 

        X_test_lists = X_test.tolist()
        y_test_lists = y_test.tolist()

        X_test_json_str = json.dumps(X_test_lists)
        y_test_json_str = json.dumps(y_test_lists)
        files = {
            'X_test': ("X_test", X_test_json_str, 'application/json'),
            'y_test': ("y_test", y_test_json_str, 'application/json'),
            'model': (self.model_name, open(self.model_name + ".zip", 'rb'), 'application/octet-stream'),
            'model_2': ("use_case_test_model_2", open("use_case_test_model_2.zip", 'rb'), 'application/octet-stream')
        }

        r = requests.post('http://{0}:{1}/surrogate'.format(self.xAI_SERVER_IP, self.xAI_PORT), params=my_param, files=files)
        resp = r.json()
        print(resp)


        # TODO
        # - laden der Testdaten
        # - Vorverarbeitung in PP-Service
        # - Unterteilen in zwei Teildatensätze
        #   - Teil 1: Initiales Modelltraining + Evaluation
        #   - Teil 2: Feintuning des bestehenden Modells mit dem zweiten Datensatz + Evaluation
        # - Vergleichen der Evaluationsergebnisse aus initialem Training + Feintuning

        pass