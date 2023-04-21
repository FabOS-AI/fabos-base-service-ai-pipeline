
import pickle
import shutil

import tf2onnx
import autokeras as ak
import flask
import json
import numpy as np
import pandas as pd
import os

from numpy.random import sample
from flask import jsonify, request, send_file, send_from_directory
from sklearn import base
from sklearn.utils import all_estimators
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import tb_backend as tb

app = flask.Flask(__name__)
app.config["DEBUG"] = True

PORT = 5001

# ================================================================================
# ========================== Structured Data =====================================

# TODO 
# - Classification
# - Regression
# - Die API erwartet die Übergabe der Bilder im base64 Format!!!!!!!!!!!!!!!!!
# - Clustering Ansätze über AUtoML ?

@app.route("/ai-toolbox/", methods=["GET"])
def ai_toolbox_main():
    """
    Hello ai-toolbox
    """

    return "Hello from AI Toolbox"


@app.route("/automl", methods=["POST"])
def automl():
    """
    BESCHREIBUNG
    PARAMETER
    RÜCKGABEWERTE
    The function to read all the details from the http request
    Getting all the parameters required for model creation from the request
    Calling the backend to train the model with the training dataset
    Returns the onnx model as an zip file for future use
    """

    # Read parameters
    if "model" in request.args: # Classification / Regression
        model_param = request.args.get("model", type=str)
    else:
        model_param = "reg"
    
    if "target" in request.args: # Target column in dataframe
        target_param = request.args.getlist("target", type=str)        
    else:
        return "\"target\" parameter not set."

    if "epochs" in request.args: # Metric for training
        model_epoch = request.args.get("epochs", type=int)
    else:
        model_epoch = 1

    if "max_trials" in request.args: # Metric for training
        model_max_trials = request.args.get("max_trials", type=int)
    else:
        model_max_trials = 1
    
    if "loss" in request.args: # Loss function for training
        model_loss = request.args.get("loss", type=str)
    else:
        model_loss = "mse"

    if "metric" in request.args: # Metric for training
        model_metric = request.args.getlist("metric", type=str)
    else:
        model_metric = ["mse"]

    if "model_name" in request.args: # Metric for training
        model_name = request.args.get("model_name", type=str)
    else:
        model_name = "generic_model_name"

    if "test_size" in request.args:
        test_size = request.args.get("test_size", type=int)
    else:
        # Use default train/test split of 80/20
        test_size = 0.8


    try:
        
        X_train = request.files['X_train'].read()
        X_train = json.loads(X_train)
        X_train = np.array(X_train)
        
        y_train = request.files['y_train'].read()
        y_train = json.loads(y_train)
        y_train = np.array(y_train)

        # Call backend and compute model with train dataset
        model = tb.automl_tts(X_train, y_train, model_param, model_epoch, model_max_trials, model_loss, model_metric)

    except Exception as err:
        print(err)

        # TODO
        # - was wenn metriken nicht erlaubt sind

        # Load complete dataset from "data" if no train/test data are available as separate files
        data_content = request.files['data'].read()
        df_data_json = json.loads(data_content)
        df_data = pd.DataFrame.from_dict(df_data_json)

        # Get feature column names
        feature_columns =  df_data.drop(labels=target_param, axis=1).values

        # Call backend and compute model with pandas dataframe
        model = tb.automl_df(df_data, target_param, model_param, model_epoch, model_max_trials, model_loss, model_metric, test_size)



    # # Load dataset
    # data_content = request.files['data'].read()
    # df_data_json = json.loads(data_content)
    # df_data = pd.DataFrame.from_dict(df_data_json)
    
    # Model folder
    model_path_temp = os.path.join(os.getcwd(), 'Models')

    # Path to the model folder
    model_name_path = os.path.join(model_path_temp, model_name)
    model.save(model_name_path, save_format='tf')

    # Generate .zip file
    model_zip_temp = os.path.join(os.getcwd(), 'zipped_model')
    model_name_zip = os.path.join(model_zip_temp, model_name)
    filename = shutil.make_archive(model_name_zip, 'zip', model_name_path)
  
    # Returns the onnx model and other necessary stuff as dictionary in json
    return send_from_directory(directory=model_zip_temp, filename=model_name + ".zip" , as_attachment=True)


@app.route("/ai-toolbox/wbs", methods=["POST"])
def wbs():
    """
    BESCHREIBUNG
    PARAMETER
    RÜCKGABEWERTE
    """

    # TODO
    # - documentation schreiben
    # - Formatierung der Daten prüfen
    # - Ergebnis als Dict zurückgeben

    # TODO
    # WBD (Wirtschaftlichkeits Basisdienst)
    # - kann hamming distance verwendet werden, um Ähnlichkeit zwischen Workflows zu bewerten?


    # Read parameters
    target = ""
    if "classes" in request.args:
        classes = request.args.get("classes", type=int)
    
    if "target" in request.args:
        target = request.args.get("target", type=str)

    # Read Payload == Dataset
    df = pd.read_json(request.get_json())


    # 

    # Call backend
    wbs()

    # Return result
    return "Success"

@app.route("/transfer-learning", methods=["POST"])
def tf():

    # Read parameters
    if "target" in request.args: # Target column in dataframe
        target_param = request.args.getlist("target", type=str)        
    else:
        return "\"target\" parameter not set."
    
    if "test_size" in request.args:
        test_size = request.args.get("test_size", type=int)
    else:
        # Use default train/test split of 80/20
        test_size = 0.8
    
    if "epochs" in request.args: # Metric for training
        model_epoch = request.args.get("epochs", type=int)
    else:
        model_epoch = 1

    if "max_trials" in request.args: # Metric for training
        model_max_trials = request.args.get("max_trials", type=int)
    else:
        model_max_trials = 1
    
    if "loss" in request.args: # Loss function for training
        model_loss = request.args.get("loss", type=str)
    else:
        model_loss = "mse"

    if "metric" in request.args: # Metric for training
        model_metric = request.args.getlist("metric", type=str)
    else:
        model_metric = ["mse"]

    if "layers" in request.args:
        n = request.args.get("layers", type=int)
    else:
        n = 1
    
    # Read testdata
    data_content = request.files['data'].read()
    df_data_json = json.loads(data_content)
    df_data = pd.DataFrame.from_dict(df_data_json)

    # Read bytes of model.zip and build a new zip file
    filename = request.files['model'].filename
    zipped_model = request.files['model'].read()
    with open(filename + ".zip", 'wb') as s:
        s.write(zipped_model)
    
    model = tb.tf(filename, df_data, test_size, target_param, model_loss, model_metric, model_epoch, n)

    del s

    return 'call succeeded'


# ============================================================================
# =============================== Images =====================================

# TODO Classification
# TODO Regression

@app.route("/v1/toolbox/auto_ml/img", methods=["GET", "POST"])
def auto_ml_img():
    """
    BESCHREIBUNG
    PARAMETER
    RÜCKGABEWERTE
    """

    # Read parameters
    model_param = ""
    if "model" in request.args:
        model_param = request.args.getlist("model", type=str)
        pass


    # Read payload
    # TODO Dienst erwartet eine liste von base64 kodierten Bildern
    # TODO Dienst erwartet Classification oder Regression Parameter


    # Return Model
    # TODO Trainiertes Model in Onnx erstellen
    # TODO Model zurückgeben


    # TODO magnetic tile datensatz testen
    # TODO autokeras benötigt x_train und y_train datensätze


    my_dict = json.loads(request.get_json())

    # Get X_train and y_train
    X_train = np.asarray(my_dict["X"])
    y_train = np.asarray(my_dict["y"])


    # TODO Konvertierung der Daten erfolgt erst im backend
    # TODO wie kann die schnittstelle asynchron gemacht werden?`

    model = tb.auto_ml_img(X_train, y_train, automl_param=model_param)
    # TODO Das Trainierte Modell in Onnx Format zurückgeben

    return "Alles gut"


# ============================================================================
# =============================== Service Init ===============================

if __name__ == "__main__":

    app.run(host="0.0.0.0", debug=False, port=PORT)