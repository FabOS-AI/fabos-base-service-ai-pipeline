import flask
import autokeras as ak
import pandas as pd
import numpy as np

from flask import request, jsonify
from tensorflow import keras

import os
import json
import eval_backend as eval

PORT = 5002

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route("/evaluate", methods=["GET", "POST"])
def evaluate_model():
    """
    Reads the metrics and the model to be evaluated   
    Copies the model from the http request file parameter to a temp model
    The temp model is sent to the eval_backend for evaluation
    The results contain the evaluation results from the backend
    deletes the temp model
    """

    # Get the metric
    if "metric" in request.args:
        metric = request.args.getlist('metric', type=str)
    else:
        metric = ["mse", "rmse", "mae", "mape", "msle"]

    # Read testdata
    X_test = request.files['X_test'].read()
    X_test = json.loads(X_test)
    X_test = np.array(X_test)
    
    y_test = request.files['y_test'].read()
    y_test = json.loads(y_test)
    y_test = np.array(y_test)

    # Read bytes of model.zip and build a new zip file
    filename = request.files['model'].filename
    zipped_model = request.files['model'].read()
    with open(filename + ".zip", 'wb') as s:
        s.write(zipped_model)
        
    #Evaluate model and return loss
    result = eval.evaluate(filename, X_test, y_test, metric)

    # delete model after evaluation
    del s

    return result


@app.route("/benchmark", methods=["GET", "POST"])
def benchmark_models():
    """
    Reads the metrics and the list of models to be evaluated   
    Copies one model at a time from the list to a temp model
    The temp model is sent to the eval_backend for evaluation
    The results contain the evaluation results from the backend
    The procedure is repeated for all the models in the list
    A dictionary is created containing the model names and their evaluation results for benchmarking
    The dictionary is returned as a json dump
    """

    # Get the metric
    if "metric" in request.args:
        metric = request.args.getlist('metric', type=str)
    else:
        metric = ["mse", "rmse", "mae", "mape", "msle"]

    # Read testdata
    X_test = request.files['X_test'].read()
    X_test = json.loads(X_test)
    X_test = np.array(X_test)
    
    y_test = request.files['y_test'].read()
    y_test = json.loads(y_test)
    y_test = np.array(y_test)

    # Check received files for models
    model_list = []
    for file in request.files:

        if "model" in file:
            filename = request.files[file].filename
            zipped_model = request.files[file].read()
            with open(filename + ".zip", 'wb') as s:
                s.write(zipped_model)

            model_list.append(filename)
        else:
            print("nope")
    
    # Evaluate models and store the results
    results = {}
    print(model_list)
    for model in model_list:
        #Evaluate model and return loss
        result = eval.evaluate(model, X_test, y_test, metric)
        results[model] = result

    # Convert python dict to json format
    json_result = json.dumps(results)
    return json_result


# ============================================================================
# =============================== Service Init ===============================

if __name__ == "__main__":

    app.run(debug=True, port=PORT)