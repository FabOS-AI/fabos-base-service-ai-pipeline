import flask
import json
import numpy as np

from numpy.random import sample
from flask import request

import xai_backend as xai

app = flask.Flask(__name__)
app.config["DEBUG"] = True

PORT = 5003

@app.route("/xai/", methods=["GET"])
def xai_main():
    """
    Hello xai
    """

    return "Hello from xAI service"


@app.route("/shap", methods=["GET", "POST"])
def shap():

    # Read testdata
    X_test = request.files['X_test'].read()
    X_test = json.loads(X_test)
    X_test = np.array(X_test)
    
    y_test = request.files['y_test'].read()
    y_test = json.loads(y_test)
    y_test = np.array(y_test)

    # Set the columns to preprocess
    column_list = []
    if "col" in request.args:
        # Get specific columns to process
        column_list = request.args.getlist("col", type=str)

    # Read bytes of model.zip and build a new zip file
    filename = request.files['model'].filename
    zipped_model = request.files['model'].read()
    with open(filename + ".zip", 'wb') as s:
        s.write(zipped_model)

    explain = xai.run_shap_tabular(filename, X_test, column_list)

    del s

    return explain


@app.route("/lime", methods=["GET", "POST"])
def lime():

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

    # Set the columns to preprocess
    column_list = []
    if "col" in request.args:
        # Get specific columns to process
        column_list = request.args.getlist("col", type=str)

    feature_explain = xai.run_lime_tabular(filename, X_test, y_test, column_list)

    del s

    return feature_explain

@app.route("/surrogate", methods=["GET", "POST"])
def surrogate():

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

    surrogate = xai.run_surrogate(filename, X_test, y_test)

    del s

    return surrogate


if __name__ == "__main__":    

    # Run the app on the given port
    # app.run(debug=True, port=PORT)
    app.run(host="0.0.0.0", debug=True, port=PORT)