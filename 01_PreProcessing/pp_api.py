import flask
from flask import request, jsonify, url_for

import pandas as pd
import pp_backend as pp

import json

# Zeitreihendaten
    # Numerische Datentypen
    # Kategorische Datentypen

# TODO mögliche Schritte der Datenvorverarbeitung
# Normalization / Standardisation
# Null-Werte filtern
# Outlier (Ausreißer) bestimmen
# Statistiken (Verteilung, Mittelwert, Varianz, etc.)
# Bayesian Modelling
# Taking care of Categorical Features

app = flask.Flask(__name__)
app.config["DEBUG"] = True
PORT = 5000


@app.route("/nan", methods=['POST'])
def process_ts_rm_nan():
    # The function removes all nan (not a number) values from the datset
    # It returns a dataframe containing the dataset values with removed nan values
    # TODO 
    # - Dokumentation

    # Load data from file
    data_content = request.files['data'].read()
    df_data_json = json.loads(data_content)
    df = pd.DataFrame.from_dict(df_data_json)

    # Reduce memory usage by setting column data types
    df = pp._reduce_mem_usage(df)

    # Set the columns to preprocess
    column_list = []
    if "col" in request.args:
        # Get specific columns to process
        column_list = request.args.getlist("col", type=str)
    else:
        # Take all columns
        column_list = df.columns.values

    # Remove nan values in dataframe on all or just specified columns
    rm_nan_df = pp.rm_nan(df, column_list)

    # generate valid json from the dataframe
    rm_nan_df = rm_nan_df.to_json()

    # return dataframe with removed nan values
    return rm_nan_df


@app.route("/std", methods=['POST'])
def process_ts_std():
    # The function standardizes all values in all or a specific column in the dataset
    # It returns a dataframe containing the dataset values with standardized values
    # TODO 
    # - Dokumentation

    # Load data from file
    data_content = request.files['data'].read()
    df_data_json = json.loads(data_content)
    df = pd.DataFrame.from_dict(df_data_json)
    
    # Reduce memory usage by setting column data types
    df = pp._reduce_mem_usage(df)

    column_list = []
    if "col" in request.args:
        # Get columns to standardize
        column_list = request.args.getlist("col", type=str)
    else:
        # If no columns are set then convert all
        column_list = df.columns

    # standardize the dataframe on all or just specific columns
    standardize_df = pp.std(df, column_list)

    # generate valid json from the dataframe
    standardize_df = standardize_df.to_json()

    # return standardized dataframe
    return standardize_df


@app.route("/norm", methods=["POST"])
def process_ts_norm():
    """
    The function normalizes all values in all or a specific column in the dataset
    It returns a dataframe containing the dataset values with normalized values
    """
    
    # TODO 
    # Dokumentation

    # Load data from file
    data_content = request.files['data'].read()
    df_data_json = json.loads(data_content)
    df = pd.DataFrame.from_dict(df_data_json)

    # Reduce memory usage by setting column data types
    df = pp._reduce_mem_usage(df)

    column_list = []
    if "col" in request.args:
        # Get columns to standardize
        column_list = request.args.getlist("col", type=str)
    else:
        # If no columns are set then convert all
        column_list = df.columns

    # normalize the dataframe and if available a specific column
    normalized_df = pp.norm(df, column_list)

    # generate valid json from the dataframe
    normalized_df = normalized_df.to_json()

    # return normalized dataframe
    return normalized_df


@app.route("/rm_corr", methods=['POST'])
def process_ts_corr():
    """
    Remove strongly correlating features depending on the set threshold parameter.
    """
    
    # TODO 
    # - Korrelation zwischen Spalten berechnen und diese ggf. bei Überschreiten löschen

    if "thresh" in request.args:
        corr_threshold = request.args.get("thresh", type=float)
    else:
        corr_threshold = 0.8

    # Load data from file
    data_content = request.files['data'].read()
    df_data_json = json.loads(data_content)
    df = pd.DataFrame.from_dict(df_data_json)

    # Reduce memory usage by setting column data types
    df = pp._reduce_mem_usage(df)

    # Remove correlating features
    df_new = pp.corr(df, corr_limit=corr_threshold)
    df_new = df_new.to_json()

    return df_new


@app.route("/enc", methods=['POST'])
def process_ts_enc():
    """
    DOKU
    Encode categorical features to numerical

    """

    # - oneHotEncoding
    # - 

    
    if "thresh" in request.args:
        corr_threshold = request.args.get("thresh", type=float)
    else:
        corr_threshold = 0.8




    # TODO 
    # - Kategorische Dataen in numerische umwandeln
    pass


@app.route("/stats", methods=['POST'])
def process_ts_stat():
    """
    Get data statistics
    """

    # Load data from file
    data_content = request.files['data'].read()
    df_data_json = json.loads(data_content)
    df = pd.DataFrame.from_dict(df_data_json)

    # Reduce memory usage by setting column data types
    df = pp._reduce_mem_usage(df)
    stats = pp.stat(df)

    stats = stats.to_json()
    return stats


@app.route("/info", methods=['POST'])
def process_ts_info():
    # The function returns a description of the dataset
    # Load data from file
    data_content = request.files['data'].read()
    df_data_json = json.loads(data_content)
    df = pd.DataFrame.from_dict(df_data_json)

    # Compute basic statistical infos of the dataset
    info_df = df.describe()
    return info_df.to_json()


if __name__ == "__main__":    

    # Run the app on the given port
    # app.run(debug=True, port=PORT)
    app.run(host="0.0.0.0", debug=True, port=PORT)


# ============================================================================
# ========================== Images ==========================================

# @app.route("/v1/preprocessing/img/ds", methods=['POST'])
# def pp_img_ds():
#     # TODO Den Datensatz vorverarbeiten
#     # Bilder klassifizieren und 

#     # Create training data
#     # DIR = "../06_Daten/Magnetic-tile-defect-datasets-master"
#     # CATEGORIES = ["MT_Blowhole", "MT_Break", "MT_Crack", "MT_Fray", "MT_Free", "MT_Uneven"]
#     # IMG_SIZE = 128
#     # training_data = []

#     # for category in CATEGORIES:
#     #     path = os.path.join(DIR, category)
#     #     class_num = CATEGORIES.index(category)

#     #     try:
#     #         for img in os.listdir(path):
#     #             img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
#     #             new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
#     #             training_data.append([new_array, class_num])
#     #     except Exception as e:
#     #         pass

#     # payload = {"images": "test",
#     #            "target": "TODO"}


#     pass

    
# @app.route("/v1/preprocessing/img/auto", methods=['POST'])
# def process_img_auto():
#     # image preprocessing
#     # TODO Input: Ein Bild / eine Reihe von Bildern als numpy arrays
#     # TODO Input: Parameter mit Preprocessingschritten
#     # TODO Output: Rückgabe vorverarbeiteter Bilder


#     # TODO Skalierung der Bilder auf einheitliche Größe
#     # TODO Mean Normalization  
#     # TODO Standardization
#     # TODO Whitening

#     pass


# @app.route("/v1/preprocessing/img/scale", methods=["POST"])
# def process_img_scale():
#     # TODO DOKUMENTATION

#     # TODO Skalierung der Bilder auf einheitliche Größe
#     # TODO: Die Größenverhältnisse der Bilder bestimmen

#     # payload = request.form.to_dict()
#     payload = request.form.to_dict(flat=False)
#     # payload = request.form.to_dict(flat=False)
    
#     images = payload['images']
#     class_ids = payload['class_ids']

#     #im_binary = base64.b64decode(images[0])

#     # Scale iamges to same size
#     pp.scale(images)

#     # im_b64 = [0]
#     # im_binary = base64.b64decode(im_b64)

#     print(payload)



#     # Take a look at the dataset
#     # for category in CATEGORIES:
#     #     path = os.path.join(DIR, category)
#     #     # print(path)

#     #     for img in os.listdir(path):
#     #         img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
#     #         plt.imshow(img_array, cmap="gray")
#     #         plt.show()
#     #         break
#     #     break


#     pass


# @app.route("/v1/preprocessing/img/mean_norm", methods=['POST'])
# def process_img_mean_norm():
#     # TODO Mean Normalization  
    
#     pass


# @app.route("/v1/preprocessing/img/std", methods=['POST'])
# def process_img_std():
#     # TODO Standardization
    
#     pass


# @app.route("/v1/preprocessing/img/white", methods=['POST'])
# def process_img_whitening():
#     # TODO Whitening

#     pass

