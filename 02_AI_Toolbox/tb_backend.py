
from email.mime import base
import shutil
import autokeras as ak
from sqlalchemy import false

from tensorflow import keras

# import keras2onnx

from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split

# Deep Learning Modul

# Keras Ecosystem
    # AutoKeras
    # Tensorflow Model Optimization Toolkit

# ================================================================================
# ========================== Pandas Dataframe =====================================

def get_data(df, target, test_size = 0.2, random_state = 42):
    """
    BESCHREIBUNG
    PARAMETER
    RÜCKGABEWERTE
    Splits the dataset into training and testing sets
    """

    # Returns the training data as X and y
    tmp_df = df
    y = tmp_df[target].values
    X = tmp_df.drop(target, axis=1).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state = random_state) 


    return X_train, X_test, y_train, y_test


# ================================================================================
# ========================== Structured Data =====================================


def automl_df(df, target, automl_param, my_epochs, trials, loss, metrics, test_size):
    # 
    # - TODO Dokumentation
    # Based on the parameters from tb_api creates an trains either an regression or classification model

    # Insert the received test data and extract traning and test datasets
    X_train, X_test, y_train, y_test = get_data(df, target, test_size)
    
    model = None
    model_name = ""

    if automl_param == 'reg': # Regression model
        model = ak.StructuredDataRegressor(overwrite=True, loss = loss, metrics=metrics, max_trials=trials)
        model.fit(X_train, y_train, epochs=my_epochs)

    elif automl_param == "clf": # Classification model
        model = ak.StructuredDataClassifier(overwrite=True, max_trials=1)
        model.fit(X_train, y_train, epochs=my_epochs)

        # Evaluate the best model with test data
        # print(model.evaluate(X_test, y_test))        
    
    best_model = model.export_model()
    
    # Return the 
    return best_model


def automl_tts(X_train, y_train, automl_param, my_epochs, trials, loss, metrics):
    # 
    # - TODO Dokumentation
    # Based on the parameters from tb_api creates an trains either an regression or classification model

    model = None
    model_name = ""

    if automl_param == 'reg': # Regression model
        model = ak.StructuredDataRegressor(overwrite=True, loss = loss, metrics=metrics, max_trials=trials)
        model.fit(X_train, y_train, epochs=my_epochs)

    elif automl_param == "clf": # Classification model
        model = ak.StructuredDataClassifier(overwrite=True, max_trials=1)
        model.fit(X_train, y_train, epochs=my_epochs)

        # Evaluate the best model with test data
        # print(model.evaluate(X_test, y_test))        
    
    best_model = model.export_model()
    
    # Return the 
    return best_model

def tf(filename, df_data, test_size, target_param, loss, metric, epochs, n):

    X_train, X_test, y_train, y_test = get_data(df_data, target_param, test_size)

    # Unpack tensorflow model
    shutil.unpack_archive(filename= filename + ".zip", extract_dir="test_model")

    # load keras model
    base_model = keras.models.load_model('test_model', custom_objects=ak.CUSTOM_OBJECTS)
    l = len(base_model.layers())
    if n < l/2 :

        # base_model.summary()
        # Freeze Base model
        base_model.trainable = false
        n_model = keras.Model(base_model.inputs, base_model.layers[-n].output)
        i = keras.Input(shape=X_train.shape())
        x = n_model(i, training=false)

        for i in range(n):
            x = base_model.get_layer(i)(x)
            #x.add(temp)
        #o_layer = keras.layers.ReLU()(x)

        new_model = keras.Model(i, x)
        new_model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),metrics=keras.metrics.BinaryAccuracy())
        # new_model.compile(optimizer='adam', loss=loss, metrics=metric)
        new_model.fit(X_train, y_train, epochs=epochs)

        new_model.summary()

        return new_model
    
    else:
        return "Too many layers to be changed"

def tuning(tf_model, hyperparameters):
    
    pass


# ================================================================================
# ========================== Images =====================================



# def auto_ml_img(X, y, automl_param, my_epochs = 20):
#     # TODO Dokumentation

#     # TODO Unterscheidung auf Datentyp?
#     # TODO Preprocessingdienst aufrufen

#     # Suchen einer passenden 
#     # https://machinelearningmastery.com/autokeras-for-classification-and-regression/

#     # StructuredDataClassifier
#     # StructuredDataRegressor
    
#     # Generate training and testset
#     X_train, X_test, y_train, y_test = train_test_split(X, y)
#     model = None

#     if automl_param == "reg":

#         # Initialize the image regressor
#         model = ak.ImageRegressor(overwrite=True, max_trials=1)

#         # Feed the image classifier with training data
#         model.fit(X_train, y_train, epochs=my_epochs)

#         # Evaluate the best model with test data
#         print(model.evaluate(X_test, y_test))

#         pass
#     elif automl_param == "clf":

#         # Initialize the image classifier
#         model = ak.ImageClassifier(overwrite=True, max_trials=1)

#         # Feed the image classifier with training data
#         model.fit(X_train, y_train, epochs=my_epochs)

#         # Evaluate the best model with test data
#         print(model.evaluate(X_test, y_test))

#         pass

#     onnx_model = keras2onnx.convert_keras(model, model.name)

#     # onnx_model = onnxmltools.convert_keras(model)

#     # TODO Das Trainierte Modell in Onnx Format zurückgeben
#     # Muss dafür noch was gemacht werden?
#     return onnx_model

    pass


def wbs():


    pass



def make_moons_ds(nr_samples = 500):
    return make_moons(n_samples=nr_samples, shuffle=True, random_state=42)

    pass

def make_circles_ds(nr_samples = 500):
    return make_circles(n_samples=nr_samples, noise=0.2, factor=0.5, random_state=42)

    pass

def make_classification_ds(nr_samples = 500):
    return make_classification(n_samples=nr_samples, n_features = 4, n_clusters_per_class=1)

    pass