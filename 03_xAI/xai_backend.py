import numpy as np
import shutil
import autokeras as ak
import pandas as pd

from modules.surrogate import Surrogate
from modules.feature_explain import FeatureExplainer

from tensorflow import keras

RANDOM = 42

def run_lime_tabular(filename, X_test, y_test, column_list):

    # Unpack tensorflow model
    shutil.unpack_archive(filename= filename + ".zip", extract_dir="test_model")

    # load keras model
    model = keras.models.load_model('test_model', custom_objects=ak.CUSTOM_OBJECTS)

    np.random.state = RANDOM

    X_test_df = pd.DataFrame(data = X_test,
                          columns = column_list)

    exp = FeatureExplainer("lime", "tabular", model.predict, data=X_test, class_names=np.unique(y_test))
    res = exp.explain(X_test_df.iloc[0:2], num_features=2)

    return res


def run_surrogate(filename, X_test, y_test):

    # Unpack tensorflow model
    shutil.unpack_archive(filename= filename + ".zip", extract_dir="test_model")

    # load keras model
    model = keras.models.load_model('test_model', custom_objects=ak.CUSTOM_OBJECTS)

    np.random.state = RANDOM

    # Train surrogate
    surrogate = Surrogate(y_test.tolist())
    result = surrogate.generate_surrogate(X_test, model, 'dt')

    return result


def run_shap_tabular(filename, X_test, column_list):

     # Unpack tensorflow model
    shutil.unpack_archive(filename= filename + ".zip", extract_dir="test_model")

    # load keras model
    model = keras.models.load_model('test_model', custom_objects=ak.CUSTOM_OBJECTS)

    np.random.state = RANDOM

    X_test_df = pd.DataFrame(data = X_test,
                          columns = column_list)
    
    exp = FeatureExplainer("shap", "tabular", model, data=X_test_df, shap_method="linear")

    explanation = exp.explain(X_test_df.iloc[:20])

    return explanation


def run_shap_images_tf():

    import tensorflow as tf
    import shap

    model = tf.keras.models.load_model('data/model_mnist_tf.h5')
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255


    background = x_train[np.random.choice(x_train.shape[0], 4, replace=False)]
    x = x_test[0:2]

    exp = FeatureExplainer("shap", "images", model, data=tf.convert_to_tensor(background), shap_method="gradient", shap_summarize=None)
    shap_values = exp.explain(tf.convert_to_tensor(x))

    shap.image_plot(shap_values, -x)