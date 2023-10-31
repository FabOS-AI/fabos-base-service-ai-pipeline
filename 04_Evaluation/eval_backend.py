
import shutil
import autokeras as ak

from flask import request, jsonify
from tensorflow import keras

# TODO
# - sklearn modelle evaluieren!!
# - wie können die Modelle anhand ihrer lkib unterschieden werden?


def evaluate(filename, X_test, y_test, metric):
    """
    Evaluate a model with the given test data.
    Return different metrics.
    """

    # Unpack tensorflow model
    shutil.unpack_archive(filename = filename + ".zip", extract_dir="test_model")

    # load keras model
    model = keras.models.load_model('test_model', custom_objects=ak.CUSTOM_OBJECTS)
    print('load model')
    # Load evaluation data
    results = model.evaluate(x = X_test, y = y_test, return_dict=True)


    return results