---
permalink: /docs/usage/ai-toolbox/
---

# AI ToolBox
This module has an api for creating and training AI models for a given dataset.
It makes use of AutoML to make models without needing the user to configure model parameters, but the user still needs to provide some hyperparameters.
This is hosted on the port '5001'.

## APIs
::: details AutoML

This API calls the AI-toolbox backend to create and train AI models.
It reads all the hyperparameters required for model training from http request along with the training data.
To call this API the request can be send to AI-toolbox port + '/automl'.

function:

    automl()

    This function extracts all the parameters from the http request to train the model along with the training data.
    The training dataset if available as separate feature and target, i.e., X and y then it calls 
    :ref:`tb.automl_tts(X_train, y_train, model_param, model_epoch, model_max_trials, model_loss, model_metric) <aibackend>`
    and if that is not available but just the data and target parameter then it calls
    :ref:`tb.automl_df(df_data, target_param, model_param, model_epoch, model_max_trials, model_loss, model_metric, test_size) <aibackend>`
    to create and train the AI model.

    After the model is trained it saves the model as zip file at the path provided and returns the model and path as a dictionary in json::
    
        # Model folder
        model_path_temp = os.path.join(os.getcwd(), 'Models')

        # Path to the model folder
        model_name_path = os.path.join(model_path_temp, model_name)
        model.save(model_name_path, save_format='tf')

        # Generate .zip file
        model_zip_temp = os.path.join(os.getcwd(), 'zipped_model')
        model_name_zip = os.path.join(model_zip_temp, model_name)
        filename = shutil.make_archive(model_name_zip, 'zip', model_name_path)

    :return: Returns the onnx model and and path as dictionary in json
    :rtype: JSON
:::

::: details Transfer Learning

This API calls the AI-toolbox backend to change the layers of a pre-existing model and re-train it for a similar problem.
It reads all the necessary hyper-parameters along with the model and the number of layers that needs to be changed.
To call this API the request can be send to AI-toolbox port + '/transfer-learning'

function:
    
    tf()

    This extracts all the parameters from the http request to train the model along with the training data and the number of layers to be replaced.
    It calls the function tb.tf(filename, df_data, test_size, target_param, loss, metric, epochs, n) to
    re-train the models by replacing 'n' layers from the origninal model.

    :return: Returns the model and a path as dictionary in json
    :rtype: JSON
:::
