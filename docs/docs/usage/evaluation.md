---
permalink: /docs/usage/evaluation/
---

# Evaluation
This module is used for evaluation and benchmarking the AI models.
It runs on the port '5002'.

## APIs

::: details Evaluate

This api evaluates the given model in the http request and returns the evaluation results.
It is called by sending a request to the evaluation port + '/evaluate'

function: 

    evaluate_model()

    This function gets the testing dataset from the http request along with the zipped model to be evaluated.
    It extracts the zipped model and adds it to temporary zipped model.::
        
        # Read bytes of model.zip and build a new zip file
        filename = request.files['model'].filename
        zipped_model = request.files['model'].read()
        with open(filename + ".zip", 'wb') as s:
            s.write(zipped_model)
    
    It calls the :ref:`eval.evaluate(filename, X_test, y_test, metric) <evalbackend>` for evaluation of the model.
    
    :return: dictionary of evaluation results of the given model
    :rtype: dictionary

:::

::: details Benchmark

This api is called when multiple models are needed to be evaluated and benchmarked for optimization.
It is called by sending a request to the evaluation port + '/benchmark'

function: 

    benchmark_models()

    It reads the metrics and the list of models to be evaluated from the http request.
    Copies each model in the model list to the temp location so that the backend function can use them from that location for evaluation::

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
    
    Then it calls :ref:`eval.evaluate(filename, X_test, y_test, metric) <evalbackend>` for each model in the model list::
        
        results = {}
        for model in model_list:
            #Evaluate model and return loss
            result = eval.evaluate(model, X_test, y_test, metric)
            results[model] = result

    :return: It returns a json made from the dictionary of evaluation results of all the models
    :rtype: JSON

:::