import lime
import shap
import numpy as np
import pandas as pd



#TODO: Update docs
#TODO: Check if Exceptions are correct for respective erros
#TODO: Refactor some staticmethods into helper module / class?
#TODO: Demos
#TODO: Force plots? Can they be applied to lime?
#TODO: Other visualizations. Maybe transfer to visualization module?
#TODO: Add documentation for kwargs
#TODO: Unifiy outputs
#TODO: Proper Logging

class FeatureExplainer:

    def __init__(self, methods, data_type, model, data=None, shap_method="auto", shap_summarize="auto", n_background_samples=100, **kwargs):
        """
        Explains machine Learning Algorithms using local explantion methods.

        Args:
            method (string): "lime" or "sharp"
            model (function / object):
            data_type (string):
            data: Used in Lime tabular?. Can be used in shap tabular tree. Has to be used for shap tabular kernel as reference image
            shap_method (string) = "auto": Which shap method (kernel, tree, deep, gradient, linear) should be used. Is ignored for lime.
                                         "auto" tries to choose the appropiate method in case of tabular data. Defaults to GradientExplainer in case of image.
            shap_summarize(string) = "median: How to summarize Input data (None, "median", "mean", "random", callable). Can be set to None for small datasets.
                                            That however, would be very slow. If

        Returns:
        """
        if type(methods) is str:
            self.methods = [methods.lower()]
        else:
            self.methods = [m.lower() for m in methods]

        self.data_type = data_type.lower()
        if self.data_type == "images":
            self.data_type = "image"

        self.n_background_samples = n_background_samples
        self.global_custom_args = kwargs

        if "shap" in self.methods:
            self.shap_method = self._choose_shap_method(shap_method, model, data_type)
        else:
            self.shap_method = None
        if hasattr(model, "predict_proba"):
            self.model = model.predict_proba
            if (self.shap_method not in ["linear", "tree", "deep", "gradient"]):
                self.model_shap = model.predict_proba
            else:
                self.model_shap = model
        else:
            self.model = model
            self.model_shap = self.model

        self.feature_names = None

        if ("feature_names" not in kwargs) and (type(data) is pd.core.frame.DataFrame) and (data_type == "tabular"):
            self.feature_names = data.columns.values

        if data is None:
            self.data = None
        elif ("lime" in self.methods) or (("shap" in self.methods) and (self.shap_method not in ["deep", "gradient"])):
            self.data = self._convert_to_array(data)
        else:  # keep data as tensor
            self.data = data

        self._validate_input()
        self.explainers = {m: '' for m in self.methods}

        self.shap_summarize = shap_summarize # shap need seperate data summary if user wnats explanation for more than 1 method
        self.data_summarized = self.data
        if "shap" in self.methods:
            self._shap_data_summary()


    def _validate_input(self):
        """
        Validates input data and the choosen methods. Only for internal use. Raises ValueError, NotImplementedError, TypeError.

        Args:

        Returns:

        """
        allowed_methods = ["lime", "shap"]
        allowed_data_types = ["tabular", "image"]

        if (self.data is None) and (self.data_type == "tabular"):
            if ("lime" in self.methods):
                raise ValueError("self.data must not be none if you want to use lime for tabular data.")

            elif ("shap" in self.methods) and (self.shap_method == "kernel"):
                raise ValueError("self.data must not be none if you want to use shap (kernel) for tabular data.")
        for m in self.methods:
            if m not in allowed_methods:
                raise NotImplementedError(f"{m} not implemented. At the moment only the methods {allowed_methods} are implemented.")

        if self.data_type not in allowed_data_types:
            raise NotImplementedError(f"{self.data_type} not implemented. At the moment only the data_types {allowed_data_types} are implemented.")

        if ("shap" in self.methods) and (self.shap_method == "tree") and (not self._is_tree(self.model_shap)):
            raise TypeError(f"Tried to use a (shap) tree based method for a non tree object of type {type(self.model_shap)}")


    def _shap_data_summary(self):
        """
        Creates data summary for shap explanation. Raises NotImplementedError, ValueError.

        Args:

        Returns:

        """
        if (self.data_type == "tabular") and (self.data is not None):
            if self.shap_summarize == "auto":
                self.shap_summarize = "median"

            if self.shap_summarize is None:
                pass
            elif self.shap_summarize == "median":
                self.data_summarized = np.median(self.data, axis=0).reshape((1, self.data.shape[1]))
            elif self.shap_summarize == "mean":
                self.data_summarized = np.mean(self.data, axis=0).reshape((1, self.data.shape[1]))
            elif callable(self.shap_summarize):
                self.data_summarized = self.shap_summarize(self.data).reshape((1, self.data.shape[1]))
            else:
                raise NotImplementedError(f"Summarizing method {self.shap_summarize} not found. Please try one of (None, 'median', 'mean', callable).")

        elif (self.data_type == "image") and (self.data is not None):
            if self.shap_summarize == "auto":
                self.shap_summarize = "random"

            if self.shap_summarize is None:
                pass
            elif self.shap_summarize == "random":
                self.data_summarized = self.data[np.random.choice(len(self.data), size=self.n_background_samples)]
            else:
                ValueError("At the moment only passing background data or random sampling is implemented for shap for images.")


    def explain(self, x, labels=None, **kwargs):
        """
        Calculate feature importance

        Args:
            x (np.array | DataFrame): Datapoints / image to explain.
            labels ([int]): Labels to explain. At the moment only relevant for lime_image. Will be ignored for lime_tabular.
            summarize
        Returns:
            np.array | [np.array]: feature importance
        """

        results = []
        for m in self.methods:
            if (m == "lime") and (self.data_type == "tabular"):
                x = self._convert_to_array(x)
                if self._is_1d(x):
                    results.append(self._lime_tabular(x, **kwargs))
                else:
                    results.append(self._handle_multiple(x, self._lime_tabular, **kwargs))

            elif (m == "lime") and (self.data_type == "image"):
                x = self._convert_to_array(x)
                if (labels is not None) and (type(labels) is int):  # convert single label to [int]
                    labels = [labels]
                results.append(self._lime_image(x, labels, **kwargs))

            elif m == "shap":
                results.append(self._shap(x, **kwargs))

            else:
                raise NotImplementedError(f"Method {m} not implemented.")  # should never happen, because it's checked in the constructor
        return results


    @staticmethod
    def _convert_to_array(data):
        """
        Converts data structure to numpy array.
            \\TODO Better integration, but works for now.

        Args:
            data: matrix like structure

        Returns:
            np.array
        """

        return np.array(data)


    @staticmethod
    def _convert_to_tensor(data, framework):
        if framework == "pytorch":
            import torch
            data = torch.FloatTensor(data)
        else:
            import tensorflow as tf
            data = tf.convert_to_tensor(data, dtype_hint="float")
        return data


    @staticmethod
    def _handle_multiple(x, function, **kwargs):
        """
        Applies a function, that expects an 1d input, multiple times to each row of a 2d input.

        Args:
            x: np.array
            function: callable
            custom_args: dictionary

        Returns:
            Results of multiple calls of function.
        """

        if not FeatureExplainer._is_1d(x):
            list_of_results = []

            for ind in range(x.shape[0]):
                data_point = x[ind]
                result = function(data_point, **kwargs)
                list_of_results.append(result)
            return list_of_results
        else:
            return function(x, **kwargs)


    @staticmethod
    def _is_1d(x):
        """
        Checks if input is 1d or 2d array.

        Args:
            x: np.array

        Returns:
            Boolean
        """

        if (type(x) is int) or (type(x) is float):
            return True

        if not hasattr(x, "shape"):
            x = FeatureExplainer._convert_to_array(x)

        return (x.shape == ()) or (type(x.shape) is int) or (len(x.shape) == 1)


    @staticmethod
    def _is_tensor(x):
        if ("torch.Tensor" in str(type(x))) or ("tensorflow.python.framework.ops.EagerTensor" in str(type(x))):
            return True
        else:
            return False


    def _lime_tabular(self, x, **kwargs):
        """
        Calculate lime explanations in case of lime for tabular data

        Args:
            x: point

        Returns:
            np.array: lime explanations
        """
        if ("feature_names" not in self.global_custom_args) and (self.feature_names is not None):
            self.explainers["lime"] = lime.lime_tabular.LimeTabularExplainer(self.data, mode="regression", feature_names=self.feature_names, **self.global_custom_args)
        else:
            self.explainers["lime"] = lime.lime_tabular.LimeTabularExplainer(self.data, **self.global_custom_args)

        explanation = self.explainers["lime"].explain_instance(x, self.model, **kwargs)

        return explanation.as_list()


    def _lime_image(self, x, labels=None, positive_only=True, num_features=5, hide_rest=False, **kwargs):
        """
        Calculate lime explanations in the case of lime for image

        Args:
            x (np.array): Image to explain
            labels ([int] | None): Label to explain. If None explains the image with the highest probability.
                \\TODO: Integrate string/int labels?
            positive_only (bool):
            num_features (int):
            hide_rest (bool):
            **kwargs: arguments that should be passed to lime.lime_image.LimeImageExplainer.explain_instance

        Returns:
            np.array: mask which highligts relevant parts for the decision.
        """

        self.explainers["lime"] = lime.lime_image.LimeImageExplainer(**self.global_custom_args)

        if labels is None:  # No Information about which label is to be explained. Explains the label with the highest probability.
            explanation = self.explainers["lime"].explain_instance(x, self.model, top_labels=1, **kwargs)
            top_label = explanation.top_labels[0]
            _, mask = explanation.get_image_and_mask(top_label, positive_only=positive_only, num_features=num_features, hide_rest=hide_rest)

        else:
            explanation = self.explainers["lime"].explain_instance(x, self.model, labels=labels, top_labels=None, **kwargs)
            if len(labels) == 1:  # only explain one label. Can return the mask
                _, mask = explanation.get_image_and_mask(labels[0], positive_only=positive_only, num_features=num_features, hide_rest=hide_rest)

            else:  # multiple labels. Has to return a list of labels
                masks = []
                for label in labels:
                    _, mask = explanation.get_image_and_mask(label, positive_only=positive_only, num_features=num_features, hide_rest=hide_rest)
                    masks.append(mask)
                return masks

        return mask


    @staticmethod
    def _deep_learning_framework(model):
        try:
            model.named_parameters()
            return 'pytorch'
        except:
            return 'tensorflow'


    @staticmethod
    def _is_tree(model):
        """
        Check if a given model is a tree and thus can be used with shaps TreeExplainer.

        Args:
            model (object): Model to be evaluated. Can be any Python object or function.

        Returns:
            boolean: Tree or no tree.
        """
        tree_models = ["lightgbm", "xgboost", "sklearn.tree", "sklearn.ensemble.forest", "sklearn.ensemble.RandomForestRegressor", "sklearn.ensemble.iforest",
                         "sklearn.ensemble.IsolationForest", "sklearn.enselble.ExtraTreesRegressor", "sklearn.ensemble.RandomForestClassifier",
                         "sklearn.ensemble.ExtraTreesClassifier"]
        current_type = str(type(model))
        return np.any([tree_sub in current_type for tree_sub in tree_models])


    @staticmethod
    def _is_linear(model):
        """
        Check if a given model is a sklearn.linear_model and thus can be used with shaps LinearExplainer.

        Args:
            model (object): Model to be evaluated. Can be any Python object or function.

        Returns:
            boolean: sklearn.linear_model or no sklearn.linear_model.
        """
        current_type = str(type(model))
        return "sklearn.linear_model" in current_type


    @staticmethod
    def _choose_shap_method(shap_method, model, data_type):
        """
        Chooses the appropiate shap method for the given model and data type. If shap_method != 'auto' it just returns the given method. Otherwise it tries to choose
        TreeExplainer (tree) for tree models and LinearExplainer (linear) for linear_models.

        Args:
            shap_method (string): The method to be evaluated. Can be one of "auto", "kernel", "linear", "tree", "deep", "gradient".
            model (object): Model to be evaluated. Can be any Python object or function.
            data_type (string): Datatype of the input. Can be image oder tabular.

        Returns:
            string: The choosen shap method.
        """
        shap_method = shap_method.lower()
        valid_methods = ["auto", "kernel", "linear", "tree", "deep", "gradient"]

        if shap_method not in valid_methods:
            raise NotImplementedError(f"Method {shap_method} not implemented. Please try one of {valid_methods}.")

        if shap_method != "auto":
            return shap_method

        elif data_type == "tabular":
            if FeatureExplainer._is_tree(model):
                return "tree"
            elif FeatureExplainer._is_linear(model):
                return "linear"

        elif data_type == "image":
            return "gradient"

        return "kernel"


    def _shap(self, x, **kwargs):
        """
        Calculate shap explanations.

        Args:
            x (np.ndarray): Datapoints to be explained.
            kwargs:

        Returns:
            np.array: lime explanations
        """

        if self.shap_method in ["gradient", "deep"]:
            if not self._is_tensor(x) and (self._deep_learning_framework(self.model_shap) != "tensorflow"):
                x = self._convert_to_tensor(x, self._deep_learning_framework(self.model_shap))
            if (self.data_summarized is not None) and (not self._is_tensor(self.data_summarized)) and (self._deep_learning_framework(self.model_shap) != "tensorflow"):
                self.data_summarized = self._convert_to_tensor(self.data_summarized, self._deep_learning_framework(self.model_shap))

            if self._is_tensor(x) and (self._deep_learning_framework(self.model_shap) == "tensorflow"):
                x = self._convert_to_array(x)

        if self.shap_method == "tree":
            self.explainers["shap"] = shap.TreeExplainer(self.model_shap, self.data_summarized, **self.global_custom_args)

        elif self.shap_method == "kernel":
            self.explainers["shap"] = shap.KernelExplainer(self.model_shap, self.data_summarized, **self.global_custom_args)

        elif self.shap_method == "linear":
            self.explainers["shap"] = shap.LinearExplainer(self.model_shap, self.data_summarized, **self.global_custom_args)

        elif self.shap_method == "gradient":
            self.explainers["shap"] = shap.GradientExplainer(self.model_shap, self.data_summarized, **self.global_custom_args)

        elif self.shap_method == "deep":
            self.explainers["shap"] = shap.DeepExplainer(self.model_shap, self.data_summarized, **self.global_custom_args)
            # currently only GradientExplainer works for tf 2.0. See https://github.com/slundberg/shap/issues/548

        explanation = self.explainers["shap"].shap_values(x, **kwargs)

        if (self.shap_method in ["gradient", "deep"]) and (self._deep_learning_framework(self.model_shap) == "pytorch") and (len(explanation[0].shape) == 4):
            print("SWAPAXES")
            shape_before = explanation[0].shape
            explanation = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in explanation]
            print(f"The order of explanation channels (shape = {shape_before}) has been swapped to shape = {explanation[0].shape} for input in shap.image_plot.")  # reorient channels?
            print("To switch back, try: 'shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]'")

        return explanation
