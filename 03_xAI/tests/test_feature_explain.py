from modules.feature_explain import FeatureExplainer
import numpy as np
import pandas as pd
from sklearn import datasets, tree, model_selection, ensemble, svm, linear_model
import pytest
import importlib


def get_data():
    random = 42
    bc_dict = datasets.load_breast_cancer()
    bc_data = np.concatenate((bc_dict.data, bc_dict.target.reshape(-1, 1)), axis=1)
    df = pd.DataFrame(bc_dict.data, columns=bc_dict.feature_names.tolist())

    X_train, X_test, y_train, y_test = model_selection.train_test_split(df, bc_dict.target, test_size=0.25, random_state=random)
    return X_train, X_test, y_train, y_test


@pytest.fixture
def get_trained_dt():
    X_train, X_test, y_train, y_test = get_data()

    rf = tree.DecisionTreeClassifier()
    rf.fit(X_train, y_train)

    return (rf, X_train, X_test, y_train, y_test)


@pytest.fixture
def get_trained_linear():
    X_train, X_test, y_train, y_test = get_data()

    model = linear_model.LogisticRegression()
    model.fit(X_train, y_train)

    return (model, X_train, X_test, y_train, y_test)


@pytest.fixture
def get_trained_nn_torch():
    from torch import nn
    import torch.nn.functional as F
    import torch

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout2d(0.25)
            self.dropout2 = nn.Dropout2d(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            output = F.log_softmax(x, dim=1)
            return output

    from torchvision import transforms, datasets
    import torch

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    model = Net()
    model.load_state_dict(torch.load("data/model_mnist_state_dict.pt")) #model = torch.load("data/model_mnist.pt")
    mnist_test = datasets.MNIST('data/mnist/', train=False, transform=transform, target_transform=None, download=True)
    mnist_train = datasets.MNIST('data/mnist/', train=True, transform=transform, target_transform=None, download=True)

    return model, mnist_train, mnist_test


@pytest.fixture
def get_trained_nn_tf():
    import tensorflow as tf
    model = tf.keras.models.load_model('data/model_mnist_tf.h5')
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    return model, x_train, x_test


def test__is_1d():
    assert FeatureExplainer._is_1d(1) == True
    assert FeatureExplainer._is_1d(np.array(1)) == True
    assert FeatureExplainer._is_1d(np.array([1])) == True
    assert FeatureExplainer._is_1d(np.array([1, 2, 3, 4, 5])) == True
    assert FeatureExplainer._is_1d([1, 2, 3, 4, 5]) == True

    assert FeatureExplainer._is_1d([[1, 2, 3], [4, 5, 6]]) == False
    assert FeatureExplainer._is_1d([[[1], [2]], [[3], [4]]]) == False


def test__handle_multiple():
    square = lambda x: x ** 2
    assert np.all(FeatureExplainer._handle_multiple(np.array([1, 2]), square) == np.array([1, 4]))

    assert type(FeatureExplainer._handle_multiple(np.array([[1]]), square)) is list
    assert FeatureExplainer._handle_multiple(np.array([[1]]), square) == [np.array([1])]
    assert FeatureExplainer._handle_multiple(np.array([[1], [2]]), square) == [np.array([1]), np.array([4])]

    def do_something(x, what=None):
        if what is None:
            return x
        elif what == "zero":
            return 0

    assert FeatureExplainer._handle_multiple(np.array([[1], [2]]), do_something) == [np.array([1]), np.array([2])]
    assert FeatureExplainer._handle_multiple(np.array([[1], [2]]), do_something, what="zero") == [np.array([0]), np.array([0])]


def test__convert_to_array():
    assert np.all(FeatureExplainer._convert_to_array([1, 2]) == np.array([1, 2]))
    assert np.all(np.array(pd.DataFrame([[1, 1], [2, 2]])) == np.array([[1, 1], [2, 2]]))


def test_feature_explainer_init():
    with pytest.raises(NotImplementedError):
        FeatureExplainer("dime", "tabular", "model") # raises because of dime typo

    with pytest.raises(NotImplementedError):
        FeatureExplainer("lime", "fabular", "model") #raises because of fabular typo

    with pytest.raises(ValueError):
        FeatureExplainer("lime", "tabular", "model", data=None) #raises because data is None

    with pytest.raises(TypeError):
        FeatureExplainer("shap", "tabular", svm.SVC(), shap_method="tree", data=None) #raises because svm is no tree

    with pytest.raises(ValueError):
        FeatureExplainer("shap", "tabular", svm.SVC(), shap_method="kernel", data=None) #raises because method kernel and data is none

    exp = FeatureExplainer("shap", "tabular", svm.SVC(), data=[[1, 1], [2, 2], [10, 10]], shap_summarize=None)
    assert np.allclose(exp.data_summarized, np.array([[1, 1], [2, 2], [10, 10]]))

    exp = FeatureExplainer("shap", "tabular", svm.SVC(), data=[[1, 1], [2, 2], [10, 10]], shap_summarize="mean")
    assert np.allclose(exp.data_summarized,  np.array([4.33333333, 4.33333333]))

    exp = FeatureExplainer("shap", "tabular", svm.SVC(), data=[[1, 1], [2, 2], [10, 10]], shap_summarize="median")
    assert np.allclose(exp.data_summarized, np.array([2, 2]))

    exp = FeatureExplainer("shap", "tabular", svm.SVC(), data=[[1, 1], [2, 2], [10, 10]], shap_summarize=lambda x: np.max(x, axis=0))
    assert np.allclose(exp.data_summarized, np.array([10, 10]))

    exp = FeatureExplainer("shap", "tabular", tree.DecisionTreeClassifier(), data=None)
    assert exp.data is None

    exp = FeatureExplainer("shap", "tabular", svm.SVC(probability=True), data=[[1, 1]])
    assert callable(exp.model_shap) # converted to predict proba

    exp = FeatureExplainer("shap", "tabular", tree.DecisionTreeClassifier(), data=[[1, 1]])
    assert type(exp.model_shap) is tree.DecisionTreeClassifier

    exp = FeatureExplainer("shap", "tabular", tree.DecisionTreeClassifier(), data=[[1, 1]], shap_method="kernel")
    assert not type(exp.model_shap) is tree.DecisionTreeClassifier # because method='kernel'

    exp = FeatureExplainer("lime", "tabular", tree.DecisionTreeClassifier(), data=[[1, 1]])
    assert callable(exp.model_shap)


def test_lime_tabular(get_trained_dt):
    rf, X_train, X_test, y_train, y_test = get_trained_dt

    exp = FeatureExplainer("lime", "tabular", rf.predict_proba, data=X_train, class_names=np.unique(y_train))
    explanation = exp.explain(X_train.iloc[0:2], num_features=4)

    assert type(explanation) is list
    assert type(explanation[0]) is list
    assert len(explanation[0]) == 2 # two datapoints
    assert len(explanation[0][0]) == 4 #4 features

    explanation = exp.explain(X_train.iloc[10], num_features=3)
    assert type(explanation) is list
    assert type(explanation[0]) is list
    assert len(explanation[0]) == 3 # 3 features

    all_columns = X_train.columns.values
    condition = explanation[0][0][0]
    any_column_in_condition = False
    for col in all_columns:
        if col in condition:
            any_column_in_condition = True
            break
    assert any_column_in_condition # correctly extracts column names from df


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="pytorch not present")
def test_lime_image():
    from PIL import Image
    import torch.nn as nn
    import torch
    from torchvision import models, transforms
    import torch.nn.functional as F
    from lime import lime_image

    def get_image(path):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB') 

    img = get_image('data/dogs.png')

    model = models.inception_v3(pretrained=True)

    pill_transf = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224)])
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])       
    preprocess_transform = transforms.Compose([transforms.ToTensor(), normalize])

    def batch_predict(images):
        model.eval()
        batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)

        logits = model(batch)
        probs = F.softmax(logits, dim=1)
        return probs.detach().numpy()

    explainer = FeatureExplainer("lime", "image", batch_predict)
    mask = explainer.explain(np.array(pill_transf(img)), labels=[239], num_samples=100, num_features=1)
    assert mask[0].shape == (224, 224)


def test__is_tree():
    models = [tree.DecisionTreeClassifier(), tree.DecisionTreeRegressor(), ensemble.RandomForestClassifier(), ensemble.RandomForestRegressor()]
    not_tree = [svm.SVC(), linear_model.LogisticRegression(), ensemble.AdaBoostClassifier()]

    for model in models:
        assert FeatureExplainer._is_tree(model)

    for model in not_tree:
        assert not FeatureExplainer._is_tree(model)

def test__is_linear():
    linear_models = [linear_model.LogisticRegression(), linear_model.LinearRegression()]
    not_linear = [svm.SVC(), tree.DecisionTreeRegressor(), ensemble.RandomForestClassifier(), ensemble.AdaBoostClassifier()]

    for model in linear_models:
        assert FeatureExplainer._is_linear(model)

    for model in not_linear:
        assert not FeatureExplainer._is_linear(model)


def test__choose_shap_method():
    methods = ["kernel", "tree", "deep", "gradient", "linear"]
    models = [tree.DecisionTreeClassifier(), ensemble.RandomForestClassifier()]

    for method in methods:
        assert FeatureExplainer._choose_shap_method(method, None, "tabular") == method

    for model in models:
        assert FeatureExplainer._choose_shap_method("auto", model, "tabular") == "tree"

    assert FeatureExplainer._choose_shap_method("auto", linear_model.LogisticRegression(), "tabular")

    assert FeatureExplainer._choose_shap_method("auto", None, "image") == "gradient"
    assert FeatureExplainer._choose_shap_method("AuTo", None, "image") == "gradient"
    assert FeatureExplainer._choose_shap_method("DEEP", None, "image") == "deep"
    assert FeatureExplainer._choose_shap_method("deep", None, "image") == "deep"

    with pytest.raises(NotImplementedError):
        FeatureExplainer("shap", "tabular", None, shap_method="notimplementedmethod")


@pytest.mark.skipif(importlib.util.find_spec("xgboost") is None, reason="Xgboost not present")
def test_xgboost_integration():
    import xgboost
    model = xgboost.XGBClassifier()
    assert FeatureExplainer._is_tree(model)
    assert FeatureExplainer._choose_shap_method("auto", model, "tabular") == "tree"


@pytest.mark.skipif(importlib.util.find_spec("lightgbm") is None, reason="Lightgbm not present")
def test_lightgbm_integration():
    import lightgbm
    model = lightgbm.LGBMClassifier()
    assert FeatureExplainer._is_tree(model)
    assert FeatureExplainer._choose_shap_method("auto", model, "tabular") == "tree"


def test_shap_tabular(get_trained_dt, get_trained_linear):
    rf, X_train, _, _, _ = get_trained_dt
    lr, X_train, X_test, y_train, y_test = get_trained_linear

    n_samples = 10
    n_features = 30

    exp = FeatureExplainer(methods="shap", data_type="tabular", model=rf, data=None, shap_method="auto")
    explanation = exp.explain(X_train.iloc[:n_samples])
    assert type(explanation) is list
    assert type(explanation[0]) is list
    assert len(explanation[0]) == 2
    assert type(explanation[0][0]) is np.ndarray
    assert explanation[0][0].shape == (n_samples, n_features)

    explanation = exp.explain(X_train.iloc[0])
    assert FeatureExplainer._is_1d(X_train.iloc[0])
    assert type(explanation) is list
    assert type(explanation[0]) is list
    assert len(explanation[0]) == 2
    assert type(explanation[0][0]) is np.ndarray
    assert explanation[0][0].shape == (n_features,)

    exp = FeatureExplainer(methods="shap", data_type="tabular", model=lr, data=X_train, shap_method="kernel")
    explanation = exp.explain(X_train.iloc[:n_samples])
    assert type(explanation) is list
    assert type(explanation[0]) is list
    assert len(explanation[0]) == 2
    assert type(explanation[0][0]) is np.ndarray
    assert explanation[0][0].shape == (n_samples, n_features)

    exp = FeatureExplainer(methods="shap", data_type="tabular", model=lr, data=X_train, shap_method="linear")
    explanation = exp.explain(X_train.iloc[:n_samples])
    assert type(explanation[0]) is np.ndarray
    assert explanation[0].shape == (n_samples, n_features)

def test_shap_tabular_with_data(get_trained_dt, get_trained_linear):
    rf, X_train, _, _, _ = get_trained_dt
    lr, X_train, X_test, y_train, y_test = get_trained_linear

    n_samples = 10
    n_features = 30

    exp = FeatureExplainer(methods="shap", data_type="tabular", model=rf, data=X_train, shap_method="auto")
    explanation = exp.explain(X_train.iloc[:n_samples])
    assert type(explanation) is list
    assert type(explanation[0]) is list
    assert len(explanation[0]) == 2
    assert type(explanation[0][0]) is np.ndarray
    assert explanation[0][0].shape == (n_samples, n_features)

    explanation = exp.explain(X_train.iloc[0])
    assert FeatureExplainer._is_1d(X_train.iloc[0])
    assert type(explanation) is list
    assert type(explanation[0]) is list
    assert len(explanation[0]) == 2
    assert type(explanation[0][0]) is np.ndarray
    assert explanation[0][0].shape == (n_features,)

    explanation = exp.explain(X_train.iloc[1])
    assert FeatureExplainer._is_1d(X_train.iloc[1])
    assert type(explanation) is list
    assert type(explanation[0]) is list
    assert len(explanation[0]) == 2
    assert type(explanation[0][0]) is np.ndarray
    assert explanation[0][0].shape == (n_features,)

    exp = FeatureExplainer(methods="shap", data_type="tabular", model=lr, data=X_train, shap_method="kernel")
    explanation = exp.explain(X_train.iloc[:n_samples])
    assert type(explanation) is list
    assert type(explanation[0]) is list
    assert len(explanation[0]) == 2
    assert type(explanation[0][0]) is np.ndarray
    assert explanation[0][0].shape == (n_samples, n_features)

    exp = FeatureExplainer(methods="shap", data_type="tabular", model=lr, data=X_train, shap_method="linear")
    explanation = exp.explain(X_train.iloc[:n_samples])
    assert type(explanation[0]) is np.ndarray
    assert explanation[0].shape == (n_samples, n_features)

def test_shap_tabular_execution_after_error(get_trained_dt, get_trained_linear):
    rf, X_train, _, _, _ = get_trained_dt
    lr, X_train, X_test, y_train, y_test = get_trained_linear

    n_samples = 10
    n_features = 30

    exp = FeatureExplainer(methods="shap", data_type="tabular", model=rf, data=X_train, shap_method="auto")
    try:
        explanation = exp.explain(X_train.iloc[n_samples+1])
    except IndexError:
        pass
    explanation = exp.explain(X_train.iloc[:n_samples])
    assert type(explanation) is list
    assert type(explanation[0]) is list
    assert len(explanation[0]) == 2
    assert type(explanation[0][0]) is np.ndarray
    assert explanation[0][0].shape == (n_samples, n_features)
    


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="pytorch not present")
def test__deep_learning_framework_pytorch():
    import torch
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    assert FeatureExplainer._deep_learning_framework(model) == "pytorch"


@pytest.mark.skipif(importlib.util.find_spec("tensorflow") is None, reason="tensorflow not present")
def test__deep_learning_framework_tf():
    from tensorflow import keras
    model = keras.Model()
    assert FeatureExplainer._deep_learning_framework(model) == "tensorflow"


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="pytorch not present")
def test_shap_image_torch(get_trained_nn_torch):
    model, mnist_train, mnist_test = get_trained_nn_torch

    background = mnist_train.data[np.random.choice(mnist_train.data.shape[0], 4, replace=False)].float().reshape(-1, 1, 28, 28)
    x = mnist_test.data[0:2].float().reshape(-1, 1, 28, 28)


    exp = FeatureExplainer("shap", "images", model, data=background, shap_method="gradient", shap_summarize=None)
    explanation = exp.explain(x)
    shap_values = explanation[0]

    assert len(exp.data) == 4

    assert type(shap_values) is list
    assert len(shap_values) == 10
    assert shap_values[0].shape == (2, 28, 28, 1)

    exp = FeatureExplainer("shap", "images", model, data=np.array(background), shap_method="deep", shap_summarize=None)
    explanation = exp.explain(x.numpy())
    shap_values = explanation[0]
    assert type(shap_values) is list
    assert len(shap_values) == 10
    assert shap_values[0].shape == (2, 28, 28, 1)

    background = mnist_train.data[np.random.choice(mnist_train.data.shape[0], 200, replace=False)].float().reshape(-1, 1, 28, 28)
    exp = FeatureExplainer("shap", "IMAGES", model, data=background, shap_method="GRADIENT", shap_summarize="auto")
    assert len(exp.data_summarized) == 100
    assert exp.shap_method == "gradient"
    assert exp.data_type == "image"
    assert FeatureExplainer._is_tensor(exp.data)


def test_shap_image_tf(get_trained_nn_tf):
    import tensorflow as tf
    model, x_train, x_test = get_trained_nn_tf

    background = x_train[np.random.choice(x_train.shape[0], 4, replace=False)]
    x = x_test[0:2]

    exp = FeatureExplainer("shap", "images", model, data=np.array(background), shap_method="gradient", shap_summarize=None)
    explanation = exp.explain(tf.convert_to_tensor(x))
    shap_values = explanation[0]
    assert len(exp.data) == 4
    assert type(shap_values) is list
    assert len(shap_values) == 10
    assert shap_values[0].shape == (2, 28, 28, 1)

    exp = FeatureExplainer("SHap", "iMage", model, data=tf.convert_to_tensor(background), shap_method="GRAdient", shap_summarize=None)
    explanation = exp.explain(np.array(x))
    shap_values = explanation[0]
    assert type(shap_values) is list
    assert len(shap_values) == 10
    assert shap_values[0].shape == (2, 28, 28, 1)

    with pytest.raises(RuntimeError): # does not work for tf 2.0 yet. If it works, test should be changed.
        exp = FeatureExplainer("shap", "image", model, data=tf.convert_to_tensor(background), shap_method="deep", shap_summarize=None)
        shap_values = exp.explain(np.array(x))


def test__is_tensor():
    assert not FeatureExplainer._is_tensor(1)
    assert not FeatureExplainer._is_tensor([1, 2])
    assert not FeatureExplainer._is_tensor("hafdsl")


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="pytorch not present")
def test__is_tensor_pytorch():
    import torch

    assert FeatureExplainer._is_tensor(torch.FloatTensor(1))
    assert FeatureExplainer._is_tensor(torch.FloatTensor([1]))
    assert FeatureExplainer._is_tensor(torch.FloatTensor([[2, 2], [3, 4]]))
    assert FeatureExplainer._is_tensor(torch.LongTensor([[2, 2], [3, 4]]))

    assert FeatureExplainer._is_tensor(FeatureExplainer._convert_to_tensor([[2, 2], [3, 4]], "pytorch"))
    assert FeatureExplainer._is_tensor(FeatureExplainer._convert_to_tensor([1], "pytorch"))


@pytest.mark.skipif(importlib.util.find_spec("tensorflow") is None, reason="tensorflow not present")
def test__is_tensor_tf():
    import tensorflow as tf
    assert FeatureExplainer._is_tensor(tf.convert_to_tensor(1))
    assert FeatureExplainer._is_tensor(tf.convert_to_tensor([1], dtype="float"))
    assert FeatureExplainer._is_tensor(tf.convert_to_tensor([[2, 2], [3, 4.1]]))

    assert FeatureExplainer._is_tensor(FeatureExplainer._convert_to_tensor(1, "tensorflow"))
    assert FeatureExplainer._is_tensor(FeatureExplainer._convert_to_tensor([[2, 2], [3, 4.1]], "tensorflow"))

def test_lime_then_shap_tabular(get_trained_dt, get_trained_linear):
    rf, X_train, _, _, _ = get_trained_dt
    lr, X_train, X_test, y_train, y_test = get_trained_linear

    n_samples = 10
    n_features = 30

    exp1 = FeatureExplainer(methods=["lime", "shap"], data_type="tabular", model=rf, data=X_train, shap_method="auto")
    explanation1 = exp1.explain(X_train.iloc[:n_samples])
    assert type(explanation1) is list
    assert len(explanation1) == 2
    assert type(explanation1[0]) is list
    assert type(explanation1[1]) is list

def test_shap_then_lime_tabular(get_trained_dt, get_trained_linear):
    rf, X_train, _, _, _ = get_trained_dt
    lr, X_train, X_test, y_train, y_test = get_trained_linear

    n_samples = 10
    n_features = 30

    exp2 = FeatureExplainer(methods=["shap", "lime"], data_type="tabular", model=rf, data=X_train, shap_method="auto")
    explanation2 = exp2.explain(X_train.iloc[:n_samples])
    assert type(explanation2) is list
    assert len(explanation2) == 2
    assert type(explanation2[0]) is list
    assert type(explanation2[1]) is list