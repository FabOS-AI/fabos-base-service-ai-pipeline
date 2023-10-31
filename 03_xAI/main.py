#!/usr/bin/python
import numpy as np
import pandas as pd

from modules.surrogate import Surrogate
from modules.feature_explain import FeatureExplainer

from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import ensemble, tree, linear_model

RANDOM = 42


def run_lime_tabular():
    np.random.state = RANDOM
    bc_dict = load_breast_cancer()
    bc_data = np.concatenate(
        (bc_dict.data, bc_dict.target.reshape(-1, 1)), axis=1)
    df = pd.DataFrame(bc_dict.data, columns=bc_dict.feature_names.tolist())

    X_train, X_test, y_train, y_test = train_test_split(df, bc_dict.target, test_size=0.25, random_state=RANDOM)

    rf = tree.DecisionTreeClassifier()
    rf.fit(X_train, y_train)

    exp = FeatureExplainer("lime", "tabular", rf.predict_proba, data=X_train, class_names=np.unique(y_train))
    res = exp.explain(X_test.iloc[0:2], num_features=2)
    print(res)


def run_lime_image():
    from PIL import Image
    import torch.nn as nn
    import os, json
    from matplotlib import pyplot as plt

    import torch
    from torchvision import models, transforms
    from torch.autograd import Variable
    import torch.nn.functional as F
    from lime import lime_image
    from skimage.segmentation import mark_boundaries

    def get_image(path):
        with open(os.path.abspath(path), 'rb') as f:
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

    explainer = FeatureExplainer("lime", batch_predict, "image")
    mask = explainer.explain(np.array(pill_transf(img)), labels=[239], num_samples=100, num_features=1)
    print(mask)
    img_boundry1 = mark_boundaries(np.array(pill_transf(img))/255.0, mask)
    plt.imshow(img_boundry1)
    plt.show()


def run_surrogate():
    np.random.state = RANDOM

    # Load dataset
    bc_dict = load_breast_cancer()
    bc_data = np.concatenate((bc_dict.data, bc_dict.target.reshape(-1, 1)), axis=1)
    df = pd.DataFrame(bc_dict.data, columns=bc_dict.feature_names.tolist())

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        df, bc_dict.target, test_size=0.25)

    # Train MLP
    clf = MLPClassifier(hidden_layer_sizes=(20,), random_state=RANDOM)
    clf.fit(X_train, y_train)

    # Test MLP
    pred = clf.predict(X_test)
    print("Accuracy: {:.2f}".format(accuracy_score(y_test, pred)))

    # Train surrogate
    surrogate = Surrogate(bc_dict.target_names.tolist())
    result = surrogate.generate_surrogate(X_train, clf, 'dt')


def run_shap_tabular():
    np.random.state = RANDOM
    bc_dict = load_breast_cancer()
    bc_data = np.concatenate((bc_dict.data, bc_dict.target.reshape(-1, 1)), axis=1)
    df = pd.DataFrame(bc_dict.data, columns=bc_dict.feature_names.tolist())

    X_train, X_test, y_train, y_test = train_test_split(df, bc_dict.target, test_size=0.25)

    clf = linear_model.LogisticRegression()
    clf.fit(X_train, y_train)

    exp = FeatureExplainer("shap", "tabular", clf, data=X_train, shap_method="linear")
    explanation = exp.explain(X_train.iloc[:20])

    print(explanation)

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


def run_shap_images():
    from matplotlib import pyplot as plt
    from torchvision import transforms, datasets
    import shap

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    model = torch.load("data/model_mnist.pt")
    print(model)
    mnist_test = datasets.MNIST('../data/', train=False, transform=transform, target_transform=None)
    mnist_train = datasets.MNIST('../data/', train=True, transform=transform, target_transform=None)


    background = mnist_train.data[np.random.choice(mnist_train.data.shape[0], 200, replace=False)].float().reshape(-1, 1, 28, 28)
    x = mnist_test.data[0:5].float().reshape(-1, 1, 28, 28)


    #e = shap.DeepExplainer(model, background)
    #shap_values = e.shap_values(x)
    
    exp = FeatureExplainer("shap", "images", model, data=background, shap_method="gradient", shap_summarize="auto")
    print(len(exp.data))

    shap_values = exp.explain(x)
    
    shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values] #channels last
    
    
    shap.image_plot(shap_numpy, -x.float().numpy().reshape(-1, 28, 28))
    plt.show()


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



if __name__ == '__main__':

    #run_lime_tabular()
    run_lime_image()
    #run_surrogate()
    #run_shap_tabular()
    #run_shap_images()
    #run_shap_images_tf()