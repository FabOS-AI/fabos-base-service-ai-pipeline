# xAI

## Already existing tool boxes

- [interpret](https://github.com/microsoft/interpret) by Microsoft, build on top of lime and shap
- [AIX360](http://aix360.mybluemix.net/) by IBM
- [DiCE](https://github.com/microsoft/DiCE) by Microsoft for counterfactual explanation of tabular data
- [Contrastive Explanation Method](https://github.com/IBM/Contrastive-Explanation-Method) by IBM for image data
- [lime](https://github.com/marcotcr/lime)
- [LRP toolbox](https://github.com/sebastian-lapuschkin/lrp_toolbox)
- [iNNvestigate](https://github.com/albermax/innvestigate) based on LRP and others
- [SHAP](https://github.com/slundberg/shap) based on shapley values
- [LORE](https://github.com/riccotti/LORE) lime alternative
- [treeinterpreter](https://github.com/andosa/treeinterpreter) for decision tree or random forrest, already pretty old
- [DeepExplain](https://github.com/marcoancona/DeepExplain) includes DeepLIFT, LRP, Shapley Values
- [Interpretation by meaningful perturbation](https://github.com/jacobgil/pytorch-explain-black-box) for CNNs
- [skater](https://oracle.github.io/Skater/reference/interpretation.html#overview) Feature Importance, Partial Dependence, Tree surrogates, BRL, LIME, DeepInterpreter, e-LRP, Integrated Gradient, Occlusion

## Possible package structure

- Counterfactuals sub module
- sub module for
  * surrogate model extraction
  * feature importance (include several methods to compare them)
  * partial dependence
- fit tree feature
- out of the box plotting methods


additionally later on:
- LO regularization sub module
- simple CNN interpretable feature extraction


#### counterfactuals
```python
from CCIexplain import counterfactuals

my_cfs = counterfactuals.generate_cfs(model,
                            x0,
                            number_cfs,
                            feature_list,
                            features_to_vary,
                            variation_range,
                            desired_class)

my_cfs.visualize_as_df()
```

#### feature importance
```python
from CCIexplain import feature_explain

explainer = feature_explain.TreeExplainer(scikit_rf_model,
                        feature_list, method='lime')# shap, lore, LRP
feature_importance = explainer.feature_importance(x0)

# partial dependence
```

#### fit tree feature
```python
from CCIexplain import fit_tree

dt, fitting_results = fit_tree(model, x_train, depth_max=17, pruning=True)
```

## Coding Guidelines
Used guidelines for writing source code and code documentation are based on 
- PEP-8 Coding Convention: https://pep8.org/
- PEP-257 Docstring Convention: https://www.python.org/dev/peps/pep-0257/


