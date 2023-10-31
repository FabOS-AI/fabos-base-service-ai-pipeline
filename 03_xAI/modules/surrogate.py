
import os
import pydotplus
import ruleset as rs

from sklearn.tree import DecisionTreeClassifier, export_graphviz

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class Surrogate:
    def __init__(self, class_names):
        self.classes = class_names
        self.surrogate_fn = {
            'dt': self.build_dt,
            'rs': self.build_ruleSet
        }
        self.logDir = './output/'


    def build_dt(self, data, predictions, outFile):
        """
        Train surrogate decision tree
        @param data: DataFrame
        @param predictions: list
        @param outFile: string

        @return dt
        """
        feature_names = data.columns.to_list()
        dt = DecisionTreeClassifier()
        dt = dt.fit(data.to_numpy(), predictions)

        dot_data = export_graphviz(dt, out_file=None,
                                   feature_names=feature_names,
                                   class_names=self.classes,
                                   filled=True, rounded=True,
                                   special_characters=True)

        graph = pydotplus.graph_from_dot_data(dot_data)
        if not os.path.exists(self.logDir):
            os.makedirs(self.logDir)
        graph.write_pdf(self.logDir + outFile + '.pdf')
        print("Saved Decision Tree in: {}".format(self.logDir))

        return dt


    def build_ruleSet(self, df, predictions, outFile):
        """
        Train surrogate rule set
        @param data: DataFrame
        @param predictions: list
        @param outFile: string

        @return ruleset
        """
        model = rs.BayesianRuleSet(method='forest')
        model.fit(df, predictions)

        if not os.path.exists(self.logDir):
            os.makedirs(self.logDir)

        ruleLog = open(self.logDir + outFile + '.txt', 'w')

        for i, rule in enumerate(model.predicted_rules):
            ruleLog.write("Rule {}: \n{}\n".format(
                i+1, model.rule_explainations[rule][0]))

        ruleLog.close()
        print("Saved Ruleset in: {}".format(self.logDir))

        return model


    def generate_surrogate(self, df, model, surrogateType='dt', outFile='surrogate'):
        """
        Extracts a surrogate model from a given black-box model

        @param df: DataFrame
            \\TODO: how to handle standardized an non-standardized features?
        @param model: ML-Model with implemented predict function
        @param surrogateType: str ('dt'|'rs'):
            rs: ruleSet can handle only binary classification
            TODO: find a way to catct this

        @return surrogate
        """
        print('Generate Surrogate')

        predictions = model.predict(df)
        if predictions.shape[0] == len(df.index):
            return self.surrogate_fn[surrogateType](df, predictions, outFile)
        else:
            print('Wrong prediction shape!')
            return False
