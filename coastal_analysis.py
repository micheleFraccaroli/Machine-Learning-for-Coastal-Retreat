import sys
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, KFold, StratifiedKFold
import numpy as np
from sklearn.impute import SimpleImputer
from skopt import BayesSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import pickle
import time
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import argparse
import tqdm


class CoastalAnalysis:
    def __init__(self, data_path, output_path, oversamp, hpo, test_size, target):
        self.data_path = data_path
        self.output_path = output_path
        self.oversample = oversamp
        self.hpo = hpo
        self.test_size = test_size
        self.target = target
    
    # Utility functions ----------------------------------------------------------------------------------------
    def preprocessing(self, df, feature_to_drop, strategy_missing_values, missing_values):
        features = df
        try:
            features = features.drop(columns=feature_to_drop)
            features = features.drop(columns=self.target)
        except Exception as e:
            print(e)
        target = df[self.target]
        target = target.fillna(0)

        if strategy_missing_values == "zero":
            features = features.fillna(0)
            # imp = SimpleImputer(missing_values=missing_values, strategy='constant', fill_value=0)
            # features = imp.fit_transform(features)
        elif strategy_missing_values == "mean":
            imp = SimpleImputer(missing_values=missing_values, strategy='mean')
            features = imp.fit_transform(features)
        elif strategy_missing_values == "median":
            imp = SimpleImputer(missing_values=missing_values, strategy='median')
            features = imp.fit_transform(features)
        elif strategy_missing_values == "nan":
            features = features.fillna(missing_values)

        return features, target

    def hyperparameter_optimization(self, pipeline, parameters, X_train, y_train, X_test, y_test, filename, n_iter=None):
        start = time.time()

        if self.hpo == 'grid_search':
            grid_obj = GridSearchCV(estimator=pipeline,
                                    param_grid=parameters,
                                    refit=True,
                                    #cv=cv,
                                    return_train_score=False,
                                    scoring='accuracy',
                                    verbose=0
                                    )
            
            grid_obj.fit(X_train, y_train, )

        elif self.hpo == 'random_search':
            grid_obj = RandomizedSearchCV(estimator=pipeline,
                                        param_distributions=parameters,
                                        n_iter=n_iter,
                                        #cv=cv,
                                        refit=True,
                                        scoring='accuracy',
                                        return_train_score=False,
                                        verbose=0
                                        )
            grid_obj.fit(X_train, y_train, )

        elif self.hpo == 'bayes_search':
            grid_obj = BayesSearchCV(
                estimator=pipeline,
                search_spaces=parameters,
                scoring='accuracy',
                # cv=cv,
                n_iter=n_iter,
                verbose=0
            )
            grid_obj.fit(X_train, y_train, )
        else:
            print('enter search method')
            return

        estimator = grid_obj.best_estimator_

        print("##### Results")
        print("Score best parameters: ", grid_obj.best_score_)
        print("Best parameters: ", grid_obj.best_params_)
        # print("Cross-validation Score Accuracy: ", cvs.mean())
        print("Test Score: ", estimator.score(X_test, y_test))
        print(classification_report(
            y_test, estimator.predict(X_test), target_names=["0", "1"]))

        print("Time elapsed: ", time.time() - start)
        cm = confusion_matrix(y_test, estimator.predict(X_test))
        disp = ConfusionMatrixDisplay(cm, display_labels=["0", "1"])
        disp.plot()
        plt.savefig(filename)

        results = [grid_obj.best_score_, estimator.score(
            X_test, y_test), time.time() - start]  # ,result.shape[0]]
        return results, estimator

    def print_decision_rules(self, rf, features_names):
        f = open("decision_path_forest_exp_v3.txt", "w")
        for tree_idx, est in enumerate(rf.estimators_):
            tree = est.tree_
            assert tree.value.shape[1] == 1  # no support for multi-output

            f.write('--------------\nTREE: {}\n'.format(tree_idx))

            iterator = enumerate(zip(
                tree.children_left, tree.children_right, tree.feature, tree.threshold, tree.value))
            for node_idx, data in iterator:
                left, right, feature, th, value = data

                # left: index of left child (if any)
                # right: index of right child (if any)
                # feature: index of the feature to check
                # th: the threshold to compare against
                # value: values associated with classes

                # for classifier, value is 0 except the index of the class to return
                class_idx = np.argmax(value[0])

                if left == -1 and right == -1:
                    f.write('{} LEAF: return class={}\n'.format(
                        node_idx, class_idx))
                else:
                    # f.write('{} NODE: if feature[{}] < {} then next={} else next={}\n'.format(node_idx, feature, th, left, right))
                    f.write('{} NODE: if {} < {} then next={} else next={}\n'.format(
                        node_idx, features_names[feature], th, left, right))
        f.close()
        
    def plot_feature_importances(self, clf, feature_names, filename):
        features_imp = []
        plt.figure(figsize=(150, 100))
        importances = clf.feature_importances_
        importances = importances[:30]
        indices = np.argsort(importances)
        plt.title('Feature Importances', fontsize=58)
        plt.barh(range(len(indices)),
                importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [feature_names[i]
                                        for i in indices], fontsize=58, ma='left')
        plt.xlabel('Relative Importance (MDI)', fontsize=58)
        plt.xticks(importances[indices], fontsize=58, rotation=45)
        plt.savefig(filename)

        for i in indices:
            if importances[i] != 0:
                features_imp.append(feature_names[i])

        return features_imp
    # ----------------------------------------------------------------------------------------------------------
    
    # Dataset --------------------------------------------------------------------------------------------------
    def dataset_preparation(self, feature_to_drop):
        data = pd.read_excel(self.data_path)
        missing_values = [" "]
        label = []

        X, y = self.preprocessing(data, feature_to_drop=feature_to_drop, strategy_missing_values="zero", missing_values=missing_values)

        features = list(X.columns)
        for i in y:
            if i < 0:
                label.append(0)
            else:
                label.append(1)

        if self.oversample:
            oversample = SMOTE()
            X, label = oversample.fit_resample(X, label)

        X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=self.test_size)

        return X_train, X_test, y_train, y_test, features
    # ----------------------------------------------------------------------------------------------------------
    
    # Machine Learning Models ----------------------------------------------------------------------------------
    def RandomForest(self, feature_to_drop, model_name="RF"):
        saved_model_name = self.output_path + model_name + ".pickle"
        rf_grid = {
            'criterion': ["gini", "entropy"],
            'max_depth': [2, 5, 7, 9, 11, None],
            'max_features': ["log2", "sqrt"],
            'n_estimators': [100, 150, 200],
            'min_samples_leaf': [1, 2],
            'min_samples_split': [2, 5],
        }
        
        X_train, X_test, y_train, y_test, features = self.dataset_preparation(
            feature_to_drop)
        
        path_cm = self.output_path + "CM_rf.png"
        clf = RandomForestClassifier()
        res, est = self.hyperparameter_optimization(
            clf, rf_grid, X_train, y_train, X_test, y_test, path_cm)

        with open(saved_model_name, 'wb') as handle:
            pickle.dump(est, handle)

        path_fi = self.output_path + "FI_rf.png"
        f_i = self.plot_feature_importances(est, features, path_fi)
        
    def XGBoost(self, feature_to_drop, model_name="XGBoost"):
        saved_model_name = self.output_path + model_name + ".pickle"
        xgb_grid = {
            "learning_rate": [0.1, 0.2, 0.3, 0.05, 0.4],
            "max_depth": [3, 4, 5, 6, 7, 8, 9],
            "n_estimators": [50, 100, 130, 150]
        }
        
        X_train, y_train, X_test, y_test, features = self.dataset_preparation(feature_to_drop)

        path_cm = self.output_path + "CM_xg.png"
        clf = GradientBoostingClassifier()
        res, est = self.hyperparameter_optimization(
            clf, xgb_grid, X_train, y_train, X_test, y_test, path_cm)

        with open(saved_model_name, 'wb') as handle:
            pickle.dump(est, handle)

        path_fi = self.output_path + "FI_xg.png"
        f_i = self.plot_feature_importances(
            est, features, path_fi)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("d", type=str, help="Dataset path")
    parser.add_argument("out", type=str, help="Output path")
    parser.add_argument("aug", type=bool, help="Oversample the dataset: True | False")
    parser.add_argument("tsize", type=float, help="Size of the test set")
    parser.add_argument("hpo", type=str, help="Type of hyper-parameter optimization. Choice between: 'grid_search' | 'random_search' | 'bayes_search'")
    parser.add_argument("t", type=str, help="Target feature")
    parser.add_argument("m", type=str, help="Model: RF for random forest or XG for XGBoost")
    
    args = parser.parse_args()
    
    ca = CoastalAnalysis(args.d, args.out, args.aug, args.hpo, args.tsize, args.t)
    feature_to_drop = ['transetto', 'Hs (m)', 'Q-SLOPE']
    if args.m == "RF":
        ca.RandomForest(feature_to_drop)
    elif args.m == "XG":
        ca.RandomForest(feature_to_drop)
    else:
        print("model not found")
