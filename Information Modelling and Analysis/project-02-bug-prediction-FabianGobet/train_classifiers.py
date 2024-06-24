from sklearn.model_selection import cross_validate
import javalang
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
import sys


from sklearn.model_selection import cross_validate
from sklearn.utils import shuffle
from sklearn.base import BaseEstimator, ClassifierMixin

class AlwaysBuggy(BaseEstimator, ClassifierMixin):
        def fit(self, X, Y):
            self.classes_ = np.unique(Y)
            return self

        def predict(self, X):
            return [1 for _ in range(len(X))]

        def get_params(self, deep=True):
            return {}

        def set_params(self, **parameters):
            for parameter, value in parameters.items():
                setattr(self, parameter, value)
            return self

def train_classifiers(labeled_feature_csv):
    df_labeled = pd.read_csv(labeled_feature_csv)
    df_labeled.drop(columns=['file_path'], inplace=True)

    # 5-fold split
    X = df_labeled.drop(columns=['class_name', 'buggy'])
    Y = df_labeled['buggy']

    dt = DecisionTreeClassifier(criterion = 'gini', max_depth = 60, min_samples_leaf = 8, min_samples_split = 2, splitter = 'random')
    nb = GaussianNB()
    svm = SVC(C = 10, gamma = 0.0001, kernel = 'rbf')
    mlp = MLPClassifier(activation = 'relu', alpha = 0.05, hidden_layer_sizes = (12, 50, 25), learning_rate = 'constant', solver = 'adam')
    rf = RandomForestClassifier(criterion = 'gini', max_depth = 8, max_features = 'sqrt', n_estimators = 10)


    classifiers = {
        'dt': {'classifier': dt, 'scores': 0},
        'nb': {'classifier': nb, 'scores': 0},
        'svm': {'classifier': svm, 'scores': 0},
        'mlp': {'classifier': mlp, 'scores': 0},
        'rf': {'classifier': rf, 'scores': 0},
        'always_buggy': {'classifier': AlwaysBuggy(), 'scores': 0}
    }

    X,Y = shuffle(X,Y)    

    for classifier in classifiers.keys():
        scores = cross_validate(classifiers[classifier]['classifier'], X, Y, cv=5, scoring=('precision', 'recall', 'f1'), return_train_score=False)
        classifiers[classifier]['scores'] = scores
    
    return classifiers


def print_scores(classifiers):
    for classifier in classifiers.keys():
        print(f'{classifier} classifier:')
        print(f'Precision: {classifiers[classifier]["scores"]["test_precision"].mean()}')
        print(f'Recall: {classifiers[classifier]["scores"]["test_recall"].mean()}')
        print(f'F1: {classifiers[classifier]["scores"]["test_f1"].mean()}')
        print()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python evaluate_classifiers.py <labeled_feature_csv>')
        sys.exit(1)
    labeled_feature_csv = sys.argv[1]
    classifiers = train_classifiers(labeled_feature_csv)
    print_scores(classifiers)