import matplotlib.pyplot as plt
from train_classifiers import train_classifiers
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
import numpy as np
import pandas as pd
import sys

def evaluate_classifiers(labeled_feature_csv):
    classifiers = train_classifiers('../generated/class_metrics_labeled.csv')

    classifications = []

    for _ in range(20):
        classifications.append(train_classifiers('../generated/class_metrics_labeled.csv'))
        
    a = ['dt', 'nb', 'svm', 'mlp', 'rf', 'always_buggy']
    b = ['precision', 'recall', 'f1']

    summary = {}

    for i in a:
        summary[i] = {}
        for j in b:
            summary[i][j] = []

    for classification in classifications:
        for algo in classification.keys():
            for v in classification[algo]['scores']['test_precision']:
                summary[algo]['precision'].append(v)
            for v in classification[algo]['scores']['test_recall']:
                summary[algo]['recall'].append(v)
            for v in classification[algo]['scores']['test_f1']:
                summary[algo]['f1'].append(v)

    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    fig.suptitle('Precision, Recall and F1 Scores for 6 Classifiers')
    for i, metric in enumerate(b):
        data = []
        for algo in a:
            data.append(summary[algo][metric])
        axs[i].boxplot(data)
        axs[i].set_title(metric)
        axs[i].set_xticklabels(a)
        axs[i].set_ylabel('Score')
        axs[i].set_xlabel('Classifier')
    plt.subplots_adjust(hspace = 0.5)
    plt.savefig('../generated/boxplots.png')
    plt.show()
    
    def compare_dists(values1, values2, name1, name2, save=False):
        stat, p = wilcoxon(values1, values2)
        fig = plt.figure(figsize =(10, 5))
        plt.boxplot([values1, values2])
        plt.xticks([1, 2], [name1, name2])
        #plt.figtext(0.15, 0.8, f'stat: {stat:.2f}')
        plt.figtext(0.15, 0.75, f'p: {p:.6f}')
        if save:
            plt.savefig(f'../generated/{name1}_{name2}.png')
        plt.show()
        return [np.mean(values1), np.std(values1)], [np.mean(values2), np.std(values2)], p

    name_map = {
        'dt': 'Decision Tree',
        'nb': 'Naive Bayes',
        'svm': 'SVM',
        'mlp': 'MLP',
        'rf': 'Random Forest',
        'always_buggy': 'Always Buggy'
    }

    name_map_inverse = {v: k for k,v in name_map.items()}

    keys = list(summary.keys())
    info = {
        'pvals': {}
    }
    df = pd.DataFrame(columns=['method','Decision Tree', 'Naive Bayes', 'SVM', 'MLP', 'Random Forest', 'Always Buggy'])
    df['method'] = ['Decision Tree', 'Naive Bayes', 'SVM', 'MLP', 'Random Forest', 'Always Buggy']

    df2 = pd.DataFrame(columns=['metric','Decision Tree', 'Naive Bayes', 'SVM', 'MLP', 'Random Forest', 'Always Buggy'])
    df2['metric'] = ['f1 mean','f1 std']

    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            a,b,c = compare_dists(summary[keys[i]]['f1'], summary[keys[j]]['f1'], name_map[keys[i]], name_map[keys[j]], save=True)
            row = name_map[keys[i]]
            column = name_map[keys[j]]
            df.loc[df['method'] == row, column] = c
            df2.loc[df2['metric'] == 'f1 mean', row] = a[0]
            df2.loc[df2['metric'] == 'f1 mean', column] = b[0]
            df2.loc[df2['metric'] == 'f1 std', row] = a[1]
            df2.loc[df2['metric'] == 'f1 std', column] = b[1]
            df2.loc[df2['metric'] == 'precision mean', row] = np.mean(summary[keys[i]]['precision'])
            df2.loc[df2['metric'] == 'precision mean', column] = np.mean(summary[keys[j]]['precision'])
            df2.loc[df2['metric'] == 'precision std', row] = np.std(summary[keys[i]]['precision'])
            df2.loc[df2['metric'] == 'precision std', column] = np.std(summary[keys[j]]['precision'])
            df2.loc[df2['metric'] == 'recall mean', row] = np.mean(summary[keys[i]]['recall'])
            df2.loc[df2['metric'] == 'recall mean', column] = np.mean(summary[keys[j]]['recall'])
            df2.loc[df2['metric'] == 'recall std', row] = np.std(summary[keys[i]]['recall'])
            df2.loc[df2['metric'] == 'recall std', column] = np.std(summary[keys[j]]['recall'])
    df2.to_csv('../generated/metrics_stats.csv', index=False)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python evaluate_classifiers.py <labeled feature vectors csv path>')
        sys.exit(1)
    labeled_feature_csv = sys.argv[1]
    evaluate_classifiers(labeled_feature_csv)

    