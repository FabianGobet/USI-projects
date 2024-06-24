import pandas as pd
import os
import sys
from extract_feature_vectors import get_feature_vectors

def get_buggy_classes(buggy_classes_dir_path):
    buggy_classes = []
    for path_walk, _, files_walk in os.walk(buggy_classes_dir_path):
            for file in files_walk:
                if file.endswith('.src'):
                    with open(os.path.join(path_walk, file), 'r') as f:
                        buggy_classes.append(f.read().split('.')[-1].strip('\n'))
    return buggy_classes

def label_feature_vectors(feature_vectors_csv_path, buggy_classes, save_path = None):
    df = pd.read_csv(feature_vectors_csv_path)
    df['buggy'] = 0
    for i in range(len(df)):
        if df.loc[i, 'class_name'] in buggy_classes:
            df.loc[i, 'buggy'] = 1
    if save_path is not None:
        df.to_csv(save_path, index=False)
    return df


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python extract_feature_vectors.py <feature vectors csv path> <modified(buggy) classes dir path>')
        sys.exit(1)
    csv_path = sys.argv[1]
    buggy_classes = sys.argv[2] 
    label_feature_vectors(csv_path, buggy_classes, save_path = './labeled_feature_vectors.csv')
