import os
import sys
import javalang
import pandas as pd
from typing import List

def get_god_classes(path_to_directory: str) -> List[str]:
    df = generate_class_method_count_df(path_to_directory)
    df['is_god'] = df['number_of_methods'].apply(lambda x: x > df['number_of_methods'].mean() + 6*df['number_of_methods'].std())
    return df[df['is_god']]

def generate_class_method_count_df(path_to_directory: str) -> pd.DataFrame:
    df = pd.DataFrame(columns=['class_name', 'number_of_methods', 'path'])
    for path_walk, _, files_walk in os.walk(path_to_directory):
        for file in files_walk:
            if file.endswith('.java'):
                with open(os.path.join(path_walk, file), 'r') as f:
                    tree = javalang.parse.parse(f.read())
                for _, class_declaration in tree.filter(javalang.tree.ClassDeclaration):
                    number_of_methods = len(class_declaration.methods)
                    df.loc[len(df.index)] = [class_declaration.name, number_of_methods, os.path.join(path_walk, file).replace('\\', '/')]
    return df
    

if __name__ == '__main__': 
    path_to_directory = sys.argv[1]
    god_df = get_god_classes(path_to_directory)
    print(god_df['class_name'].tolist())

