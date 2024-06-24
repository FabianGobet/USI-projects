import javalang
import pandas as pd
from typing import List
import sys

def get_fields(class_declaration: javalang.tree.ClassDeclaration) -> set[str]:
    set_fields = set()
    for m in class_declaration.fields:
        set_fields.add(m.declarators[0].name)
    return set_fields

def get_methods(class_declaration: javalang.tree.ClassDeclaration) -> set[str]:
    set_methods = set()
    for m in class_declaration.methods:
        set_methods.add(m.name)
    return set_methods

def get_fields_accessed_by_method(method_declaration: javalang.tree.MethodDeclaration) -> set[str]:
    set_field_accesses = set()
    for _,p in method_declaration.filter(javalang.tree.MemberReference):
        set_field_accesses.add(p.qualifier if p.qualifier != '' else p.member)
    return set_field_accesses

def get_methods_accessed_by_method(method_declaration: javalang.tree.MethodDeclaration) -> set[str]:
    set_method_accesses = set()
    for _,p in method_declaration.filter(javalang.tree.MethodInvocation):
        set_method_accesses.add(p.member)
    return set_method_accesses


def generate_feature_dataframe(node: javalang.tree.ClassDeclaration, set_class_methods: set, set_class_fields: set) -> pd.DataFrame:
    features = set()
    features.update(set_class_fields)
    features.update(set_class_methods)
    features = list(features)
    df = pd.DataFrame(columns=['method_name']+features)
    for m in list(set_class_methods):
        df.loc[len(df)] = {'method_name': m}
    for m in node.methods:
        method_name = m.name
        method_features = set()
        method_features = method_features.union(get_fields_accessed_by_method(m))
        method_features = method_features.union(get_methods_accessed_by_method(m))
        for f in list(method_features):
            if f in features:
                if not df['method_name'].isin([method_name]).any():
                    df.loc[len(df)] = {'method_name': method_name}
                df.loc[df['method_name'] == method_name, f] = 1
    return df


def extract_feature_vectors(path_java_file: str, save_directory_path: str = './') -> pd.DataFrame:
    with open(path_java_file, 'r') as f:
        tree = javalang.parse.parse(f.read())
    class_name = path_java_file.split('/')[-1].split('.')[0]
    class_features = {}
    for _,n in tree.filter(javalang.tree.ClassDeclaration):
        if(n.name == class_name):
            df = generate_feature_dataframe(n, get_methods(n), get_fields(n))
            df = df.fillna(0)
            column_names = df.columns.difference(['method_name'])
            df[column_names] = df[column_names].astype(int)
            if not save_directory_path.endswith('/') != './':
                save_directory_path = save_directory_path+'/'
            df.to_csv(save_directory_path+class_name+'.csv', index=False)
            class_features[class_name] = df
    return class_features


if __name__ == '__main__': 
    path_to_java_file = sys.argv[1]
    if len(sys.argv)>2:
        extract_feature_vectors(path_to_java_file, sys.argv[2])
    else:
        extract_feature_vectors(path_to_java_file)