import re
import ast
import pandas as pd
import os
import sys

def is_blacklist(name):
    """
    Check if the given node name is in the blacklist.

    Parameters
    ----------
    name : str
        The name of the node.

    Returns
    ------
    bool
        True if the node is in the blacklist, False otherwise.
    """
    return (name == 'main') or (re.search(r'test', name, re.IGNORECASE)) or (name[0] == '_')

def get_comment_from_item(item):
    """
    Check if the given node name is in the blacklist.

    Parameters
    ----------
    name : item
        The item attribute of the node.

    Returns
    ------
    str
        String containing the comment of the node.
    """
    comment = ''
    for item_comment in item.body:
        if isinstance(item_comment, ast.Expr) and isinstance(item_comment.value, ast.Str):
            comment = item_comment.value.s
            break
    return comment.replace('\n', ' ').strip()

def extract_info_from_node(node, data, file_path):
    """
    Check if the given node name is in the blacklist.

    Parameters
    ----------
    name : node
        The AST node.
    name : data
        A dictionary with keys 'functions', 'classes', and 'methods'.

    Returns
    ------
    pandas dataframe
        pandas dataframe containing all the organized information.
    """
    for item in node.body:
        is_func = isinstance(item, ast.FunctionDef)
        is_class = isinstance(item, ast.ClassDef)
        if (is_func or  is_class) and not(is_blacklist(item.name)):
            comment = get_comment_from_item(item)
            if is_func:
                data['functions'].append((item.name, file_path, item.lineno, comment)) 
            elif is_class:
                data['classes'].append((item.name, file_path, item.lineno, comment))
                for class_item in item.body:
                    if isinstance(class_item, ast.FunctionDef) and not(is_blacklist(class_item.name)):
                        class_comment = get_comment_from_item(class_item)
                        data['methods'].append((class_item.name, file_path, class_item.lineno, class_comment))
    return data


def extract_info_from_file(dir_path, save_csv_path=None):
    """
    Extract information from a given file.

    Parameters
    ----------
    file_path : str
        The file path.

    Returns
    ------
    pandas dataframe
        pandas dataframe containing all the organized information.
    """
    file_paths = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.py'):
                file_paths.append(os.path.join(root, file))

    data = {'functions': [], 'classes': [], 'methods': []}
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            node = ast.parse(f.read())
            data = extract_info_from_node(node, data, file_path)

    df = pd.DataFrame(columns=['name', 'file', 'line', 'type', 'comment'])
    for key, value in data.items():
        for item in value:
                loca = re.sub(r'\\', '/', item[1])
                df.loc[len(df)] = [item[0], loca, item[2], key, item[3]]

    if save_csv_path:
        df.to_csv(save_csv_path+'data.csv', index=False)
    return df

if __name__ == '__main__':
    if len(sys.argv)==2:
        extract_info_from_file(sys.argv[1])
    elif len(sys.argv)==3:
        extract_info_from_file(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python extract-data.py path_to_dir [save_csv_path]")
        print("Example: python extract-data.py ./tensorflow ./generated_files/")
        sys.exit()