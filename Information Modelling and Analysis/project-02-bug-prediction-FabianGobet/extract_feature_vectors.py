import javalang
import pandas as pd
import os
import sys

def get_mth(class_declaration: javalang.tree.ClassDeclaration) -> int:
    return len(class_declaration.methods)

def get_fld(class_declaration: javalang.tree.ClassDeclaration) -> int:
    return len(class_declaration.fields)

def get_rfc(class_declaration: javalang.tree.ClassDeclaration) -> int:
    num_public_methods = 0
    invoked_methods = 0
    for method in class_declaration.methods:
        if 'public' in method.modifiers:
            num_public_methods += 1
    for _,method_invocation in class_declaration.filter(javalang.tree.MethodInvocation):
        invoked_methods += 1
    return num_public_methods + invoked_methods

def get_interfaces(class_declaration: javalang.tree.ClassDeclaration) -> int:
    count = 0
    for i,j, in class_declaration.filter(javalang.tree.InterfaceDeclaration):
        count += 1
    return count

def get_ex(class_declaration: javalang.tree.ClassDeclaration) -> int:
    num_exceptions = 0
    for _,throws in class_declaration.filter(javalang.tree.ThrowStatement):
        num_exceptions += 1
    return num_exceptions

def get_ret(class_declaration: javalang.tree.ClassDeclaration) -> int:
    num_return = 0
    for _,s in class_declaration.filter(javalang.tree.ReturnStatement):
        num_return += 1
    return num_return

def get_bcm_wrd(class_declaration: javalang.tree.ClassDeclaration) -> int:
    blocks = 0
    num_words = 0
    for _,j in class_declaration:
        try:
            if j.documentation is not None:
                blocks += 1
                num_words += len(j.documentation.split())
        except:
            pass
    return blocks, num_words
    
def get_nml(class_declaration: javalang.tree.ClassDeclaration) -> int:
    ''' average length of method names'''
    num_methods = 0.0
    count = 0.0
    for _,method in class_declaration.filter(javalang.tree.MethodDeclaration):
        num_methods += 1
        count += len(method.name)
    return count/num_methods if num_methods > 0 else 0

def get_sz_cpx(class_declaration: javalang.tree.ClassDeclaration) -> int:
    num_statements = 0
    num_cpx = 0
    for _,statement in class_declaration.filter(javalang.tree.Statement):
        num_statements += 1
        if isinstance(statement, javalang.tree.IfStatement) or isinstance(statement, javalang.tree.WhileStatement) or isinstance(statement, javalang.tree.ForStatement) or isinstance(statement, javalang.tree.DoStatement) or isinstance(statement, javalang.tree.SwitchStatement):
            num_cpx += 1
    return num_statements, num_cpx


def get_class_metrics(class_declaration: javalang.tree.ClassDeclaration) -> dict:
    mth = get_mth(class_declaration)
    fld = get_fld(class_declaration)
    rfc = get_rfc(class_declaration)
    inter = get_interfaces(class_declaration)
    ex = get_ex(class_declaration)
    ret = get_ret(class_declaration)
    bcm, wrd = get_bcm_wrd(class_declaration)
    nml = get_nml(class_declaration)
    sz, cpx = get_sz_cpx(class_declaration)
    dcm = wrd/sz if sz > 0 else 0

    return {'mth': mth, 'fld': fld, 'rfc': rfc, 'int': inter, 'ex': ex, 'ret': ret, 'bcm': bcm, 'wrd': wrd, 'nml': nml, 'sz': sz, 'cpx': cpx, 'dcm': dcm}

def get_feature_vectors(directory_path: str, save_path = None) -> pd.DataFrame:
    df = pd.DataFrame(columns=['class_name', 'mth', 'fld', 'rfc', 'int', 'ex', 'ret', 'bcm', 'wrd', 'nml', 'sz', 'cpx', 'dcm', 'file_path'])
    for path_walk, _, files_walk in os.walk(directory_path):
        for file in files_walk:
            if file.endswith('.java'):
                with open(os.path.join(path_walk, file), 'r') as f:
                    tree = javalang.parse.parse(f.read())
                inner_classes = []
                for _,class_declaration in tree.filter(javalang.tree.ClassDeclaration):
                    name = class_declaration.name
                    if name not in inner_classes:
                        for _, inner_class in class_declaration.filter(javalang.tree.ClassDeclaration):
                            inner_classes.append(inner_class.name)
                        metrics = get_class_metrics(class_declaration)
                        idx = len(df)
                        metrics['class_name'] = name
                        metrics['file_path'] = os.path.join(path_walk, file)
                        for c in metrics.keys():
                            df.loc[idx, c] = metrics[c]
    if save_path is not None:
        df.to_csv(save_path, index=False)            
    return df

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python extract_feature_vectors.py <directory_path> <optional: save_path>')
        sys.exit(1)
    directory_path = sys.argv[1]
    save_path = sys.argv[2] if len(sys.argv) > 2 else './feature_vectors.csv'
    get_feature_vectors(directory_path, save_path)