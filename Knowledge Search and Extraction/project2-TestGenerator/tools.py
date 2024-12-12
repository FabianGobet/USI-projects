import os
from archive_testcase import Archive
import ast
from typing import Callable

def function_def_info(node: ast.FunctionDef) -> list:
    arg_names, arg_types = [], []
    for arg in node.args.args:
        arg_names.append(arg.arg)
        arg_types.append(arg.annotation.id)
    num_compares = 0
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == 'evaluate_condition':
            if child.args[0].value > num_compares:
                num_compares = child.args[0].value

    # [node_name, [(var_name,var_type),...] , num_compares]
    return node.name, arg_names, arg_types, num_compares

def get_instrumented_archive(generation_func: Callable, instrumented_script_path: str, **kwargs) -> Archive:
    with open(instrumented_script_path, "r") as f:
        source_code = f.read()

    tree = ast.parse(source_code)
    function_def_infos = [function_def_info(node) for node in tree.body if isinstance(node, ast.FunctionDef)]

    script_globals = {}
    exec(source_code, script_globals)
    archive = Archive(script_globals)

    for fn_name, arg_names, arg_types, num_compares in function_def_infos:
        archive = generation_func(archive, fn_name, arg_types, arg_names, num_compares, **kwargs)
        script_globals['get_branch_distances'](reset=True)
    return archive

def generate_tests(generation_func: Callable, instrumented_files_dir_path: str, original_files_dir_path: str, output_tests_dir_path: str, delta: float = 1e-7, **kwargs) -> None:
    assert os.path.isdir(instrumented_files_dir_path), f"{instrumented_files_dir_path} is not a directory."
    assert os.path.isdir(original_files_dir_path), f"{original_files_dir_path} is not a directory."

    output_tests_dir_path = os.path.normpath(output_tests_dir_path)
    instrumented_files_dir_path = os.path.normpath(instrumented_files_dir_path)
    original_files_dir_path = os.path.normpath(original_files_dir_path)
    
    if not os.path.isdir(output_tests_dir_path):
        os.mkdir(output_tests_dir_path)
    
    
    instrumented_files = [os.path.join(instrumented_files_dir_path, f) for f in os.listdir(instrumented_files_dir_path) if f.endswith("_instrumented.py")]

    for instrumented_file in instrumented_files:
        name = instrumented_file.split(os.sep)[-1].removesuffix('_instrumented.py')
        archive = get_instrumented_archive(generation_func, instrumented_file, **kwargs)
        original_file_path = os.path.join(original_files_dir_path, name+'.py')
        assert os.path.isfile(original_file_path), f"Original file {original_file_path} not found."
        with open(os.path.join(output_tests_dir_path, name+'_tests.py'), "w") as f:
            f.write('from unittest import TestCase\n')
            f.write(f'from {original_file_path.removesuffix(".py").replace(os.sep,".")} import {", ".join(archive.refactored_fn_names())}\n\n')
            f.write(archive.dump(name,delta=delta))
        del archive
