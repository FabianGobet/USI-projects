import ast
from nltk.metrics.distance import edit_distance
import sys
from typing import Union, Tuple
import os

true_branch_distances = {}
false_branch_distances = {}

def evaluate_condition(num: int, op: str, lhs: ast.expr, rhs: ast.expr) -> bool:
    """
    Computes the branch distance for a given condition and updates global variables.
    Returns True if the distance to the true branch is zero, otherwise False.
    """
    def are_strings(s1: Union[str, int] ,s2: Union[str, int]) -> bool:
        return isinstance(s1, str) and isinstance(s2, str)
    
    global true_branch_distances, false_branch_distances
    
    ops = {
        'Lt': lambda lhs, rhs: (lhs - rhs + 1 if lhs >= rhs else 0, rhs - lhs if lhs < rhs else 0),
        'Gt': lambda lhs, rhs: (rhs - lhs + 1 if lhs <= rhs else 0, lhs - rhs if lhs > rhs else 0),
        'LtE': lambda lhs, rhs: (lhs - rhs if lhs > rhs else 0, rhs - lhs + 1 if lhs <= rhs else 0),
        'GtE': lambda lhs, rhs: (rhs - lhs if lhs < rhs else 0, lhs - rhs + 1 if lhs >= rhs else 0),
        'Eq': lambda lhs, rhs: (edit_distance(lhs, rhs) if are_strings(lhs,rhs) else abs(lhs - rhs), 1 if lhs == rhs else 0),
        'NotEq': lambda lhs, rhs: (1 if lhs == rhs else 0, edit_distance(lhs, rhs) if are_strings(lhs,rhs) else abs(lhs - rhs))
    }

    if are_strings(lhs,rhs) and len(lhs) == len(rhs) == 1:
        lhs, rhs = ord(lhs), ord(rhs)

    if op not in ops:
        raise ValueError(f"Unsupported operation: {op}")

    true_dist, false_dist = ops[op](lhs, rhs)
    true_branch_distances[num] = min(true_branch_distances.get(num, float('inf')), true_dist)
    false_branch_distances[num] = min(false_branch_distances.get(num, float('inf')), false_dist)

    return true_dist == 0

def get_branch_distances(reset: bool = False) -> Tuple[dict, dict]:
    global true_branch_distances, false_branch_distances
    last_true, last_false = true_branch_distances, false_branch_distances
    if reset:
        true_branch_distances, false_branch_distances = {}, {}
    return last_true, last_false




class Instrumentor(ast.NodeTransformer):
    """
    AST Node Transformer to instrument functions by appending '_instrumented'
    to function names and replacing Compare nodes with evaluate_condition calls.
    """
    def __init__(self):
        super().__init__()
        self.instrumented_functions = {}
        self.counter = None
        self.num_nested_counters = []

    def _increment_counter(self) -> int:
        self.counter += 1
        return self.counter
    
    def _move_counter(self, new_func=True) -> None:
        if new_func:
            self.counter = 0 if len(self.num_nested_counters) == 0 else self.counter
            self.num_nested_counters.append(True)
        else:
            self.num_nested_counters.pop()
            self.counter = None if len(self.num_nested_counters) == 0 else self.counter

    def _recursive_f_id_helper(self, node: ast.AST) -> ast.AST:
        try:
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in self.instrumented_functions:
                node.func.id = self.instrumented_functions[node.func.id]
            for child in ast.iter_child_nodes(node):
                self._recursive_f_id_helper(child)
            return node
        except:
            print(ast.dump(node))
            sys.exit(1)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self._move_counter()
        original_name = node.name
        self.instrumented_functions[original_name] = original_name + "_instrumented"
        node.name = node.name + "_instrumented"
        self.generic_visit(node)
        del self.instrumented_functions[original_name]
        self._move_counter(new_func=False)
        return node
    
    def visit_Return(self, node: ast.Return) -> ast.Call:
        return self._recursive_f_id_helper(node)
    
    def visit_Assert(self, node: ast.Assert) -> ast.Call:
        return self._recursive_f_id_helper(node)
    
    def visit_Call(self, node: ast.Call) -> ast.Call:
        if isinstance(node.func, ast.Name) and node.func.id in self.instrumented_functions:
            node.func.id = self.instrumented_functions[node.func.id]
        self.generic_visit(node)
        return node

    def visit_Compare(self, node: ast.Compare) -> ast.Call:
        lhs = node.left
        rhs = node.comparators[0]
        op = type(node.ops[0]).__name__ 

        call = ast.Call(
            func=ast.Name(id='evaluate_condition', ctx=ast.Load()),
            args=[ast.Constant(value=self._increment_counter()), ast.Constant(value=op), lhs, rhs],
            # args=[ast.Constant(value=id(node)), ast.Constant(value=op), lhs, rhs],
            keywords=[]
        )
        return call

def instrument_file(input_file: str, output_file: str) -> None:
    with open(input_file, "r") as f:
        source_code = f.read()

    tree = ast.parse(source_code)
            
    instrumentor = Instrumentor()
    instrumented_tree = instrumentor.visit(tree)

    instrumented_code = ast.unparse(instrumented_tree)

    with open(output_file, "w") as f:
        f.write(f"from instrumentor import evaluate_condition, get_branch_distances\n\n")
        f.write(instrumented_code)


if __name__ == "__main__":

    assert 2<=len(sys.argv)<=3, \
        "Usage: python instrumentor.py input_files_dir [output file dir, default='./benchmark_instrumented/']"

    input_files_dir = os.path.normpath(sys.argv[1])
    if os.path.isdir(input_files_dir):
        input_files = [os.path.join(input_files_dir, f) for f in os.listdir(input_files_dir) if f.endswith(".py")]
    else:
        raise ValueError(f"Input directory {input_files_dir} does not exist.")
    

    if not os.path.exists(os.path.join(input_files_dir,"__init__.py")):
        with open(os.path.join(input_files_dir,"__init__.py"), "w") as f:
            pass
    
    output_files_dir = sys.argv[2] if len(sys.argv)==3 else os.path.normpath("./benchmark_instrumented/")
    if not os.path.exists(output_files_dir):
        os.makedirs(output_files_dir)
    
    for input_file_path in input_files:
        if input_file_path.split(os.sep)[-1].split(".")[0] == "__init__":
            continue
        output_file_path = os.path.join(output_files_dir, input_file_path.split(os.sep)[-1].split(".")[0] + "_instrumented.py")
        instrument_file(input_file_path, output_file_path)
