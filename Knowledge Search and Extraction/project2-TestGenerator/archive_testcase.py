from typing import List, Tuple, Dict, Union

class TestCase:
    def __init__(self, arg_types: List[Union[str, int, Tuple[str, int]]], arg_names: List[str], arg_values: List[Union[int, str, Tuple[str, int]]]):
        self.arg_types = arg_types

        self.arg_names = arg_names
        self.arg_values = arg_values

        self.output = None
        self.input_size = len(arg_types)
    
    def get_types(self) -> List[Union[str, int, Tuple[str, int]]]:
        return self.arg_types
    
    def get_input_values(self):
        return self.arg_values
    
    def get_input_arg_names(self):
        return self.arg_names
    
    def set_output(self, output):
        self.output = [str(type(output)).split(" ")[1].split('>')[0].replace("'",'')], [output]

    def get_output(self):
        return self.output
    
    def _ensure_tuple(self, arg_values):
        return arg_values if isinstance(arg_values, tuple) else tuple(arg_values)

    def __len__(self):
        return self.input_size
    
    def __eq__(self, other):
        is_equal = True
        if not isinstance(other, TestCase):
            is_equal = False
        else:
            for i in range(self.input_size):
                if self.arg_values[i] != other.arg_values[i] or self.arg_types[i] != other.arg_types[i] or \
                    self.arg_names[i] != other.arg_names[i]:
                    is_equal = False
                    break
        return is_equal
    
    def __hash__(self):
        return hash(self._ensure_tuple(self.arg_values))
    
    

class Archive:
    def __init__(self, script_globals: dict):
        self.fn_to_num_branches = {} # map fn_name to amount of branches
        self.fn_to_current_true_branch_distance = {} # map fn_name to current true branch distances
        self.fn_to_current_false_branch_distance = {} # map fn_name to current true branch distances
        self.script_globals = script_globals
        self.fn_test_cases = {} # testcases stores for each function
        self.fn_to_disregarded_test_cases = {} # testcases that were disregarded for each function
        self.fn_to_fully_covered = {} # map fn_name to whether all branches are covered
    
    def add_fn(self, fn_name: str, num_branches: int):
        if fn_name not in self.fn_to_num_branches.keys():
            self.fn_to_num_branches[fn_name] = num_branches
            self.fn_to_current_true_branch_distance[fn_name] = {}
            self.fn_to_current_false_branch_distance[fn_name] = {}
            self.fn_test_cases[fn_name] = []
            self.fn_to_disregarded_test_cases[fn_name] = []
            self.fn_to_fully_covered[fn_name] = False
        else:
            if self.fn_to_num_branches[fn_name] != num_branches:
                raise ValueError(f"Function {fn_name} already exists in the archive with {self.fn_to_num_branches[fn_name]} branches.")


    def _is_fully_covered(self, fn_name: str):
        is_covered = True
        if len(self.fn_to_current_true_branch_distance[fn_name]) < self.fn_to_num_branches[fn_name] or \
            len(self.fn_to_current_false_branch_distance[fn_name]) < self.fn_to_num_branches[fn_name]:
            is_covered = False
        else:
            for k in range(1, self.fn_to_num_branches[fn_name] + 1):
                if self.fn_to_current_true_branch_distance[fn_name][k]!=0 or \
                    self.fn_to_current_false_branch_distance[fn_name][k]!=0:
                    is_covered = False
                    break
        self.fn_to_fully_covered[fn_name] = is_covered
        

    def _add_testcase(self, fn_name: str, testcase: TestCase, true_branch_distances: Dict[int, int], false_branch_distances: Dict[int, int]) -> bool:
        self.fn_test_cases[fn_name].append(testcase)
        for k,v in true_branch_distances.items():
            if k not in self.fn_to_current_true_branch_distance[fn_name].keys() or v < self.fn_to_current_true_branch_distance[fn_name][k]:
                self.fn_to_current_true_branch_distance[fn_name][k] = v
        for k,v in false_branch_distances.items():
            if k not in self.fn_to_current_false_branch_distance[fn_name].keys() or v < self.fn_to_current_false_branch_distance[fn_name][k]:
                self.fn_to_current_false_branch_distance[fn_name][k] = v
        return self._is_fully_covered(fn_name)

    def _is_improving(self, branch_distances: Dict[int, int], current_branch_distances: Dict[int, int]) -> bool:
        is_improving = False
        for k,v in branch_distances.items():
            if v < current_branch_distances[k]:
                is_improving = True
                break
        return is_improving

    def _evaluate_test_case(self, fn_name: str, testcase: TestCase, true_branch_distances: Dict[int, int], false_branch_distances: Dict[int, int]):
        new_covered_true_branches = set(true_branch_distances.keys()) - set(self.fn_to_current_true_branch_distance[fn_name].keys())
        new_covered_false_branches = set(false_branch_distances.keys()) - set(self.fn_to_current_false_branch_distance[fn_name].keys())
        
        if len(new_covered_true_branches)>0 or len(new_covered_false_branches)>0:
            self._add_testcase(fn_name, testcase, true_branch_distances, false_branch_distances)
        else:
            if self._is_improving(true_branch_distances, self.fn_to_current_true_branch_distance[fn_name]) or \
                self._is_improving(false_branch_distances, self.fn_to_current_false_branch_distance[fn_name]):
                self._add_testcase(fn_name, testcase, true_branch_distances, false_branch_distances)
            else:
                self.fn_to_disregarded_test_cases[fn_name].append(testcase)


    def consider_fn_testcase(self, fn_name: str, testcase: TestCase) -> bool:
        if fn_name not in self.fn_to_num_branches.keys():
            raise ValueError(f"Function {fn_name} not found in the archive.")
        
        if testcase not in set(self.fn_test_cases[fn_name] + self.fn_to_disregarded_test_cases[fn_name]):
            try:
                output = self.script_globals[fn_name](*testcase.get_input_values())
                if isinstance(output, str):
                    output = output.replace('\\', '\\\\').replace('"', '\\"')
                testcase.set_output(output)
                true_branch_distances, false_branch_distances = self.script_globals['get_branch_distances']()
                self._evaluate_test_case(fn_name, testcase, true_branch_distances, false_branch_distances)
            except AssertionError as e:
                self.fn_to_disregarded_test_cases[fn_name].append(testcase)
        return self.fn_to_fully_covered[fn_name]
    
    def _write_throughput_types_helper(self, arg_types: List[Union[str, int, Tuple[str, int]]], input_values: List[Union[int, str, Tuple[str, int]]], is_output: bool = False, delta = 1e-7) -> str:
        type_value_map = {
            'int': lambda x: f'{str(value)},',
            'bool': lambda x: f'{str(value)},',
            'float': lambda x: f'{str(value)},',
            'str': lambda x: f'"{str(x)}",',
            'list': lambda x: f'[{",".join(str(v) for v in x)}],',
            'kv': lambda x: f'"{str(x[0])}",{str(x[1])},'
        }

        h = ''
        if is_output:
            h += f'self.assert' + ('AlmostEqual' if 'float' in arg_types else 'Equal') + '(y,'

        if 'kv' in arg_types:
            h += type_value_map['kv'](input_values)
        else:
            for typ, value in zip(arg_types, input_values):
                h += type_value_map[typ](value)

        if is_output:
            if 'float' in arg_types:
                h += f'delta={delta}'
            h = h.removesuffix(',') + '),'

        return h.removesuffix(',')

    def set_fn_testcases(self, fn_name: str, testcases: List[TestCase]):
        self.fn_test_cases[fn_name] = testcases
    
    def dump(self, original_script_name: str, delta = 1e-7) -> str:
        original_script_name = original_script_name.replace('.py', '')
        print(f'Dumping test cases for {original_script_name}')
        dump = f'class Test_{original_script_name}(TestCase):\n'
        for fn_name, testcases in self.fn_test_cases.items():
            testcases = set(testcases)
            fn = fn_name.removesuffix('_instrumented')
            if len(testcases) == 0:
                dump +=f'\tdef test_{fn}_1(self):\n'
                dump += f'\t\tpass # No test cases found. Refine hyperparameters.\n'
            else:
                for i,tc in enumerate(testcases, 1):
                    dump += f'\tdef test_{fn}_{i}(self):\n'
                    dump += f'\t\ty = {fn}({self._write_throughput_types_helper(tc.get_types(),tc.get_input_values())})\n'
                    dump += f'\t\t{self._write_throughput_types_helper(*tc.get_output(), is_output=True, delta=delta)}\n\n'
        return dump
    
    def refactored_fn_names(self):
        return [fn.removesuffix('_instrumented') for fn in self.fn_test_cases.keys()]
    
    def get_fn_num_branches(self, fn_name: str) -> int:
        return self.fn_to_num_branches[fn_name]
