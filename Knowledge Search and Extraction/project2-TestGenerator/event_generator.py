import random
from typing import Tuple, Union, List
import string

INPUT_TYPES = ("int", "str", "kv")

# HYPERPARAMETERS
MIN_INT = -100
MAX_INT = 100
MAX_STRING_LENGTH = 20

class EventGenerator:
    def __init__(self, input_types: List[str],  min_int: int = None, max_int: int = None, max_string_length: int = None):
        assert all([len(input_types)==1 or 'kv' not in input_types, \
            len(set(input_types))==1 and 1<=len(input_types)<=3]), "Input types must be either homogeneous with 1 to 3 elements, or singleton with 'kv'"
        assert all([in_type in INPUT_TYPES for in_type in input_types]), f"Unsupported input type, use only {INPUT_TYPES}"
        
        self.input_types = input_types
        self.min_int = MIN_INT if min_int is None else min_int
        self.max_int = MAX_INT if max_int is None else max_int
        self.max_string_length = MAX_STRING_LENGTH if max_string_length is None else max_string_length
        

    def _random_int_initializor(self) -> int:
        return random.randint(self.min_int, self.max_int)

    def _random_string_initializor(self) -> str:
        return ''.join(random.choices(string.ascii_lowercase, k=random.randint(0, self.max_string_length)))
    
    def _random_kv_initializor(self) -> Tuple[str, int]:
        return self._random_string_initializor(), self._random_int_initializor()
    
    def generate_random_input(self) -> Tuple:
        input = [] if 'kv' not in self.input_types else None
        for in_type in self.input_types:
            match in_type:
                case "int":
                    input.append(self._random_int_initializor())
                case "str":
                    input.append(self._random_string_initializor())
                case "kv":
                    input = self._random_kv_initializor()
        return input
    
    def _mutate_string(self, input_value: str) -> str:
        new_str = input_value
        if len(input_value) > 0:
            idx = random.randint(0, len(input_value) - 1)
            new_str = input_value[:idx] + random.choice(string.ascii_lowercase) + input_value[idx + 1:]
        return new_str
    
    def _mutate_kv(self, input_value: Tuple[str, int]) -> Tuple[str, int]:
        key, value = input_value
        if random.choice([True, False]):
            key = self._mutate_string(key)
        else:
            value = self._random_int_initializor()
        return key, value
    
    def _mutate_single_input(self, input_value: Union[int, str, Tuple[str, int]], input_type: str) -> Union[int, str, Tuple[str, int]]:
        new_value = input_value
        match input_type:
            case "int":
                new_value = self._random_int_initializor()
            case "str":
                new_value = self._mutate_string(input_value)
            case "kv":
                new_value = self._mutate_kv(input_value)
        return new_value
    
    def mutate_input(self, input: Union[int, str, Tuple[str, int]]) -> Union[int, str, Tuple[str, int]]:
        new_input = []
        for i, in_type in enumerate(self.input_types):
            if in_type == 'kv':
                new_input = self._mutate_single_input(input, in_type)
            else:
                new_input.append(self._mutate_single_input(input[i], in_type))
        return new_input

    def _crossover_strings(self, str0: str, str1: str) -> str:
        tail_idx = max(0, min(len(str0), len(str1))-1)
        return str0[:tail_idx] + str1[tail_idx:], str1[:tail_idx] + str0[tail_idx:]
    
    def _ensure_list(self, obj):
        return obj if isinstance(obj, list) else [obj]
    
    def crossover_inputs(self, input0: Tuple, input1: Tuple) -> Tuple:
        new_inputs = [None,None]
        match self.input_types[-1]:
            case "int":
                new_inputs[0] = self._ensure_list(input0[:-1]) + [input1[-1]]
                new_inputs[1] = self._ensure_list(input1[:-1]) + [input0[-1]]
            case "str":
                str_idx0 = random.sample(range(len(self.input_types)), 1)[0]
                str_idx1 = random.sample(range(len(self.input_types)), 1)[0]
                str0 = input0[str_idx0]
                str1 = input1[str_idx1]
                new0, new1 = self._crossover_strings(str0, str1)    
                new_inputs[0] = input0[:str_idx0] + [new0] + input0[str_idx0 + 1:]
                new_inputs[1] = input1[:str_idx1] + [new1] + input1[str_idx1 + 1:]
            case "kv":
                k0,v0 = input0
                k1,v1 = input1
                newk0, newk1 = self._crossover_strings(k0, k1)
                new_inputs[0] = newk0, v0
                new_inputs[1] = newk1, v1
        return new_inputs