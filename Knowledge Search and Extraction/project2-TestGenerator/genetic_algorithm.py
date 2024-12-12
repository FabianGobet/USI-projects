from deap import creator, base, tools, algorithms
from typing import List, Dict
import os
import argparse
from archive_testcase import Archive, TestCase
from tools import generate_tests
from event_generator import EventGenerator

# hyperparameters
HYPERPARAMETERS = {
    'NPOP': 200,
    'NGEN': 20,
    'INDMUPROB': 0.05,
    'MUPROB': 0.1,
    'CXPROB': 0.5,
    'TOURNSIZE': 3,
    'REPS': 3
}

def normalize(x: int) -> float:
    return x / (1.0 + x)

def get_fitness_fn(ind, archive: Archive, fn_name: str, arg_types: List[str], arg_names: List[str]):
    input = ind[0]
    
    testcase = TestCase(arg_types, arg_names, input)
    is_all_covered = archive.consider_fn_testcase(fn_name, testcase)
    
    fitness = 0.0
    if not is_all_covered:
        true_branch_distances, false_branch_distances = archive.script_globals['get_branch_distances']()

        for branch in range(1, archive.get_fn_num_branches(fn_name)+1):
            if branch in true_branch_distances and true_branch_distances[branch] != 0:
                fitness += normalize(true_branch_distances[branch])
            if branch in false_branch_distances and false_branch_distances[branch] != 0:
                fitness += normalize(false_branch_distances[branch])
    return fitness,

def crossover_out(ind1, ind2, event_generator: EventGenerator) -> tuple:
    parent1 = ind1[0]
    parent2 = ind2[0]
    offspring1, offspring2 = event_generator.crossover_inputs(parent1, parent2)
    ind1[0] = offspring1
    ind2[0] = offspring2
    return ind1, ind2

def mutate_out(ind, event_generator: EventGenerator) -> tuple:
    input = ind[0]
    mutated = event_generator.mutate_input(input)
    ind[0] = mutated
    return ind,
    
  
def rep_ga_generation(archive: Archive, fn_name: str, arg_types: List[str], arg_names: List[str], num_branches: int) -> Archive:
    
    if arg_types == ['str', 'int']:
        arg_types = ['kv']
        
    event_generator = EventGenerator(arg_types)
    archive.add_fn(fn_name, num_branches)
    
    if hasattr(creator, "Fitness"):
        del creator.Fitness
    if hasattr(creator, "Individual"):
        del creator.Individual
    
    creator.create("Fitness", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.Fitness)
    
    crossover = lambda ind1, ind2: crossover_out(ind1, ind2, event_generator)
    mutate = lambda ind: mutate_out(ind, event_generator)
    fn_fitness = lambda ind: get_fitness_fn(ind, archive, fn_name, arg_types, arg_names)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_input", event_generator.generate_random_input)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_input, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", fn_fitness)
    toolbox.register("mate", crossover)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selTournament, tournsize=HYPERPARAMETERS['TOURNSIZE'])
    
    coverage = 0
    testcases = []

    for _ in range(HYPERPARAMETERS['REPS']):
        archive.fn_test_cases[fn_name] = []
        archive.fn_to_disregarded_test_cases[fn_name] = []
        archive.script_globals['get_branch_distances'](reset=True)
        population = toolbox.population(n=HYPERPARAMETERS['NPOP'])
        algorithms.eaSimple(population, toolbox, cxpb=HYPERPARAMETERS['CXPROB'], mutpb=HYPERPARAMETERS['MUPROB'], ngen=HYPERPARAMETERS['NGEN'], verbose=False)
        true_branch_distances, false_branch_distances = archive.script_globals['get_branch_distances']()
        cov = len([True for _,v in true_branch_distances.items() if v==0]) + len([True for _,v in false_branch_distances.items() if v==0])
        if cov > coverage:
            coverage = cov
            testcases = archive.fn_test_cases[fn_name]

    archive.set_fn_testcases(fn_name, testcases)
    archive.script_globals['get_branch_distances'](reset=True)
    return archive


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Generate tests from instrumented scripts.')
    parser.add_argument('instrumented_files_dir_path', type=str, help='Path to directory containing instrumented files.')
    parser.add_argument('original_files_dir_path', type=str, help='Path to directory containing original files.')
    parser.add_argument('output_tests_dir_path', type=str, help='Path to directory where tests will be saved.')
    parser.add_argument("--delta", type=float, default=1e-7, help="Delta value for float comparison.")
    parser.add_argument("--kwargs", nargs="*", help="Additional keyword arguments in the form key=value, for genetic algorithm. See genetic_algorithm.py for defaults.")
    args = parser.parse_args()

    kwargs = {}
    if args.kwargs:
        for arg in args.kwargs:
            key, value = arg.split('=')
            if key.upper() in HYPERPARAMETERS:
                HYPERPARAMETERS[key.upper()] = float(value)
    
    generate_tests(rep_ga_generation, args.instrumented_files_dir_path, args.original_files_dir_path, args.output_tests_dir_path, delta=args.delta, **kwargs)