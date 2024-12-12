from instrumentor import evaluate_condition, get_branch_distances

def f_instrumented(a: int, b: int) -> int:
    if evaluate_condition(1, 'Lt', a, b):
        return f_instrumented(b, a)
    return a - b