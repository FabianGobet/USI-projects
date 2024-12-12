from instrumentor import evaluate_condition, get_branch_distances

def f_instrumented(a: int, b: int) -> int:
    assert a > 0 and b > 0
    if evaluate_condition(1, 'Gt', a, b):
        return a > b
    else:
        return a > b