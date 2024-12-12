from instrumentor import evaluate_condition, get_branch_distances

def f_instrumented(a: int, b: int) -> int:
    if evaluate_condition(1, 'Gt', a, 0):
        if evaluate_condition(2, 'Lt', b, 0):
            return a
    if evaluate_condition(3, 'Gt', b, 0):
        if evaluate_condition(4, 'Lt', a, 0):
            return b
    if evaluate_condition(5, 'Gt', a, b):
        return a
    else:
        return b

def g_instrumented(a: int, b: int) -> int:
    if evaluate_condition(1, 'Lt', a, b):
        return g_instrumented(b, a)
    return a - b