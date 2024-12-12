def f(a: int, b: int) -> int:
    if a > 0:
        if b < 0:
            return a
    if b > 0:
        if a < 0:
            return b
    if a > b:
        return a
    else:
        return b

def g(a: int, b: int) -> int:
    if a < b:
        return g(b, a)
    return a - b