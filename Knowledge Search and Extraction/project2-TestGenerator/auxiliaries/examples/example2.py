def f(a: int, b: int) -> int:
    if a < b:
        return f(b, a)
    return a - b