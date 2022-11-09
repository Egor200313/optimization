import numpy as np
from typing import Callable


def calculate_fib_sequence(n: int) -> np.array:
    sequence = np.zeros(n+1)
    sequence[0] = 1
    sequence[1] = 1
    for i in np.arange(2, n + 1):
        sequence[i] = sequence[i-1] + sequence[i-2]
    return sequence


def optimize_one_dim(left_border: float, right_border: float, tolerance: float, max_iter: int, target_function: Callable, derivative: Callable = None):
    fibonacci = calculate_fib_sequence(max_iter)
    L, R = left_border, right_border
    for k in np.arange(1, max_iter):
        if R - L <= tolerance:
            return L
        middle1 = (R - L) * fibonacci[max_iter - k - 1] / fibonacci[max_iter - k + 1] + L
        middle2 = (R - L) * fibonacci[max_iter - k] / fibonacci[max_iter - k + 1] + L
        print(middle1, middle2, target_function(middle1), target_function(middle2))
        if target_function(middle1) > target_function(middle2):
            L = middle1
        else:
            R = middle2
    return L
