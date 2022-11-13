import numpy as np
from typing import Callable


def calculate_fib_sequence(n: int) -> np.array:
    sequence = np.zeros(n+1)
    sequence[0] = 1
    sequence[1] = 1
    for i in np.arange(2, n + 1):
        sequence[i] = sequence[i-1] + sequence[i-2]
    return sequence


def fibonacci(n: int) -> int:
    phi = (1 + np.sqrt(5)) / 2
    return int((phi**n - (-phi)**(-n))/(2*phi - 1))


def optimize_one_dim(left_border: float,
                     right_border: float,
                     tolerance: float,
                     max_iter: int,
                     target_function: Callable,
                     derivative: Callable = None
                     ) -> dict:
    """
    optimizes one dimensional function using Fibonacci method

    Parameters:
        left_border: float
            left border of optimization interval

        right_border: float
            right border of optimization interval

        tolerance: float
            possible mistake as maximal result interval length

        max_iter: int
            maximal number of iterations

        target_function: func
            target function to optimize, must be one dimensional

        derivative: func
            derivative of target function

    Returns:
        result of optimization
    """

    L, R = left_border, right_border
    for k in np.arange(1, max_iter + 1):
        if R - L <= tolerance:
            return {
                    'point': round((R + L) / 2, 6),
                    'tolerance': R - L,
                    'iterations': k,
                    'comp': (k - 1) * 2
                    }
        middle1 = (R - L) * fibonacci(max_iter - k - 1) / fibonacci(max_iter - k + 1) + L
        middle2 = (R - L) * fibonacci(max_iter - k) / fibonacci(max_iter - k + 1) + L
        if target_function(middle1) > target_function(middle2):
            L = middle1
        else:
            R = middle2
    return {
            'point': round((R + L) / 2, 6),
            'tolerance': R - L,
            'iterations': max_iter,
            'comp': max_iter * 2
            }
