import numpy as np
from typing import Callable, TypedDict


class Result(TypedDict):
    """
    point: float
        middle of result interval
    tolerance: float
        length of result interval
    iterations: int
        number of loop iterations
    comp: int
        number of target function computations
    """
    point: float
    tolerance: float
    iterations: int
    comp: int


def calculate_fib_sequence(n: int) -> np.array:
    sequence = np.zeros(n+1)
    sequence[0] = 1
    sequence[1] = 1
    for i in np.arange(2, n + 1):
        sequence[i] = sequence[i-1] + sequence[i-2]
    return sequence


def optimize_one_dim(left_border: float,
                     right_border: float,
                     tolerance: float,
                     max_iter: int,
                     target_function: Callable,
                     derivative: Callable = None
                     ) -> Result:
    """
    optimizes one dimensional function using Fibonacci method
    :param left_border: left border of optimization interval
    :param right_border: right border of optimization interval
    :param tolerance: possible mistake as maximal result interval length
    :param max_iter: maximal number of iterations
    :param target_function: target function to optimize, must be one dimensional
    :param derivative: derivative of target function
    :return: result of optimization
    """

    fibonacci = calculate_fib_sequence(max_iter)
    L, R = left_border, right_border
    for k in np.arange(1, max_iter + 1):
        if R - L <= tolerance:
            return {
                    'point': round((R + L) / 2, 6),
                    'tolerance': R - L,
                    'iterations': k,
                    'comp': (k - 1) * 2
                    }
        middle1 = (R - L) * fibonacci[max_iter - k - 1] / fibonacci[max_iter - k + 1] + L
        middle2 = (R - L) * fibonacci[max_iter - k] / fibonacci[max_iter - k + 1] + L
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
