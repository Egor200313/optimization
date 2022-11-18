from one_dim.optimizer import optimize_one_dim
from typing import Callable
import numpy as np


def optimize_hs(tolerance: float,
                one_dim_tolerance: float,
                max_iter: int,
                dim: int,
                target_function: Callable,
                derivative: Callable = None
                ) -> dict:
    x_cur = np.ones(dim)
    d_cur = -derivative(x_cur)
    for i in np.arange(1, dim):
        alpha = optimize_one_dim(-100, 100, one_dim_tolerance, 10000,
                                 lambda x: target_function(x_cur + x * d_cur))['point']
        x_next = x_cur + alpha*d_cur
        beta = (derivative(x_next).T @ (derivative(x_next) - derivative(x_cur)))\
               / (d_cur.T @ (derivative(x_next) - derivative(x_cur)))
        d_next = -derivative(x_next) + beta*d_cur

        x_cur, d_cur = x_next, d_next
    return {'point': x_cur}
