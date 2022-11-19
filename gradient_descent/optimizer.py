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
    
def gradient_descent(target_function: Callable,
		     derivative: Callable,
		     dim: int,
		     tolerance: float,
		     max_iter: int,
		     termination_criterion = "cauchy") -> dict:
    x_cur = np.ones(dim)
    d_cur = -derivative(x_cur)
    
    i = 0
    while i < max_iter:
    	x_next = x_cur + d_cur
    	d_next = -derivative(x_next)
    	
    	if termination_criterion == "cauchy":
    	    if abs(target_function(x_cur) - target_function(x_next)) < tolerance:
    	    	x_cur = x_next
    	    	break
    	elif termination_criterion == "cauchy_x":
    	    if sqrt((x_cur - x_next) @ (x_cur - x_next).T) < tolerance:
    	        x_cur = x_next
    	        break
    	else: # termination_criterion == "zero_gradient"
    	    if sqrt(derivative(x_next).T @ derivative(x_next)) < tolerance:
    	    	x_cur = x_next
    	    	break
    	
    	x_cur, d_cur = x_next, d_next
    	i += 1
    	
    return {'point': x_cur}
    
def hestens_stiefel(target_function: Callable,
		    derivative: Callable,
		    dim: int,
		    tolerance: float,
		    max_iter: int,
		    termination_criterion = "cauchy") -> dict:
    x_cur = np.ones(dim)
    d_cur = -derivative(x_cur)
    
    i = 0
    while i < max_iter:
    	x_next = x_cur + d_cur
    	
    	coef = (derivative(x_next).T @ (derivative(x_next) - derivative(x_cur)))\
    	       / (d_cur @ (derivative(x_next) - derivative(x_cur)).T)
    	d_next = -derivative(x_next) + coef * d_cur
    	
    	if termination_criterion == "cauchy":
    	    if abs(target_function(x_cur) - target_function(x_next)) < tolerance:
    	    	x_cur = x_next
    	    	break
    	elif termination_criterion == "cauchy_x":
    	    if sqrt((x_cur - x_next) @ (x_cur - x_next).T) < tolerance:
    	        x_cur = x_next
    	        break
    	else: # termination_criterion == "zero_gradient"
    	    if sqrt(derivative(x_next).T @ derivative(x_next)) < tolerance:
    	    	x_cur = x_next
    	    	break
    	
    	x_cur, d_cur = x_next, d_next
    	i += 1
    	
    return {'point': x_cur}

