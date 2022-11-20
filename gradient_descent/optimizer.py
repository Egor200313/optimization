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
	x_cur = np.zeros(dim)
	d_cur = -derivative(x_cur)
	for _ in np.arange(1, max_iter):
		alpha = optimize_one_dim(-100, 100, one_dim_tolerance, 10000,
                                 lambda x: target_function(x_cur + x * d_cur))['point']
		x_next = x_cur + alpha*d_cur

		if abs(d_cur.T @ (derivative(x_next) - derivative(x_cur))) < 0.0000000001:
			return {'point': x_cur}

		beta = (derivative(x_next).T @ (derivative(x_next) - derivative(x_cur)))\
               / (d_cur.T @ (derivative(x_next) - derivative(x_cur)))
		d_next = -derivative(x_next) + beta*d_cur

		x_cur, d_cur = x_next, d_next
	return {'point': x_cur}


def gradient_descent(target_function: Callable,
		     derivative: Callable,
		     dim: int,
		     x0,
		     tolerance: float,
		     one_dim_tolerance: float,
		     max_iter: int,
		     termination_criterion: str) -> dict:
	x_cur = x0
	d_cur = -derivative(x_cur)
    
	i = 0
	while i < max_iter:
		alpha = optimize_one_dim(-100, 100, one_dim_tolerance, 10000,
                                         lambda x: target_function(x_cur + x * d_cur))['point']
		x_next = x_cur + alpha * d_cur
		d_next = -derivative(x_next)
    	
		if termination_criterion == "cauchy":
			if abs(target_function(x_cur) - target_function(x_next)) < tolerance:
				return {'point': x_next}
		elif termination_criterion == "cauchy_x":
			if np.sqrt((x_cur - x_next) @ (x_cur - x_next).T) < tolerance:
				return {'point': x_next}
		elif termination_criterion == "zero_gradient":
			if np.sqrt(derivative(x_next).T @ derivative(x_next)) < tolerance:
				return {'point': x_next}
		else:
			if abs(d_cur.T @ (derivative(x_next) - derivative(x_cur))) < tolerance:
				return {'point': x_next}
    	
		x_cur, d_cur = x_next, d_next
		i += 1
    	
	return {'point': x_cur}
    
def hestenes_stiefel(target_function: Callable,
		    derivative: Callable,
		    dim: int,
		    x0,
		    tolerance: float,
		    one_dim_tolerance: float,
		    max_iter: int,
		    termination_criterion: str) -> dict:
	x_cur = x0
	d_cur = -derivative(x_cur)
    
	i = 0
	while i < max_iter:
		alpha = optimize_one_dim(-100, 100, one_dim_tolerance, 10000,
                                         lambda x: target_function(x_cur + x * d_cur))['point']
		x_next = x_cur + alpha * d_cur
    	
		coef = (derivative(x_next).T @ (derivative(x_next) - derivative(x_cur)))\
    	             / (d_cur.T @ (derivative(x_next) - derivative(x_cur)))
		d_next = -derivative(x_next) + coef * d_cur
    	
		if termination_criterion == "cauchy":
			if abs(target_function(x_cur) - target_function(x_next)) < tolerance:
				return {'point': x_next}
		elif termination_criterion == "cauchy_x":
			if np.sqrt((x_cur - x_next) @ (x_cur - x_next).T) < tolerance:
				return {'point': x_next}
		elif termination_criterion == "zero_gradient":
			if np.sqrt(derivative(x_next).T @ derivative(x_next)) < tolerance:
				return {'point': x_next}
		else:
			if abs(d_cur.T @ (derivative(x_next) - derivative(x_cur))) < tolerance:
				return {'point': x_next}
    	
		x_cur, d_cur = x_next, d_next
		i += 1
    	
	return {'point': x_cur}

