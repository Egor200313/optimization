from optimizer import optimize_one_dim
import numpy as np

def f(x):
    return (x-4.0)**2 + 6.0


def main():
    iters = 1000
    tolerance = np.arange(0.0001, 0.001, 0.0001)
    for tol in tolerance:
        result = optimize_one_dim(0.0, 100.0, tol, iters, f)
        print(result['iterations'])


if __name__ == "__main__":
    main()
