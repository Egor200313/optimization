from optimizer import optimize_hs
import numpy as np


# Первая функция в предположении что размерность 2
L = 100
mu = 0.1

A0 = np.array([[2, -1],
              [-1, 1]])

A = (L - mu) / 8 * A0 + mu / 2 * np.identity(2)

b = np.array([-2, 0]).T * (L - mu) / 8


def f(x: np.array):
    return x @ A @ x.T + x @ b


def grad(x: np.array):
    return A @ x.T + b.T


def main():
    result = optimize_hs(0.0001, 0.0000001, 40, 2, f, grad)
    print(result['point'])


if __name__ == "__main__":
    main()