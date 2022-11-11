from optimizer import optimize_one_dim


def f(x):
    return (x-4.0)**2 + 6.0


def main():
    point = optimize_one_dim(-10.0, 9.0, 1e-12, 2, f)
    print(help(optimize_one_dim))


if __name__ == "__main__":
    main()
