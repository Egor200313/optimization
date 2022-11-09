from optimizer import optimize_one_dim


def f(x):
    return (x-4.0)**2 + 6.0


def main():
    point = optimize_one_dim(-10.0, 9.0, 0.00000001, 100, f)
    print(point)


if __name__ == "__main__":
    main()
