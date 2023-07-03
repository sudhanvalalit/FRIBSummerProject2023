import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def RK4(f, x0, t0, te, N):
    h = (te - t0) / N
    times = np.arange(t0, te + h, h)
    solutionx, solutiony = [], []
    x = x0
    for t in times:
        solutionx.append(x[0])
        solutiony.append(x[1])
        k1 = h * f(x, t)
        k2 = h * f(x + 0.5 * k1, t + 0.5 * h)
        k3 = h * f(x + 0.5 * k2, t + 0.5 * h)
        k4 = h * f(x + k3, t + h)
        x += (k1 + 2 * (k2 + k3) + k4) / 6
    return times, np.array([solutionx, solutiony], float)


def func(xvec, t, *args):
    x, y = xvec[0], xvec[1]
    k1, k2 = *args
    # f1 = x - y - x * (x ** 2 + 5.0 * y ** 2)
    # f2 = x + y - y * (x ** 2 + y ** 2)

    f1 = y
    f2 = energy -k1*x**2 -k2*x**4
    return np.array([f1, f2], float)


def main():
    N1 = 200
    t0 = 0.0
    te = 20.0
    # Inside the trapped region
    xInput = np.array([0.0, 1.2])

    time1, result = RK4(func, xInput, t0, te, N1)
    # Lower than the trapped region
    x1Input = np.array([0.0, 1e-3])
    time2, result1 = RK4(func, x1Input, t0, te, N1)

    # Initial value higher than trapped region
    x2Input = np.array([0.0, 2.0])
    time3, result2 = RK4(func, x2Input, t0, te, N1)

    fig = plt.figure(figsize=(8, 6), dpi=100)
    plt.plot(result[0, :], result[1, :], "k-", label="y(0) = 1.2")
    plt.plot(result1[0, :], result1[1, :], "r-", label="y(0) = 0.001")
    plt.plot(result2[0, :], result2[1, :], "b-", label="y(0) = 2.0")
    plt.legend()

    plt.xlabel("x(t)", fontsize=16)
    plt.ylabel("y(t)", fontsize=16)
    plt.legend()
    # plt.show()
    plt.savefig("Poincare-Bendixson.pdf")


if __name__ == "__main__":
    main()
