import numpy as np

from chargingstation.price_regularizer import PriceRegularizer


def _print_errors(N: int, A: np.ndarray, c: np.ndarray, reg: PriceRegularizer) -> None:
    nsamples = 1000
    err_feas = 0
    err_comp = 0
    for i in range(nsamples):
        b = 200 * (np.random.random((N,)) - 0.5)
        x_opt = reg.solve_price_regularization(A, b, c)
        err_feas += np.linalg.norm(A @ x_opt - b)
        err_comp += x_opt[:N] @ x_opt[N:]
    err_feas = err_feas / nsamples
    err_comp = err_comp / nsamples
    print(f"Average linear constraint error: {err_feas}")
    print(f"Average complementarity error  : {err_comp}")


def main() -> None:
    N = 12
    r = 24
    A = np.block([np.eye(N), -np.eye(N)])
    c = np.ones((r,))
    reg = PriceRegularizer(N, r)

    _print_errors(N, A, c, reg)


if __name__ == "__main__":
    main()
