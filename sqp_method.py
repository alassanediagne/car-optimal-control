import warnings
import casadi as ca
import numpy as np


def sqp_method(
    F1_expr,
    F2_expr,
    F3_expr,
    lbx=None,
    ubx=None,
    x0: np.ndarray = None,
    max_iter: int = 10,
    tol: float = 1e-8,
    save_history=False,
) -> dict:
    """
    Implementation of the SQP method using the exact Hessian. Solves

    min  F1(x)
    s.t. F2(x) = 0
         F3(x) >= 0

    :param F1_expr: objective function as Casadi expression
    :param F2_expr: equality constraints as Casadi expression
    :param F3_expr: inequality constraints as Casadi expression
    :param (ndarray, optional) x0: initial guess
    :param (int, optional) max_iter: maximum number of iterations
    :param (float, optional) tol: convergence tolerance
    :param (bool, optional) save_history: return iterates
    """
    x = ca.vertcat(*ca.symvar(F1_expr))

    m1 = x.numel()
    m2 = F2_expr.numel()
    m3 = F3_expr.numel()

    if x0 is None:
        x0 = np.zeros(m1)
    if lbx is None:
        lbx = -1 * ca.DM_inf(m1)
    if ubx is None:
        ubx = ca.DM_inf(m1)

    F1_fun = ca.Function("F1_fun", [x], [F1_expr])
    F2_fun = ca.Function("F2_fun", [x], [F2_expr])
    F3_fun = ca.Function("F3_fun", [x], [F3_expr])
    F1_grad = ca.jacobian(F1_expr, x)
    F2_grad = ca.jacobian(F2_expr, x)
    F3_grad = ca.jacobian(F3_expr, x)
    F1_grad_func = ca.Function("F1_grad", [x], [F1_grad])
    F2_grad_func = ca.Function("F2_grad", [x], [F2_grad])
    F3_grad_func = ca.Function("F3_grad", [x], [F3_grad])

    lam = ca.MX.sym("lambda", m2)
    mu = ca.MX.sym("mu", m3)

    lagrangian = F1_expr
    lagrangian += ca.dot(lam, F2_expr)
    lagrangian += ca.dot(mu, F3_expr)

    H, _ = ca.hessian(lagrangian, x)
    H_func = ca.Function("H", [x, lam, mu], [H])

    d = ca.MX.sym("d", m1)

    x_k = np.array(x0)
    lam_k = np.zeros(m2)
    mu_k = np.zeros(m3)
    converged = False
    n_iter = max_iter
    if save_history:
        history = []
    for i in range(1, max_iter + 1):
        F2_xk, F3_xk = F2_fun(x_k), F3_fun(x_k)
        print(type(F3_xk))
        F1_grad_xk, F2_grad_xk, F3_grad_xk = (
            F1_grad_func(x_k),
            F2_grad_func(x_k),
            F3_grad_func(x_k),
        )
        H_xk = H_func(x_k, lam_k, mu_k)

        objective = 0.5 * d.T @ H_xk @ d + ca.dot(F1_grad_xk.T, d)
        eq_constraints = F2_grad_xk @ d + F2_xk
        ineq_constraints = F3_grad_xk @ d + F3_xk

        qp = {"f": objective, "g": ca.vertcat(eq_constraints, ineq_constraints), "x": d}
        qp_solver = ca.qpsol("sqp_qp_solver", "qpoases", qp)
        sol = qp_solver(
            ubg=ca.vertcat(ca.GenDM_zeros(m2), ca.DM_inf(m3)),
            lbg=ca.vertcat(ca.GenDM_zeros(m2), ca.GenDM_zeros(m3)),
        )
        update = sol["x"].full().squeeze()
        update_dual = sol["lam_g"].full().squeeze()
        x_k += update
        lam_k = update_dual[:m2].copy()
        mu_k = update_dual[m2:].copy()

        if save_history:
            history.append(x_k)

        if np.linalg.norm(update) < tol:
            converged = True
            n_iter = i
            print(f"SQP method converged in {n_iter} iterations")
            break

    if not converged:
        warnings.warn(f"SQP method did not converge in {n_iter} iterations")

    out = {
        "x": x_k,
        "lambda": lam_k,
        "mu": mu_k,
        "F1": F1_fun(x_k).full().item(),
        "F2": F2_fun(x_k).full().squeeze(),
        "F3": F3_fun(x_k).full().squeeze(),
        "n_iter": n_iter,
        "converged": converged,
    }

    if save_history:
        out["history"] = history

    return out
