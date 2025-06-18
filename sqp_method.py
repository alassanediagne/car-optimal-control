import warnings
import casadi as ca
import numpy as np
from prettytable import PrettyTable


def sqp_method(
    F1_expr,
    F2_expr,
    F3_expr,
    x,
    lbx=None,
    ubx=None,
    x0: np.ndarray = None,
    armijo_c=0.01,
    armijo_rho=0.49,
    max_iter: int = 100,
    tol: float = 1e-8,
    save_history=False,
    print_level=0,
) -> dict:
    """
    Implementation of the SQP method using the exact Hessian and Armijo backtracking. Solves

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
        F1_xk, F2_xk, F3_xk = F1_fun(x_k), F2_fun(x_k), F3_fun(x_k)
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

        tk = 1
        # line search
        armijo1 = F1_fun(x_k + tk * update)
        armijo2 = F1_xk + armijo_c * tk * ca.dot(F1_grad_xk.T, update)
        while armijo1 > armijo2:
            tk *= armijo_rho
        if tk < tol:
            converged = True
            n_iter = i - 1
            print(f"SQP method converged in {n_iter} iterations")
            break

        if save_history or print_level:
            history.append(x_k)

        x_k += tk * update
        lam_k = update_dual[:m2].copy()
        mu_k = update_dual[m2:].copy()

        if np.linalg.norm(update) < tol:
            converged = True
            n_iter = i
            print(f"SQP method converged in {n_iter} iterations")
            break

    if not converged:
        warnings.warn(f"SQP method did not converge in {n_iter} iterations")

    if print_level:
        for h in history:
            table = PrettyTable()
            table.field_names("x", "f", "")

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


def ms_sqp_method(
    ms_setup,
    w,
    w0,
    armijo_c=0.01,
    armijo_rho=0.49,
    max_iter: int = 100,
    tol: float = 1e-8,
    save_history=False,
    print_level=2,
) -> dict:
    """
    Mirrors sqp_method but adjusted for multiple shooting.
    Implementation of the SQP method using the exact Hessian and Armijo backtracking. Solves

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
    try:
        f = ms_setup["prob"]["f"]
        g = ms_setup["prob"]["g"]
        x = w
        lbx = ms_setup["lbx"]
        ubx = ms_setup["ubx"]
        lbg = ms_setup["lbg"]
        ubg = ms_setup["ubg"]
        x0 = w0

        m1 = x.numel()
        m2 = g.numel()

        if x0 is None:
            x0 = np.zeros(m1)
        if lbx is None:
            lbx = -1 * ca.DM_inf(m1)
        if ubx is None:
            ubx = ca.DM_inf(m1)

        f_fun = ca.Function("f_fun", [x], [f])
        g_fun = ca.Function("g_fun", [x], [g])
        f_grad = ca.jacobian(f, x)
        g_grad = ca.jacobian(g, x)
        f_grad_func = ca.Function("F1_grad", [x], [f_grad])
        g_grad_func = ca.Function("F2_grad", [x], [g_grad])

        lam = ca.MX.sym("lambda", m2)

        lagrangian = f + ca.dot(lam, g)
        lagrangian_grad_func = ca.Function("dL", [x, lam], [ca.jacobian(lagrangian, x)])

        H, _ = ca.hessian(lagrangian, x)
        H_func = ca.Function("H", [x, lam], [H + ca.MX.eye(m1) * 1e-4])

        d = ca.MX.sym("d", m1)

        x_k = np.array(x0)
        lam_k = np.zeros(m2)
        converged = False
        n_iter = max_iter
        if save_history:
            history = []
        if print_level >= 2:
            table = PrettyTable()
            table.field_names = ["iteration", "f", "||∇L(x,λ)||", "||λ||", "||d||"]
        update = 0
        tk = 1
        for i in range(max_iter):
            f_xk, g_xk = f_fun(x_k), g_fun(x_k)
            f_grad_xk, g_grad_xk = (
                f_grad_func(x_k),
                g_grad_func(x_k),
            )
            H_xk = H_func(x_k, lam_k)
            norm_current_dL = np.linalg.norm(
                lagrangian_grad_func(x_k, lam_k).full().squeeze()
            )
            if print_level >= 2:
                print(f"SQP method: iteration {i}")
                table.add_row(
                    [
                        i,
                        f_fun(x_k),
                        norm_current_dL,
                        np.linalg.norm(lam_k),
                        tk * np.linalg.norm(update),
                    ]
                )
            if print_level >= 3:
                print(
                    "Iteration: {}, objective: {}, ||∇L(x,λ)||: {}, ||λ||: {}, ||d||: {}".format(
                        i,
                        round(f_fun(x_k).full().squeeze().item(), 8),
                        np.round(norm_current_dL, 8),
                        round(np.linalg.norm(lam_k), 3),
                        round(tk * np.linalg.norm(update),8),
                    )
                )
            if norm_current_dL < tol or (tk * np.linalg.norm(update) < tol and i > 0):
                # stop if at optimum
                converged = True
                n_iter = i
                if print_level >= 1:
                    print(f"SQP method converged in {n_iter} iterations")
                break

            objective = 0.5 * d.T @ H_xk @ d + ca.dot(f_grad_xk.T, d)
            constraints = g_grad_xk @ d + g_xk

            qp = {"f": objective, "g": constraints, "x": d}
            qp_solver = ca.qpsol(
                "sqp_qp_solver",
                "qpoases",
                qp,
                {"printLevel": "low", "terminationTolerance": tol},
            )
            lbx_d = lbx - x_k
            ubx_d = ubx - x_k
            sol = qp_solver(ubx=ubx_d, lbx=lbx_d, ubg=ubg, lbg=lbg)
            update = sol["x"].full().squeeze()
            update_dual = sol["lam_g"].full().squeeze()

            tk = 1
            # line search
            armijo1 = f_fun(x_k + tk * update)
            armijo2 = f_xk + armijo_c * tk * ca.dot(f_grad_xk.T, update)
            while armijo1 > armijo2:
                tk *= armijo_rho
                if tk < tol:
                    break

            if save_history:
                history.append(
                    {"x": x_k, "f": f_fun(x_k), "g": g_fun(x_k), "lambda": lam_k}
                )

            x_k += tk * update
            x_k = np.clip(
                x_k, lbx, ubx
            )  # force x into bounds if it is outside (should not happen normally)
            lam_k = update_dual[:m2].copy()

        if not converged and print_level >= 1:
            warnings.warn(f"SQP method did not converge in {n_iter} iterations")

        if print_level >= 2:
            print(table)

        out = {
            "x": x_k,
            "lambda": lam_k,
            "f": f_fun(x_k).full().item(),
            "g": g_fun(x_k).full().squeeze(),
            "n_iter": n_iter,
            "converged": converged,
        }

        if save_history:
            out["history"] = history

        return out

    except:
        print("QP solver failed")
