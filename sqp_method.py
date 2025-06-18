import warnings
import casadi as ca
import numpy as np
from prettytable import PrettyTable


def ms_sqp_method(
    ms_setup,
    w,
    w0,
    armijo_c=0.01,
    armijo_rho=0.49,
    max_iter: int = 100,
    tol: float = 1e-8,
    save_history=False,
    print_level=3,
) -> dict:
    """
    Mirrors sqp_method but adjusted for multiple shooting.
    Implementation of the SQP method using the exact Hessian and Armijo backtracking. Solves

    min  F1(x)
    s.t. F2(x) = 0
         F3(x) >= 0

    :param ms_setup: dict with constraints and objectives
    :param w: decision variables in Multiple Shooting
    :param (ndarray, optional) w0: initial guess
    :param (float) armijo_c: Armijo parameter
    :param (float) armijo_rho: Armijo parameter
    :param (int, optional) max_iter: maximum number of iterations
    :param (float, optional) tol: convergence tolerance
    :param (bool, optional) save_history: return iterates
    :param (int) print_level: between 1-3
    """
    try:
        # get objective, constraints and variables
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

        # create functions, gradients and Hessian of Lagrangian
        f_fun = ca.Function("f_fun", [x], [f])
        g_fun = ca.Function("g_fun", [x], [g])
        f_grad = ca.jacobian(f, x)
        g_grad = ca.jacobian(g, x)
        f_grad_func = ca.Function("F1_grad", [x], [f_grad])
        g_grad_func = ca.Function("F2_grad", [x], [g_grad])

        lam = ca.MX.sym("lambda", m2)

        lagrangian = f + ca.dot(lam, g)

        H, L_grad = ca.hessian(lagrangian, x)
        lagrangian_grad_func = ca.Function("dL", [x, lam], [L_grad])
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
                        round(tk * np.linalg.norm(update), 8),
                    )
                )
            if norm_current_dL < tol or (tk * np.linalg.norm(update) < tol and i > 0):
                # stop if at optimum
                # during our test, the Lagrangian stagnated but was greater than the tolerence,
                # that is why we have the convergence criterium also on the update norm.
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
            # bound d by the space we have to the boundaries of x. We might be sometimes too
            # strict, since the update might get scaled down with line search but we expect to
            # take full steps close to the solution
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
                    # don't let tk become 0
                    break

            if save_history:
                history.append(
                    {"x": x_k, "f": f_fun(x_k), "g": g_fun(x_k), "lambda": lam_k}
                )

            x_k += tk * update
            x_k = np.clip(
                x_k, lbx, ubx
            )  # force x into bounds if it is outside (should not happen normally)
            lam_k = update_dual[
                :m2
            ].copy()  # to update lambda, we set lambda_k to the lambda of the subproblem

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
