import casadi as ca
import numpy as np
from typing import Sequence, Tuple, Optional, Dict, Any

def ms_ocp(rhs      : Dict[str, ca.MX],
           nx       : int,
           N        : int,
           T        : float,
           tf,
           nu       : int,
           x0_sym   : Optional[ca.MX] = None) -> Dict[str, Any]:
    """
    Build the symbolic NLP for a generic multiple‑shooting OCP.

    INPUTS:
    rhs      : dict  - right-hand-side of the dynamical system (as casadi symbolic expression) e.g. {'x':x, 'p':u, 'ode':xdot, 'quad':L}
    nx       : int   - size of the states in the system
    N        : int   - number of multiple shooting intervals (one less than multiple shooting nodes)
    T        : float - time horizon from t0=0 until tf=T
    nu       : int   - size of control vector (at each point in time)
    x0_sym   : ca.MX - Optional symbolic variable that represents the *given* initial state. If None, 'X_0' is created as a decision variable.
    
    OUTPUTS:
    -------
    dict with keys
        'w'   : ca.MX (stack of all decision variables)
        'X'   : list of MX state symbols [X_0 … X_N]
        'U'   : list of MX control symbols [U_0 … U_{N-1}]
        'J'   : MX scalar cost
        'g'   : MX column vector of equality constraints (continuity only)
    """
    F = ca.integrator('F', 'cvodes', rhs, 0, T / N)

    # variables (note: if all of x0 is free, then here a symbolic variable is created)
    # however, if one component of x0 should be free and the others should be fixed,
    # give None and create the fix through constraints!!!!
    X = [x0_sym if x0_sym is not None else ca.MX.sym('X_0', nx)]
    U = []
    g = []
    # integral over the objective function
    J = 0

    # Do the multiple shooting
    for k in range(N):
        Uk = ca.MX.sym(f'U_{k}', nu)
        U.append(Uk)

        Fk = F(x0=X[k], p=ca.vertcat(Uk, tf))
        J += Fk['qf']

        Xk_next = ca.MX.sym(f'X_{k+1}', nx)
        g.append(Fk['xf'] - Xk_next)   # continuity
        X.append(Xk_next)

    # package the outputs. all X and U are decision variables.
    # the matching conditions are (as always) some of the later equality constraints
    w  = ca.vertcat(*(X + U + [tf]))
    g_ = ca.vertcat(*g) if g else ca.MX()

    return dict(w=w, X=X, U=U, J=J, g=g_)

# Short thoughts about this:
# The code is supposed to be usable for different OCP Examples. Since we want the user to be able to insert his or her
# own (in-)equality constraints, we can't handle this inside the ms_ocp() function (Reason being that the constraints could
# depend on variables introduced in the shooting procedure).
# We are well aware that this is some real ugly code...
def add_constraints(ms_data              : Dict[str, Any],
                    x_bounds             : Optional[Tuple[Sequence, Sequence]] = None,
                    u_bounds             : Optional[Tuple[Sequence, Sequence]] = None,
                    extra_eq             : Optional[Sequence[ca.MX]] = None,
                    extra_ineq           : Optional[Sequence[ca.MX]] = None,
                    x0_val               : Optional[Sequence] = None
                   ) -> Dict[str, Any]:
    """
    Derive bound vectors and augment equality / inequality constraints.

    Parameters
    ----------
    ms_data                : dict           - Output of ms_ocp().
    extra_eq, extra_ineq   : list[MX]       - Additional symbolic constraints (e.g. terminal or path constraints).
    x0_val                 : array‑like     - If the initial state is known, pass its numeric value here ‑‑ lb=ub=value.
    guess_center           : float          - Default initial guess for all free decision variables.
    x_bounds, u_bounds     : (lower, upper) - Each element can be:
                                                1) scalar --> same bound for every state / control component
                                                2) 1‑D array of length nx / nu --> component‑wise bounds

    Returns
    -------
    dict with keys
        'prob' : { 'f': J, 'x': w, 'g': g_all }
        'w0'   : list initial guess
        'lbx', 'ubx', 'lbg', 'ubg' : solver bound vectors
    """
    X, U, w = ms_data['X'], ms_data['U'], ms_data['w']
    nx, nu  = X[0].size1(), U[0].size1()
    N       = len(U)

    # containers foor bounds
    lbx, ubx, w0 = [], [], []

    # Here we use a helper function, that duplicates constraints if necessary. We thought
    # that it might be useful to have to type less, if the same constraint applied to
    # all states or all controls :)
    def _lower_expand(bound, dim, reps):
        """
        Convert a scalar / vector bound specification into a flat Python list
        of length `dim * reps`.
        """
        # Since every decision variable has to be given  with bounds assume unbounded, when nothing given
        if bound is None:
            return [-ca.inf] * (dim * reps)

        # don't allow casadi symbols yet :(
        if isinstance(bound, (ca.MX, ca.SX, ca.DM)):
            raise TypeError("Bounds must be numeric, not symbolic (got CasADi object).")

        arr = np.asarray(bound, dtype=float).flatten()
        if arr.size == 1:
            arr = np.repeat(arr, dim)
        if arr.size != dim:
            raise ValueError(f"bound length {arr.size} =/= dimension {dim}")
        return list(arr) * reps
    
    def _upper_expand(bound, dim, reps):
        """
        Convert a scalar / vector bound specification into a flat Python list
        of length `dim * reps`.
        """
        # Since every decision variable has to be given  with bounds assume unbounded, when nothing given
        if bound is None:
            return [ca.inf] * (dim * reps)

        # don't allow casadi symbols yet :(
        if isinstance(bound, (ca.MX, ca.SX, ca.DM)):
            raise TypeError("Bounds must be numeric, not symbolic (got CasADi object).")

        arr = np.asarray(bound, dtype=float).flatten()
        if arr.size == 1:
            arr = np.repeat(arr, dim)
        if arr.size != dim:
            raise ValueError(f"bound length {arr.size} =/= dimension {dim}")
        return list(arr) * reps

    # state bounds
    x_lb, x_ub = (None, None) if x_bounds is None else x_bounds
    lbx += _lower_expand(x_lb, nx, len(X))
    ubx += _upper_expand(x_ub, nx, len(X))

    # if initial state is fixed, overwrite the first nx entries (this becomes a pseudo decision)
    if x0_val is not None:
        lbx[:nx] = list(x0_val)
        ubx[:nx] = list(x0_val)

    # control bounds
    u_lb, u_ub = (None, None) if u_bounds is None else u_bounds
    lbx += _lower_expand(u_lb, nu, len(U))
    ubx += _upper_expand(u_ub, nu, len(U))

    # equality and inequality constraints
    g_all = [ms_data['g']]
    if extra_eq:
        g_all += [ca.vertcat(*extra_eq)]
    g_all = ca.vertcat(*g_all)

    h_all = ca.vertcat(*extra_ineq) if extra_ineq else ca.MX()

    # assemble lower and uppper bounds for g and h
    lbg = [0.0] * g_all.size1()
    ubg = [0.0] * g_all.size1()

    if h_all.is_empty():
        lbh, ubh = [], []
    else:
        # default: inequality form  h(x) <= 0
        lbh = [0.] * h_all.size1()
        ubh = [ca.inf]     * h_all.size1()

    g_total = ca.vertcat(g_all, h_all)
    lbg += lbh
    ubg += ubh

    # return
    prob = dict(f=ms_data['J'], x=w, g=g_total)

    lbx.append(0.1)     # min time horizon
    ubx.append(1000.0)   # max time horizon
    w0.append(50.0)     # initial guess


    return dict(prob=prob, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

def unpack_ms_solution(ms_data, sol):
    """
    extract solution data for plotting
    
    INPUTS:
    ms_data : dict                 - The dictionary returned by ms_ocp()
    sol     : CasAdi solver result - The dictionary obtained from solver().

    OUTPUTS:
    states : ndarray, shape = (N+1, nx)
    controls : ndarray, shape = (N,   nu)
    """
    w_opt = sol['x'].full().ravel()

    nx  = ms_data['X'][0].numel()
    nu  = ms_data['U'][0].numel()
    N   = len(ms_data['U'])

    split = (N + 1) * nx
    states   = w_opt[:split].reshape(N + 1, nx)
    controls = w_opt[split:].reshape(N,   nu)

    return states, controls