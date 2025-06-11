import casadi as ca
from casadi import exp, sin, cos, arctan
import numpy as np

class CarModel:
    def __init__(self):
        # define states
        states = ca.MX.sym('states',7)
        x, y, v, beta, psi, wz, delta = states[0], states[1], states[2], states[3], states[4], states[5], states[6]

        # define controls
        controls = ca.MX.sym('controls', 4)
        wdelta = controls[0]
        brake_force = controls[1]
        accelerator = controls[2]
        gear = controls[3]

        # parameters
        m = 1239
        g = 9.81
        lf = 1.19016
        lr = 1.37484
        eSP = 0.5
        R = 0.302
        Izz = 1752
        cw = 0.3
        rho = 1.249512
        A = 1.4378946874
        ig1, ig2, ig3, ig4, ig5 = 3.91, 2.002, 1.33, 1.0, 0.805
        it = 3.91
        Bf, Br, Cf, Cr, Df, Dr, Ef, Er = 10.96, 12.67, 1.3, 1.3, 4560.40, 3947.81, -0.5, -0.5
        fR0, fR1, fR4 = 0.009, 0.002, 0.0003
        B = 1.5 # car width
        # Aerodynamic forces
        FAx = 0.5 * cw * rho * A * v**2 
        FAy = 0.0
        gear = 3
        ig_mu = ca.if_else(gear<1.5, ig1,
                    ca.if_else(gear<2.5, ig2,
                    ca.if_else(gear<3.5, ig3,
                    ca.if_else(gear<4.5, ig4,
                        ig5))))
        
        dpsidt = wz

        # calculate slip angles
        alphaf = delta - arctan((lf*dpsidt - v*sin(beta))/(v*cos(beta)))
        alphar = arctan((lr*dpsidt + v*sin(beta))/(v*cos(beta)))
        # calculate lateral tire forces wear
        # Fsf depends on alphaf, Fsr depends on alphar
        Fsf = Df * sin(Cf*arctan(Bf*alphaf - Ef*(Bf*alphaf - arctan(Bf*alphaf))))
        Fsr = Dr * sin(Cr*arctan(Br*alphar - Er*(Br*alphar - arctan(Br*alphar))))

        # static tire loads at the front and rear wheel
        Fzf = m*lr*g / (lf+lr)
        Fzr = m*lf*g / (lf+lr)

        # front, rear breaking forces
        FBf = 2/3*brake_force
        FBr = 1/3*brake_force

        # friction coefficient
        fRv = fR0 + fR1 * v/100 + fR4 * (v/100)**4

        # rolling resistance forces
        FRf = fRv * Fzf
        FRr = fRv * Fzr

        # motor torque
        wmot = v * ig_mu * it / R

        f1 = 1 - exp(-3*accelerator)
        f2 = -37.8 + 1.54*wmot - 0.0019*wmot**2
        f3 = -34.9 - 0.04775*wmot 

        M_mot = f1*f2 + (1.0 - f1)*f3 
        M_wheel = ig_mu * it * M_mot
        Flr = M_wheel / R - FBr - FRr
        Flf = - FBf - FRf 

        dxdt = v * cos(psi - beta)
        dydt = v * sin(psi - beta)
        dvdt = 1/m * ( (Flr - FAx)*cos(beta) + Flf*cos(delta+beta) - (Fsr - FAy)*sin(beta) - Fsf*sin(delta + beta) ) 
        dbetadt = wz - 1/(m*v) * ( (Flr - FAx)*sin(beta) + Flf*sin(delta+beta) + (Fsr - FAy)*cos(beta) + Fsf*cos(delta + beta))
        dwzdt = 1/Izz * (Fsf*lf*cos(delta) - Fsr*lr - FAy*eSP + Flf*lf*sin(delta))
        ddeltadt = wdelta

        self.car_dynamics = ca.vertcat(
            dxdt,
            dydt,
            dvdt,
            dbetadt,
            dpsidt,
            dwzdt,
            ddeltadt
        )

        self.controls = controls
        self.states = states

    def get_system(self):
        return self.states, self.controls, self.car_dynamics
    
    def trajectory(self, controls, x0, t_grid):
        """
        Compute trajectory of car
        :param controls: controls (in this order!) steering angle velocity, breaking force, breaking_force, acceleration, gear
        :param x0: initial states (in this order!) x0, y0, v0, beta0, psi0, wz0, delta0
        :param t_grid: time grid
        """
        trajectory = []
        xk = x0

        if self.is_uniform_grid(t_grid):
            # if the grid is uniform, we only need one integrator and can save time and storage
            system = {'ode': self.car_dynamics, 'x': self.states, 'p': self.controls}
            integrator = ca.integrator(f'integrator', 'cvodes', system, 0, t_grid[1]-t_grid[0])
            for i in range(t_grid.size-1):
                res = integrator(x0=xk, p=controls[i])
                xk = res['xf']
                trajectory.append(xk)

        else:
            for i in range(t_grid.size-1):
                # for not uniform time grid
                t0, t1 = t_grid[i], t_grid[i+1]
                system = {'ode': self.car_dynamics, 'x': self.states, 'p': self.controls}
                integrator = ca.integrator(f'integrator_{i}', 'cvodes', system, t0, t1)
                res = integrator(x0=xk, p=controls[i])
                xk = res['xf']
                trajectory.append(xk)

        return np.array([np.array(x).squeeze() for x in trajectory])
    
    @staticmethod
    def is_uniform_grid(grid):
        diffs = grid[1:] - grid[:-1]
        return np.allclose(diffs, diffs[0], rtol = 1e-8)